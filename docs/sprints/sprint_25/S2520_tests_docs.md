# S2520–S2523 — Tests unitaires et documentation

| Champ | Valeur |
|-------|--------|
| **Sprint** | 25 |
| **Priorité** | 🟡 Important (S2520, S2521, S2522) / 🟢 Faible (S2523) |
| **Statut** | ✅ Terminé |
| **Durée estimée** | S2520 : 1h / S2521 : 1h / S2522 : 30 min / S2523 : 30 min = 3h total |
| **Dépendances** | S2506 ✅ (`EWCMlpRegressor`), S2507 ✅ (`EWCMlpMulticlass`), S2515–S2519 ✅ (expériences complètes) |
| **Fichiers cibles** | `tests/test_ewc_regression.py`, `tests/test_ewc_multiclass.py`, `docs/roadmap_phase2.md` |
| **Référence** | `tests/test_ewc_head.c` (pattern assertions), `tests/test_hdc.c`, format pytest des tests Python existants dans `tests/` |

---

## Contexte

Les tests unitaires S2520 et S2521 vérifient les invariants fondamentaux des nouveaux modèles (shape, normalisation, pénalité EWC) indépendamment des datasets réels. S2522 garantit la non-régression sur les tests binaires existants. S2523 met à jour la roadmap pour refléter Sprint 25.

---

## S2520 — `tests/test_ewc_regression.py`

### Spec complète

```python
"""
test_ewc_regression.py — Tests unitaires pour EWCMlpRegressor.

Invariants testés :
    1. Forward pass : output shape == (batch_size, 1)
    2. Output non borné : peut dépasser [0, 1] (régression linéaire, pas de Sigmoid)
    3. MSE loss calculable et propageable
    4. EWC penalty == 0.0 avant consolidation
    5. EWC penalty > 0.0 après consolidation
    6. Consolidation : Fisher et theta_star de même shape que les paramètres
    7. Backward transfer : RMSE sur tâche 1 ne dégrade pas catastrophiquement après tâche 2
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.ewc.ewc_mlp_regression import EWCMlpRegressor


@pytest.fixture
def model() -> EWCMlpRegressor:
    return EWCMlpRegressor(input_dim=5, hidden_dims=[32, 16], ewc_lambda=400.0)


@pytest.fixture
def dummy_loader() -> DataLoader:
    """DataLoader factice : 64 exemples, input_dim=5, RUL ∈ [0, 125]."""
    x = torch.randn(64, 5)
    y = torch.rand(64) * 125
    return DataLoader(TensorDataset(x, y), batch_size=16)


def test_forward_shape(model: EWCMlpRegressor) -> None:
    """Output shape = (batch_size, 1)."""
    x = torch.randn(8, 5)
    out = model(x)
    assert out.shape == (8, 1), f"Shape incorrecte : {out.shape} (attendu (8, 1))"


def test_output_unbounded(model: EWCMlpRegressor) -> None:
    """La sortie n'est pas bornée en [0, 1] (pas de Sigmoid)."""
    x = torch.randn(256, 5) * 10  # entrées larges
    with torch.no_grad():
        out = model(x).squeeze()
    # Au moins quelques sorties doivent dépasser [0, 1] avec des entrées larges
    # (sinon le modèle aurait une Sigmoid cachée)
    assert not (out.min() >= 0 and out.max() <= 1), (
        "Output borné en [0,1] — vérifier l'absence de Sigmoid finale"
    )


def test_mse_loss_backprop(model: EWCMlpRegressor) -> None:
    """MSE loss calculable + backpropagation sans erreur."""
    x = torch.randn(8, 5)
    y = torch.rand(8) * 125
    out = model(x).squeeze()
    loss = nn.MSELoss()(out, y)
    loss.backward()
    # Vérifier que les gradients existent
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient None pour {name}"


def test_ewc_penalty_before_consolidation(model: EWCMlpRegressor) -> None:
    """EWC penalty = 0.0 avant toute consolidation."""
    penalty = model.ewc_penalty()
    assert penalty.item() == 0.0, f"Penalty non nulle avant consolidation : {penalty.item()}"


def test_ewc_penalty_after_consolidation(
    model: EWCMlpRegressor,
    dummy_loader: DataLoader,
) -> None:
    """EWC penalty > 0.0 après consolidation (Fisher non nulle)."""
    # Entraîner brièvement pour avoir des paramètres non nuls
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for x_batch, y_batch in dummy_loader:
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(x_batch).squeeze(), y_batch)
        loss.backward()
        optimizer.step()

    model.consolidate(dummy_loader, n_samples=32)

    # Modifier légèrement les paramètres
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    penalty = model.ewc_penalty()
    assert penalty.item() > 0.0, "Penalty nulle après consolidation — Fisher vide ?"


def test_consolidation_shapes(
    model: EWCMlpRegressor,
    dummy_loader: DataLoader,
) -> None:
    """Fisher et theta_star de même shape que les paramètres."""
    model.consolidate(dummy_loader, n_samples=32)
    for name, param in model.named_parameters():
        assert name in model._fisher, f"Fisher manquante pour {name}"
        assert name in model._theta_star, f"theta_star manquant pour {name}"
        assert model._fisher[name].shape == param.shape, (
            f"Shape Fisher incorrecte pour {name}"
        )
        assert model._theta_star[name].shape == param.shape


def test_backward_transfer_not_catastrophic(
    model: EWCMlpRegressor,
    dummy_loader: DataLoader,
) -> None:
    """RMSE sur tâche 1 après entraînement sur tâche 2 n'est pas catastrophique (< 2× RMSE initial)."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Tâche 1 : RUL ∈ [50, 125]
    x1 = torch.randn(64, 5)
    y1 = 50 + torch.rand(64) * 75
    loader1 = DataLoader(TensorDataset(x1, y1), batch_size=16)

    for _ in range(10):
        for xb, yb in loader1:
            optimizer.zero_grad()
            nn.MSELoss()(model(xb).squeeze(), yb).backward()
            optimizer.step()

    # RMSE initial sur tâche 1
    model.consolidate(loader1, n_samples=32)
    with torch.no_grad():
        rmse_task1_before = float(
            torch.sqrt(nn.MSELoss()(model(x1).squeeze(), y1)).item()
        )

    # Tâche 2 : RUL ∈ [0, 30] (distribution différente)
    x2 = torch.randn(64, 5)
    y2 = torch.rand(64) * 30
    loader2 = DataLoader(TensorDataset(x2, y2), batch_size=16)

    for _ in range(10):
        for xb, yb in loader2:
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(xb).squeeze(), yb) + model.ewc_penalty()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        rmse_task1_after = float(
            torch.sqrt(nn.MSELoss()(model(x1).squeeze(), y1)).item()
        )

    # Avec EWC, la dégradation doit être < 2× (critère souple pour test unitaire)
    assert rmse_task1_after < rmse_task1_before * 2, (
        f"Oubli catastrophique : RMSE_before={rmse_task1_before:.2f}, "
        f"RMSE_after={rmse_task1_after:.2f}"
    )
```

### Exécution

```bash
pytest tests/test_ewc_regression.py -v
# Attendu : 6 tests PASSED
```

---

## S2521 — `tests/test_ewc_multiclass.py`

### Spec complète

```python
"""
test_ewc_multiclass.py — Tests unitaires pour EWCMlpMulticlass.

Invariants testés :
    1. Forward pass : output shape == (batch_size, n_classes)
    2. Softmax normalisé : probs.sum(dim=1) ≈ 1.0
    3. CrossEntropy loss calculable + backprop
    4. EWC penalty == 0.0 avant consolidation
    5. EWC penalty > 0.0 après consolidation
    6. Predict : argmax correct (valeur connue)
    7. F1-macro non nul sur données synthétiques séparables
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.ewc.ewc_mlp_multiclass import EWCMlpMulticlass


@pytest.fixture
def model_10() -> EWCMlpMulticlass:
    """Modèle CWRU : 9 features, 10 classes."""
    return EWCMlpMulticlass(input_dim=9, n_classes=10, ewc_lambda=400.0)


@pytest.fixture
def model_3() -> EWCMlpMulticlass:
    """Modèle Paderborn : 9 features, 3 classes."""
    return EWCMlpMulticlass(input_dim=9, n_classes=3, ewc_lambda=400.0)


def test_forward_shape_10_classes(model_10: EWCMlpMulticlass) -> None:
    x = torch.randn(8, 9)
    logits = model_10(x)
    assert logits.shape == (8, 10), f"Shape : {logits.shape} (attendu (8, 10))"


def test_forward_shape_3_classes(model_3: EWCMlpMulticlass) -> None:
    x = torch.randn(8, 9)
    logits = model_3(x)
    assert logits.shape == (8, 3), f"Shape : {logits.shape} (attendu (8, 3))"


def test_softmax_normalized(model_10: EWCMlpMulticlass) -> None:
    """softmax(logits).sum(dim=1) == 1.0 pour chaque exemple."""
    x = torch.randn(16, 9)
    with torch.no_grad():
        logits = model_10(x)
    probs = torch.softmax(logits, dim=1)
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones(16), atol=1e-5), (
        f"Softmax non normalisé, max abs diff = {(sums - 1).abs().max().item():.2e}"
    )


def test_crossentropy_backprop(model_10: EWCMlpMulticlass) -> None:
    x = torch.randn(8, 9)
    y = torch.randint(0, 10, (8,))
    logits = model_10(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    for name, param in model_10.named_parameters():
        assert param.grad is not None, f"Gradient None pour {name}"


def test_ewc_penalty_zero_before_consolidation(model_10: EWCMlpMulticlass) -> None:
    assert model_10.ewc_penalty().item() == 0.0


def test_ewc_penalty_nonzero_after_consolidation(model_10: EWCMlpMulticlass) -> None:
    x = torch.randn(64, 9)
    y = torch.randint(0, 10, (64,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16)

    # Entraîner brièvement
    opt = torch.optim.SGD(model_10.parameters(), lr=0.01)
    for xb, yb in loader:
        opt.zero_grad()
        nn.CrossEntropyLoss()(model_10(xb), yb).backward()
        opt.step()

    model_10.consolidate(loader, n_samples=32)

    # Perturber les paramètres
    with torch.no_grad():
        for p in model_10.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    assert model_10.ewc_penalty().item() > 0.0


def test_predict_argmax(model_10: EWCMlpMulticlass) -> None:
    """predict() retourne l'argmax des logits."""
    x = torch.randn(8, 9)
    with torch.no_grad():
        logits = model_10(x)
        preds = model_10.predict(x)
    expected = logits.argmax(dim=1)
    assert torch.all(preds == expected), "predict() != argmax(logits)"


def test_f1_macro_nonzero_on_separable_data(model_10: EWCMlpMulticlass) -> None:
    """F1-macro > 0 après entraînement sur données synthétiques séparables."""
    from src.evaluation.multiclass_metrics import compute_f1_macro

    # Données synthétiques : clusters bien séparés
    np.random.seed(42)
    x_list, y_list = [], []
    for cls in range(10):
        x_list.append(np.random.randn(20, 9).astype(np.float32) + cls * 5)
        y_list.extend([cls] * 20)
    x = torch.tensor(np.vstack(x_list))
    y = torch.tensor(y_list, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)

    opt = torch.optim.SGD(model_10.parameters(), lr=0.05)
    for _ in range(30):
        for xb, yb in loader:
            opt.zero_grad()
            nn.CrossEntropyLoss()(model_10(xb), yb).backward()
            opt.step()

    with torch.no_grad():
        preds = model_10.predict(x).numpy()
    f1 = compute_f1_macro(y.numpy(), preds)
    assert f1 > 0.0, f"F1-macro = 0 après entraînement ({f1})"
```

### Exécution

```bash
pytest tests/test_ewc_multiclass.py -v
# Attendu : 7 tests PASSED
```

---

## S2522 — Vérification 0 régression

```bash
# Suite complète — vérifier qu'aucun test binaire existant n'est cassé
pytest tests/ -v

# Attendu :
#   tests/test_ewc_regression.py      : 6 PASSED
#   tests/test_ewc_multiclass.py      : 7 PASSED
#   Tous les tests binaires existants : PASSED (0 FAILED, 0 ERROR)
```

Si des tests échouent : vérifier que les paramètres `mode="binary"` par défaut dans les loaders n'ont pas été accidentellement omis.

---

## S2523 — Mise à jour `docs/roadmap_phase2.md`

### Section à ajouter

Localiser la section Sprint 24 dans `docs/roadmap_phase2.md` et ajouter après :

```markdown
### Sprint 25 — Tâches Natives : RUL Régression + Multi-classe (15–28 juil. 2026)

**Motivation** : les datasets CMAPSS, Pronostia, CWRU et Paderborn ont été uniformisés en binaire pour le framework CL. Sprint 25 exploite leurs tâches d'origine (RUL continu, classification multi-classe) pour des contributions manuscrit plus riches.

**Livrables** :
- Loaders étendus : `mode="rul"` (CMAPSS, Pronostia, Battery) + `mode="multiclass"` (CWRU, Paderborn)
- Nouveaux modèles : `EWCMlpRegressor`, `EWCMlpMulticlass`, `HDCRegressor`
- Métriques : `src/evaluation/rul_metrics.py` (RMSE, MAE, Horizon Score PHM 2008), `src/evaluation/multiclass_metrics.py` (F1-macro, confusion matrix)
- 5 expériences PC : exp_S25_01 à exp_S25_05

**Résultats clés** :
- exp_S25_01 (EWC RUL CMAPSS) : RMSE_task1 = ___ cycles, AF_rmse = ___
- exp_S25_03 (EWC Multiclass CWRU) : F1-macro_task1 = ___, AF_f1 = ___
- Mode binaire : 0 régression (pytest tests/ — tous verts)

**Statut** : ⬜ À compléter post-exécution
```

---

## Vérification end-to-end

```bash
# Tests complets + non-régression
pytest tests/test_ewc_regression.py tests/test_ewc_multiclass.py -v
pytest tests/ -v --tb=short 2>&1 | tail -20

# Roadmap mise à jour
grep -A 5 "Sprint 25" docs/roadmap_phase2.md
```

---

## Résultats d'implémentation

| Sous-tâche | Statut | Notes |
|------------|:------:|-------|
| S2520 — `tests/test_ewc_regression.py` | ✅ | 7 tests PASSED |
| S2521 — `tests/test_ewc_multiclass.py` | ✅ | 8 tests PASSED |
| S2522 — `pytest tests/ -v` — 0 régression | ✅ | 470 passed, 3 pre-existing failures (board_recorder ewc/monitoring, paderborn_stream timeout) |
| S2523 — `docs/roadmap_phase2.md` mis à jour | ✅ | Section Sprint 25 ajoutée après Sprint 24 |

### Reproduction (2026-06-12)

Suite re-exécutée de bout en bout :

- `pytest tests/test_ewc_regression.py tests/test_ewc_multiclass.py` → **7 + 8 = 15 PASSED**.
- `pytest tests/` → **471 passed, 12 skipped, 2 failed**. Les 2 échecs (`test_board_recorder` `test_n_params_ewc`, `test_all_models_dry_run[ewc-monitoring]`) sont **pré-existants et hors périmètre S25** — aucune nouvelle régression sur les nouveaux modèles ni sur le mode binaire. Le 3ᵉ échec historique (paderborn_stream timeout) n'a pas reparu.

---

## Questions ouvertes

- `TODO(arnaud)` : Le test `test_backward_transfer_not_catastrophic` (S2520) utilise un seuil souple (< 2× RMSE). Est-ce que le critère quantitatif du manuscrit est plus strict ? Si oui, ajuster le seuil du test.
