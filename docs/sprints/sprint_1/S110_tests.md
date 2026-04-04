# S1-10 — Tests unitaires modèle EWC + métriques

| Champ | Valeur |
|-------|--------|
| **ID** | S1-10 |
| **Sprint** | Sprint 1 — Semaine 1 (15–22 avril 2026) |
| **Priorité** | ✅ Terminé |
| **Durée estimée** | 2h |
| **Dépendances** | S1-04 (`ewc_mlp.py`), S1-05 (`fisher.py`), S1-06 (`baselines.py`), S1-07 (`metrics.py`), S1-08 (`memory_profiler.py`) |
| **Fichiers cibles** | `tests/test_ewc.py`, `tests/conftest.py` |

---

## Objectif

Écrire les tests unitaires couvrant les composants critiques du Sprint 1 :

1. **`EWCMlpClassifier`** — forward pass, perte EWC, snapshot θ*, nombre de paramètres
2. **`compute_fisher_diagonal`** — valeurs non-nulles, forme correcte
3. **`compute_cl_metrics`** — AA/AF/BWT sur une matrice 3×3 aux valeurs connues
4. **`memory_profiler`** — smoke test : RAM mesurée > 0, budget respecté pour EWC MLP

Les tests doivent être rapides (<10 s total) et ne pas nécessiter le dataset réel (utiliser des tenseurs synthétiques).

**Critère de succès** : `pytest tests/test_ewc.py -v` — tous les tests passent en < 10 s.

---

## Sous-tâches

### 1. Créer `tests/conftest.py` (fixtures partagées)

```python
# tests/conftest.py
import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from src.models.ewc import EWCMlpClassifier


@pytest.fixture
def model():
    """EWCMlpClassifier avec paramètres par défaut."""
    return EWCMlpClassifier()


@pytest.fixture
def synthetic_loader():
    """DataLoader synthétique 3 tâches × 64 exemples, input_dim=6."""
    def make_loader(n=64, seed=42):
        torch.manual_seed(seed)
        X = torch.randn(n, 6)
        y = torch.randint(0, 2, (n, 1)).float()
        return DataLoader(TensorDataset(X, y), batch_size=32)
    return make_loader


@pytest.fixture
def cl_tasks(synthetic_loader):
    """Liste de 3 tâches CL synthétiques (format conforme à monitoring_dataset.py)."""
    return [
        {"task_id": i + 1, "domain": d,
         "train_loader": synthetic_loader(seed=i),
         "val_loader": synthetic_loader(n=32, seed=i + 10)}
        for i, d in enumerate(["Pump", "Turbine", "Compressor"])
    ]


@pytest.fixture
def ewc_config():
    """Config EWC minimale pour les tests (sans accès au YAML réel)."""
    return {
        "model": {"input_dim": 6, "hidden_dims": [32, 16], "dropout": 0.2},
        "training": {
            "optimizer": "sgd", "learning_rate": 0.01,
            "momentum": 0.9, "epochs_per_task": 2, "batch_size": 32,
        },
        "ewc": {"lambda": 1000, "gamma": 0.9, "n_fisher_samples": 32},
    }


@pytest.fixture
def known_acc_matrix():
    """
    Matrice 3×3 avec valeurs connues pour tester les formules de métriques.

    acc_matrix[i, j] = accuracy sur tâche j après tâche i.
    Simule un oubli catastrophique modéré.
    """
    return np.array([
        [0.91, np.nan, np.nan],
        [0.88, 0.85,  np.nan],
        [0.86, 0.83,  0.89 ],
    ])
```

### 2. Tests `EWCMlpClassifier`

```python
# tests/test_ewc.py — Partie 1 : modèle

import torch
import numpy as np
import pytest
from src.models.ewc import EWCMlpClassifier


class TestEWCMlpClassifier:

    def test_forward_shape(self, model):
        """Le forward pass doit retourner [batch, 1] avec valeurs ∈ [0, 1]."""
        x = torch.randn(32, 6)
        out = model(x)
        assert out.shape == (32, 1), f"Shape attendue (32,1), obtenue {out.shape}"
        assert (out >= 0).all() and (out <= 1).all(), "Sigmoid doit borner [0,1]"

    def test_n_params(self, model):
        """769 paramètres attendus (conforme à ewc_mlp_spec.md §2)."""
        n = sum(p.numel() for p in model.parameters())
        assert n == 769, f"Attendu 769 params, obtenu {n}"

    def test_ewc_loss_task1_is_bce(self, model):
        """Sur Task 1 (fisher=None), la perte doit être pure BCE."""
        x = torch.randn(16, 6)
        y = torch.randint(0, 2, (16, 1)).float()
        loss = model.ewc_loss(x, y, fisher=None, theta_star=None, ewc_lambda=1000.0)
        assert loss.item() > 0, "BCE doit être positive"
        assert loss.requires_grad, "La perte doit être différentiable"

    def test_ewc_loss_increases_with_lambda(self, model):
        """La régularisation EWC doit croître avec λ."""
        x = torch.randn(16, 6)
        y = torch.randint(0, 2, (16, 1)).float()
        fisher = {n: torch.ones_like(p) for n, p in model.named_parameters()}
        theta_star = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        loss_low  = model.ewc_loss(x, y, fisher, theta_star, ewc_lambda=1.0)
        loss_high = model.ewc_loss(x, y, fisher, theta_star, ewc_lambda=10_000.0)
        assert loss_high.item() > loss_low.item(), "Perte EWC doit croître avec λ"

    def test_theta_star_detached(self, model):
        """Le snapshot θ* doit être détaché du graphe de calcul."""
        theta_star = model.get_theta_star()
        for name, tensor in theta_star.items():
            assert not tensor.requires_grad, f"{name} ne devrait pas avoir requires_grad"

    def test_backprop(self, model):
        """Un backward pass complet doit mettre à jour les gradients."""
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(8, 6)
        y = torch.randint(0, 2, (8, 1)).float()
        optimizer.zero_grad()
        loss = model.ewc_loss(x, y, None, None, 0.0)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "Les gradients doivent être calculés après backward()"
```

### 3. Tests Fisher

```python
class TestFisherDiagonal:

    def test_fisher_shape(self, model, synthetic_loader):
        """Fisher doit avoir la même structure de clés que model.named_parameters()."""
        from src.models.ewc.fisher import compute_fisher_diagonal
        loader = synthetic_loader()
        fisher = compute_fisher_diagonal(model, loader, device="cpu", n_samples=32)
        param_names = {n for n, _ in model.named_parameters()}
        assert set(fisher.keys()) == param_names

    def test_fisher_non_negative(self, model, synthetic_loader):
        """La Fisher diagonale doit être non-négative (carrés de gradients)."""
        from src.models.ewc.fisher import compute_fisher_diagonal
        loader = synthetic_loader()
        fisher = compute_fisher_diagonal(model, loader, device="cpu", n_samples=32)
        for name, f in fisher.items():
            assert (f >= 0).all(), f"Fisher[{name}] contient des valeurs négatives"

    def test_online_fisher_decay(self, model, synthetic_loader):
        """L'update Online doit interpoler entre l'ancienne et la nouvelle Fisher."""
        from src.models.ewc.fisher import compute_fisher_diagonal, update_online_fisher
        loader = synthetic_loader()
        f1 = compute_fisher_diagonal(model, loader, device="cpu", n_samples=32)
        f2 = compute_fisher_diagonal(model, loader, device="cpu", n_samples=32)
        f_online = update_online_fisher(f1, f2, gamma=0.9)
        for name in f1:
            expected = 0.9 * f1[name] + f2[name]
            assert torch.allclose(f_online[name], expected, atol=1e-5)
```

### 4. Tests `compute_cl_metrics`

```python
class TestCLMetrics:

    def test_aa_known_matrix(self, known_acc_matrix):
        """AA = moyenne de la dernière ligne."""
        from src.evaluation.metrics import compute_cl_metrics
        metrics = compute_cl_metrics(known_acc_matrix)
        expected_aa = (0.86 + 0.83 + 0.89) / 3
        assert abs(metrics["aa"] - expected_aa) < 1e-4, f"AA attendu {expected_aa:.4f}"

    def test_af_positive_for_forgetting(self, known_acc_matrix):
        """AF doit être positif (oubli) pour cette matrice."""
        from src.evaluation.metrics import compute_cl_metrics
        metrics = compute_cl_metrics(known_acc_matrix)
        assert metrics["af"] > 0, "AF doit être > 0 (oubli présent dans cette matrice)"

    def test_bwt_negative_for_forgetting(self, known_acc_matrix):
        """BWT doit être négatif pour une matrice présentant de l'oubli."""
        from src.evaluation.metrics import compute_cl_metrics
        metrics = compute_cl_metrics(known_acc_matrix)
        assert metrics["bwt"] < 0, "BWT doit être < 0 (oubli catastrophique)"

    def test_metrics_keys_present(self, known_acc_matrix):
        """Toutes les clés obligatoires doivent être présentes."""
        from src.evaluation.metrics import compute_cl_metrics
        metrics = compute_cl_metrics(known_acc_matrix)
        for key in ["aa", "af", "bwt", "fwt", "n_tasks", "acc_matrix"]:
            assert key in metrics, f"Clé manquante : {key}"

    def test_save_metrics_creates_file(self, tmp_path, known_acc_matrix):
        """save_metrics() doit créer le fichier JSON."""
        from src.evaluation.metrics import compute_cl_metrics, save_metrics
        metrics = compute_cl_metrics(known_acc_matrix)
        output = tmp_path / "results" / "metrics.json"
        save_metrics(metrics, str(output))
        assert output.exists(), "Le fichier JSON doit être créé"
```

### 5. Tests `memory_profiler`

```python
class TestMemoryProfiler:

    def test_forward_profile_non_zero(self, model):
        """Le pic RAM mesuré doit être > 0."""
        from src.evaluation.memory_profiler import profile_forward_pass
        result = profile_forward_pass(model, input_shape=(1, 6), n_runs=5)
        assert result["ram_peak_bytes"] > 0

    def test_ewc_within_budget(self, model):
        """EWCMlpClassifier doit respecter le budget 64 Ko."""
        from src.evaluation.memory_profiler import profile_forward_pass
        result = profile_forward_pass(model, input_shape=(1, 6), n_runs=5)
        assert result["within_budget_64ko"], (
            f"RAM peak {result['ram_peak_bytes']} B dépasse 65 536 B"
        )

    def test_n_params_matches_model(self, model):
        """n_params dans le rapport doit correspondre au modèle."""
        from src.evaluation.memory_profiler import profile_forward_pass
        result = profile_forward_pass(model, input_shape=(1, 6), n_runs=5)
        expected = sum(p.numel() for p in model.parameters())
        assert result["n_params"] == expected
```

---

## Critères d'acceptation

- [x] `tests/conftest.py` — fixtures `model`, `cl_tasks`, `ewc_config`, `known_acc_matrix` disponibles
- [x] `pytest tests/test_ewc.py::TestEWCMlpClassifier -v` — tous les tests passent
- [x] `pytest tests/test_ewc.py::TestFisherDiagonal -v` — tous les tests passent
- [x] `pytest tests/test_ewc.py::TestCLMetrics -v` — tous les tests passent
- [x] `pytest tests/test_ewc.py::TestMemoryProfiler -v` — tous les tests passent
- [x] Durée totale < 10 s — **1.32 s mesuré** (28 tests dans test_ewc.py, 71 tests total)
- [x] `pytest tests/ -v` global — aucune régression (71/71 passés)
- [ ] `ruff check tests/test_ewc.py` et `black --check` passent

## Notes d'implémentation

**Date de complétion** : 4 avril 2026

**Écarts par rapport à la spec :**

- `update_online_fisher` dans la spec → nom réel : `update_fisher_online` (fisher.py) — test adapté
- `conftest.py` : `cl_tasks` dépend de `synthetic_loader` (fixture factory, input_dim=6) ; clé `domain` ajoutée
- `test_baselines.py` (hors scope S1-10) mis à jour : `input_dim=4→6` pour cohérence avec le nouveau `cl_tasks`

**Résultat final :**

```text
pytest tests/test_ewc.py -v   →  28 passed in 1.32s
pytest tests/ -v              →  71 passed in 1.54s
```

---

## Questions ouvertes

- `TODO(arnaud)` : ajouter un test d'intégration end-to-end (S1-09 complet sur données synthétiques) dans ce fichier ou dans un fichier `tests/test_integration_ewc.py` séparé ?
- `TODO(arnaud)` : faut-il tester la reproductibilité (`set_seed(42)`) — deux runs avec la même seed doivent donner la même `acc_matrix` ?
