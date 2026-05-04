# S14-11 — Tests unitaires `EWCOneClassDetector` + `DBSCANDetector`

| Champ | Valeur |
|-------|--------|
| **ID** | S14-11 |
| **Sprint** | Sprint 14 |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 2h |
| **Dépendances** | S14-01, S14-03 |
| **Fichiers cibles** | `tests/test_ewc_oneclass.py`, `tests/test_dbscan_detector.py` |

---

## Objectif

Valider le comportement de `EWCOneClassDetector` et `DBSCANDetector` via des tests unitaires sur données synthétiques (sans accès aux datasets réels).

---

## Tests `EWCOneClassDetector`

```python
# tests/test_ewc_oneclass.py

import numpy as np
import pytest
from src.models.ewc.ewc_oneclass import EWCOneClassDetector

@pytest.fixture
def detector():
    return EWCOneClassDetector(input_dim=4, hidden_dim=16, latent_dim=4, n_epochs=3)

@pytest.fixture
def X_normal():
    rng = np.random.default_rng(42)
    return rng.standard_normal((50, 4)).astype(np.float32)

def test_fit_task_runs_without_error(detector, X_normal):
    detector.fit_task(X_normal)

def test_predict_score_shape(detector, X_normal):
    detector.fit_task(X_normal)
    scores = detector.predict_score(X_normal)
    assert scores.shape == (50,)
    assert (scores >= 0).all()

def test_predict_binary_output(detector, X_normal):
    detector.fit_task(X_normal)
    labels = detector.predict(X_normal)
    assert set(labels).issubset({0, 1})

def test_on_task_end_populates_fisher(detector, X_normal):
    detector.fit_task(X_normal)
    detector.on_task_end()
    assert detector.fisher_ is not None
    assert detector.params_star_ is not None

def test_ewc_penalty_nonzero_after_first_task(detector, X_normal):
    detector.fit_task(X_normal)
    detector.on_task_end()
    X_task2 = np.random.randn(50, 4).astype(np.float32) + 2.0
    # La loss EWC doit être > 0 lors de la tâche 2
    initial_loss = detector._compute_ewc_penalty()
    assert initial_loss > 0.0

def test_threshold_set_after_fit(detector, X_normal):
    detector.fit_task(X_normal)
    assert hasattr(detector, "threshold_")
    assert detector.threshold_ >= 0

def test_get_ram_bytes_reasonable(detector):
    ram = detector.get_ram_bytes()
    assert 0 < ram < 64 * 1024  # ≤ 64 Ko
```

---

## Tests `DBSCANDetector`

```python
# tests/test_dbscan_detector.py

import numpy as np
import pytest
from src.models.unsupervised.dbscan_detector import DBSCANDetector

@pytest.fixture
def X_normal():
    rng = np.random.default_rng(0)
    return rng.standard_normal((80, 4)).astype(np.float32)

def test_fit_task_refit(X_normal):
    det = DBSCANDetector(strategy="refit")
    det.fit_task(X_normal)
    det.on_task_end()
    det.fit_task(X_normal)  # deuxième tâche : réinitialise

def test_fit_task_accumulate(X_normal):
    det = DBSCANDetector(strategy="accumulate")
    det.fit_task(X_normal)
    det.on_task_end()
    det.fit_task(X_normal)  # accumule

def test_predict_score_shape(X_normal):
    det = DBSCANDetector(strategy="refit")
    det.fit_task(X_normal)
    scores = det.predict_score(X_normal)
    assert scores.shape == (80,)
    assert (scores >= 0).all()

def test_predict_score_anomaly_higher_than_normal(X_normal):
    det = DBSCANDetector(strategy="refit")
    det.fit_task(X_normal)
    X_anomaly = np.ones((10, 4), dtype=np.float32) * 100.0  # très éloignés
    scores_normal = det.predict_score(X_normal).mean()
    scores_anomaly = det.predict_score(X_anomaly).mean()
    assert scores_anomaly > scores_normal
```

---

## Commande d'exécution

```bash
pytest tests/test_ewc_oneclass.py tests/test_dbscan_detector.py -v
```

---

## Critères d'acceptation

- [ ] `pytest tests/test_ewc_oneclass.py -v` → 7 tests, 100% pass
- [ ] `pytest tests/test_dbscan_detector.py -v` → 4 tests, 100% pass
- [ ] Aucun accès aux fichiers `data/` dans les tests (fixtures synthétiques uniquement)

## Statut

⬜ À faire
