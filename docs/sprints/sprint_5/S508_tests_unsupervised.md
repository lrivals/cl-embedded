# S5-08 — Tests unitaires + spec `unsupervised_spec.md`

| Champ | Valeur |
|-------|--------|
| **ID** | S5-08 |
| **Sprint** | Sprint 5 — Semaine 5 (13–20 mai 2026) |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 2h |
| **Dépendances** | S5-02 (`KMeansDetector`), S5-03 (`KNNDetector`), S5-04 (`PCABaseline`) |
| **Fichiers cibles** | `tests/test_unsupervised.py`, `docs/models/unsupervised_spec.md` |
| **Complété le** | — |

---

## Objectif

Écrire la suite de tests unitaires pour les 3 modèles non supervisés (`KMeansDetector`, `KNNDetector`, `PCABaseline`) et rédiger la spec technique `unsupervised_spec.md`. Ces tests constituent la porte d'entrée qualité avant le notebook comparatif (S5-09).

**Points clés** :
- Interface commune des 3 modèles : `fit_task`, `predict`, `anomaly_score`, `score`, `save`, `load`, `summary`, `count_parameters`
- Tests rapides (<5 s sur données synthétiques 4 features)
- Couverture : init, entraînement CL séquentiel, prédiction, contraintes mémoire, sérialisation
- `unsupervised_spec.md` documente l'interface contractuelle et les choix d'architecture

**Critère de succès** : `pytest tests/test_unsupervised.py -v` passe avec ≥ 12 tests, 0 skip.

---

## Sous-tâches

### 1. Fixtures dans `tests/conftest.py`

Ajouter après les fixtures existantes :

```python
# tests/conftest.py — ajout fixtures unsupervised

import numpy as np
import pytest

N_FEATURES_MONITORING = 4
N_SAMPLES_TRAIN = 100
N_SAMPLES_VAL = 40


@pytest.fixture
def unsupervised_data():
    """
    Données synthétiques 4 features (monitoring) : normal + anomaly.

    Normale  : N(0, 1) — label 0
    Anomalie : N(5, 1) — label 1 (bien séparé pour des tests déterministes)
    """
    rng = np.random.default_rng(42)
    X_normal  = rng.normal(0, 1, (N_SAMPLES_TRAIN, N_FEATURES_MONITORING))
    X_anomaly = rng.normal(5, 1, (20, N_FEATURES_MONITORING))
    X_train = np.vstack([X_normal, X_anomaly[:10]])
    y_train = np.array([0] * N_SAMPLES_TRAIN + [1] * 10)

    X_val = np.vstack([
        rng.normal(0, 1, (N_SAMPLES_VAL, N_FEATURES_MONITORING)),
        X_anomaly[10:],
    ])
    y_val = np.array([0] * N_SAMPLES_VAL + [1] * 10)
    return {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val}


@pytest.fixture
def kmeans_config():
    """Config minimale KMeansDetector compatible avec unsupervised_config.yaml."""
    return {
        "kmeans": {
            "k_min": 2,
            "k_max": 4,
            "k_method": "silhouette",
            "anomaly_percentile": 95,
            "cl_strategy": "fit_task",
            "seed": 42,
        }
    }


@pytest.fixture
def knn_config():
    """Config minimale KNNDetector."""
    return {
        "knn": {
            "n_neighbors": 5,
            "metric": "euclidean",
            "anomaly_percentile": 95,
            "cl_strategy": "accumulate",
            "max_ref_samples": 200,
            "seed": 42,
        }
    }


@pytest.fixture
def pca_config():
    """Config minimale PCABaseline."""
    return {
        "pca": {
            "n_components": 2,
            "anomaly_percentile": 95,
            "cl_strategy": "fit_task",
            "seed": 42,
        }
    }
```

### 2. Tests `KMeansDetector`

```python
# tests/test_unsupervised.py

import numpy as np
import pytest
import tempfile
from pathlib import Path

from src.models.unsupervised import KMeansDetector, KNNDetector, PCABaseline


class TestKMeansDetector:

    def test_init_from_config(self, kmeans_config):
        """KMeansDetector s'instancie correctement depuis la config."""
        model = KMeansDetector(kmeans_config["kmeans"])
        assert model is not None

    def test_fit_task_and_predict_binary(self, kmeans_config, unsupervised_data):
        """fit_task puis predict retourne des prédictions binaires {0, 1}."""
        model = KMeansDetector(kmeans_config["kmeans"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        preds = model.predict(unsupervised_data["X_val"])
        assert preds.shape == (len(unsupervised_data["X_val"]),)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_anomaly_score_shape(self, kmeans_config, unsupervised_data):
        """anomaly_score retourne un vecteur de la bonne taille."""
        model = KMeansDetector(kmeans_config["kmeans"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        scores = model.anomaly_score(unsupervised_data["X_val"])
        assert scores.shape == (len(unsupervised_data["X_val"]),)
        assert np.all(scores >= 0), "Score d'anomalie doit être non-négatif"

    def test_score_between_0_and_1(self, kmeans_config, unsupervised_data):
        """score() retourne une accuracy entre 0 et 1."""
        model = KMeansDetector(kmeans_config["kmeans"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        acc = model.score(unsupervised_data["X_val"], unsupervised_data["y_val"])
        assert 0.0 <= acc <= 1.0

    def test_anomalies_have_higher_score(self, kmeans_config, unsupervised_data):
        """Les anomalies (N(5,1)) ont un score moyen > normaux (N(0,1))."""
        model = KMeansDetector(kmeans_config["kmeans"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        scores = model.anomaly_score(unsupervised_data["X_val"])
        y = unsupervised_data["y_val"]
        assert scores[y == 1].mean() > scores[y == 0].mean()

    def test_sequential_fit_two_tasks(self, kmeans_config, unsupervised_data):
        """fit_task séquentiel sur 2 tâches ne lève pas d'erreur."""
        model = KMeansDetector(kmeans_config["kmeans"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        model.fit_task(unsupervised_data["X_train"], task_id=1)
        preds = model.predict(unsupervised_data["X_val"])
        assert preds.shape == (len(unsupervised_data["X_val"]),)

    def test_summary_returns_string(self, kmeans_config, unsupervised_data):
        """summary() retourne une chaîne non vide après fit_task."""
        model = KMeansDetector(kmeans_config["kmeans"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        s = model.summary()
        assert isinstance(s, str) and len(s) > 0

    def test_count_parameters_positive(self, kmeans_config, unsupervised_data):
        """count_parameters() retourne un entier positif après fit."""
        model = KMeansDetector(kmeans_config["kmeans"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        n = model.count_parameters()
        assert isinstance(n, int) and n > 0

    def test_save_and_load(self, kmeans_config, unsupervised_data, tmp_path):
        """save() puis load() préserve les prédictions."""
        model = KMeansDetector(kmeans_config["kmeans"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        preds_before = model.predict(unsupervised_data["X_val"])

        checkpoint = tmp_path / "kmeans_test.pkl"
        model.save(checkpoint)

        model2 = KMeansDetector(kmeans_config["kmeans"])
        model2.load(checkpoint)
        preds_after = model2.predict(unsupervised_data["X_val"])

        np.testing.assert_array_equal(preds_before, preds_after)

    def test_ram_budget(self, kmeans_config, unsupervised_data):
        """count_parameters() × 4 bytes ≤ 64 Ko (contrainte STM32N6)."""
        model = KMeansDetector(kmeans_config["kmeans"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        ram_fp32 = model.count_parameters() * 4
        assert ram_fp32 <= 65536, f"KMeans dépasse 64 Ko : {ram_fp32} B"
```

### 3. Tests `KNNDetector`

```python
class TestKNNDetector:

    def test_init_from_config(self, knn_config):
        model = KNNDetector(knn_config["knn"])
        assert model is not None

    def test_fit_and_predict(self, knn_config, unsupervised_data):
        model = KNNDetector(knn_config["knn"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        preds = model.predict(unsupervised_data["X_val"])
        assert preds.shape == (len(unsupervised_data["X_val"]),)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_anomaly_score_monotone(self, knn_config, unsupervised_data):
        """Les anomalies (N(5,1)) ont un score de distance moyen plus élevé."""
        model = KNNDetector(knn_config["knn"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        scores = model.anomaly_score(unsupervised_data["X_val"])
        y = unsupervised_data["y_val"]
        assert scores[y == 1].mean() > scores[y == 0].mean()

    def test_accumulate_strategy_grows_ref(self, knn_config, unsupervised_data):
        """Avec cl_strategy=accumulate, X_ref_ grandit après chaque tâche."""
        model = KNNDetector(knn_config["knn"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        n_ref_0 = len(model.X_ref_)
        model.fit_task(unsupervised_data["X_train"], task_id=1)
        n_ref_1 = len(model.X_ref_)
        assert n_ref_1 >= n_ref_0, "cl_strategy=accumulate doit conserver les données passées"

    def test_save_load(self, knn_config, unsupervised_data, tmp_path):
        model = KNNDetector(knn_config["knn"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        preds_before = model.predict(unsupervised_data["X_val"])
        checkpoint = tmp_path / "knn_test.pkl"
        model.save(checkpoint)
        model2 = KNNDetector(knn_config["knn"])
        model2.load(checkpoint)
        np.testing.assert_array_equal(preds_before, model2.predict(unsupervised_data["X_val"]))
```

### 4. Tests `PCABaseline`

```python
class TestPCABaseline:

    def test_init_from_config(self, pca_config):
        model = PCABaseline(pca_config["pca"])
        assert model is not None

    def test_fit_and_predict(self, pca_config, unsupervised_data):
        model = PCABaseline(pca_config["pca"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        preds = model.predict(unsupervised_data["X_val"])
        assert preds.shape == (len(unsupervised_data["X_val"]),)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_reconstruction_error_higher_for_anomalies(self, pca_config, unsupervised_data):
        """Les anomalies ont une erreur de reconstruction moyenne plus élevée."""
        model = PCABaseline(pca_config["pca"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        scores = model.anomaly_score(unsupervised_data["X_val"])
        y = unsupervised_data["y_val"]
        assert scores[y == 1].mean() > scores[y == 0].mean()

    def test_n_components_respected(self, pca_config, unsupervised_data):
        """Le modèle PCA utilise exactement n_components composantes."""
        model = PCABaseline(pca_config["pca"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        assert model.pca_.n_components_ == pca_config["pca"]["n_components"]

    def test_save_load(self, pca_config, unsupervised_data, tmp_path):
        model = PCABaseline(pca_config["pca"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        preds_before = model.predict(unsupervised_data["X_val"])
        checkpoint = tmp_path / "pca_test.pkl"
        model.save(checkpoint)
        model2 = PCABaseline(pca_config["pca"])
        model2.load(checkpoint)
        np.testing.assert_array_equal(preds_before, model2.predict(unsupervised_data["X_val"]))

    def test_ram_budget(self, pca_config, unsupervised_data):
        """PCA composantes × n_features × 4 bytes ≤ 64 Ko."""
        model = PCABaseline(pca_config["pca"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        ram_fp32 = model.count_parameters() * 4
        assert ram_fp32 <= 65536, f"PCA dépasse 64 Ko : {ram_fp32} B"
```

### 5. Lancer les tests

```bash
# Suite complète
pytest tests/test_unsupervised.py -v

# Avec coverage
pytest tests/test_unsupervised.py -v --cov=src/models/unsupervised --cov-report=term-missing

# Test rapide (smoke test)
pytest tests/test_unsupervised.py -v -k "init or fit_and_predict"
```

---

## Spec `docs/models/unsupervised_spec.md`

Créer ce fichier avec les sections suivantes :

```markdown
# Spec — Modèles non supervisés (Sprint 5)

## 1. Interface commune

Tous les modèles (`KMeansDetector`, `KNNDetector`, `PCABaseline`) implémentent le contrat :

| Méthode | Signature | Description |
|---------|-----------|-------------|
| `fit_task` | `(X: np.ndarray, task_id: int) -> None` | Entraînement sur une tâche (sans labels) |
| `predict` | `(X: np.ndarray) -> np.ndarray[int]` | Prédiction binaire {0: normal, 1: anomalie} |
| `anomaly_score` | `(X: np.ndarray) -> np.ndarray[float]` | Score continu (plus élevé = plus anormal) |
| `score` | `(X: np.ndarray, y: np.ndarray) -> float` | Accuracy (labels utilisés uniquement en éval) |
| `save` | `(path: Path) -> None` | Sérialisation joblib |
| `load` | `(path: Path) -> None` | Désérialisation joblib |
| `summary` | `() -> str` | Résumé texte de l'état du modèle |
| `count_parameters` | `() -> int` | Nombre de paramètres stockés (pour RAM estimate) |

## 2. Stratégies CL

| Modèle | Stratégie par défaut | Description |
|--------|---------------------|-------------|
| KMeans | `fit_task` | Refait le clustering sur la tâche courante uniquement |
| KNN | `accumulate` | Accumule X_ref_ (borné par `max_ref_samples`) |
| PCA | `fit_task` | Refait la décomposition sur la tâche courante |

## 3. Empreinte mémoire estimée (Dataset 2 — 4 features, 3 tâches)

| Modèle | Paramètres stockés | RAM FP32 | RAM INT8 |
|--------|-------------------|:--------:|:--------:|
| KMeans | K centroides × 4 features | ~128 B | ~32 B |
| KNN | max_ref_samples × 4 features | ~3 200 B | ~800 B |
| PCA | n_components × 4 features | ~32 B | ~8 B |

## 4. Critères d'acceptation (mémoire)

- `count_parameters() * 4 ≤ 65 536` pour tous les modèles sur Dataset 2 (4 features)
- Pour Dataset 1 (~50 features), KNN nécessite `max_ref_samples ≤ 327` pour rester sous 64 Ko
```

---

## Critères d'acceptation

- [ ] `pytest tests/test_unsupervised.py -v` passe avec ≥ 12 tests, 0 skip
- [ ] `KMeansDetector` : 10 tests dont init, fit, predict, score, anomaly_score, summary, count_parameters, save/load, ram_budget, sequential_fit
- [ ] `KNNDetector` : ≥ 5 tests dont fit, predict, anomaly_score, accumulate_strategy, save/load
- [ ] `PCABaseline` : ≥ 5 tests dont fit, predict, reconstruction_error, n_components, save/load, ram_budget
- [ ] Fixtures `unsupervised_data`, `kmeans_config`, `knn_config`, `pca_config` ajoutées dans `conftest.py`
- [ ] `docs/models/unsupervised_spec.md` créé avec sections interface, stratégies CL, empreinte mémoire
- [ ] `ruff check tests/test_unsupervised.py` + `black --check` passent
- [ ] Tous les tests s'exécutent en < 5 s

---

## Questions ouvertes

- `TODO(arnaud)` : faut-il tester la reproductibilité (deux appels `fit_task` avec `seed=42` → résultats identiques) ou supposer que sklearn gère cela via `random_state` ?
- `TODO(arnaud)` : doit-on ajouter un test d'intégration end-to-end (`fit_task` × 3 tâches → `score`) ou garder les tests unitaires isolés ?
- `FIXME(gap2)` : la méthode `count_parameters()` doit être cohérente avec l'estimation analytique de la RAM. Vérifier que `count_parameters() * 4 == ram_peak_bytes` à ±30 % (le reste étant l'overhead Python/joblib).
