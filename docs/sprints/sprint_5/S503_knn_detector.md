# S5-03 — Implémenter `knn_detector.py` (KNN distance-based anomaly detection)

| Champ | Valeur |
|-------|--------|
| **ID** | S5-03 |
| **Sprint** | Sprint 5 — Semaine 5 (13–20 mai 2026) |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | S5-01 (structure + config YAML) |
| **Fichiers cibles** | `src/models/unsupervised/knn_detector.py` |
| **Complété le** | — |

---

## Objectif

Implémenter `KNNDetector`, une baseline de détection d'anomalies par distance aux k plus proches voisins. Le score d'anomalie est la distance moyenne aux k voisins les plus proches dans le jeu d'entraînement. Un échantillon éloigné de ses voisins est considéré comme anormal.

**Points clés** :
- Score d'anomalie = distance moyenne aux k plus proches voisins (pas de clustering)
- Stratégie CL `accumulate` : les échantillons de référence s'accumulent entre tâches (réseau de voisins grandissant)
- Stratégie CL `refit` : seuls les données de la tâche courante sont conservées
- Seuil de décision calculé sur Task 0 (percentile configurable)

**Critère de succès** : `python -c "from src.models.unsupervised import KNNDetector"` passe, et un appel `fit_task` + `predict` retourne des prédictions binaires correctes sur des données synthétiques.

---

## Sous-tâches

### 1. Constantes du module

```python
# src/models/unsupervised/knn_detector.py

# Valeurs par défaut — toujours passer par configs/unsupervised_config.yaml
N_NEIGHBORS_DEFAULT: int = 5
METRIC_DEFAULT: str = "euclidean"
ANOMALY_PERCENTILE_DEFAULT: int = 95
```

### 2. Classe `KNNDetector`

```python
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors


class KNNDetector:
    """
    Baseline de détection d'anomalies par distance aux k plus proches voisins.

    Le score d'anomalie est la distance moyenne aux k plus proches voisins dans
    l'ensemble de référence. Aucun label n'est utilisé pendant l'entraînement.

    En scénario CL (cl_strategy="accumulate"), les échantillons de référence
    s'accumulent entre tâches — le modèle grandit mais ne « oublie » pas.
    En scénario CL (cl_strategy="refit"), seule la tâche courante est conservée.

    Parameters
    ----------
    config : dict
        Sous-section "knn" de unsupervised_config.yaml.

    Attributes
    ----------
    nn_ : sklearn.neighbors.NearestNeighbors | None
        Modèle KNN entraîné (None avant fit_task).
    X_ref_ : np.ndarray | None
        Données de référence stockées ([N_total, n_features]).
    threshold_ : float | None
        Seuil de décision (calculé sur Task 0).
    task_id_ : int
        Index de la dernière tâche entraînée.

    Notes
    -----
    PC-only — pas de contrainte 64 Ko STM32N6.
    `cl_strategy="accumulate"` : RAM croissante au fil des tâches (acceptable PC-only).
    """

    def __init__(self, config: dict) -> None:
        self.n_neighbors: int = config.get("n_neighbors", N_NEIGHBORS_DEFAULT)
        self.metric: str = config.get("metric", METRIC_DEFAULT)
        self.anomaly_threshold: float | None = config.get("anomaly_threshold", None)
        self.anomaly_percentile: int = config.get("anomaly_percentile", ANOMALY_PERCENTILE_DEFAULT)
        self.cl_strategy: str = config.get("cl_strategy", "accumulate")

        self.nn_: NearestNeighbors | None = None
        self.X_ref_: np.ndarray | None = None
        self.threshold_: float | None = self.anomaly_threshold
        self.task_id_: int = -1

    def fit_task(self, X: np.ndarray, task_id: int) -> "KNNDetector":
        """
        Met à jour le modèle KNN avec les données d'une nouvelle tâche.

        Si cl_strategy=="accumulate", les données sont concaténées aux références existantes.
        Si cl_strategy=="refit", les références sont remplacées par les nouvelles données.

        Le seuil de décision est calculé une seule fois sur Task 0.

        Parameters
        ----------
        X : np.ndarray [N, n_features]
            Données d'entraînement (sans labels).
        task_id : int
            Index de la tâche (0-based).

        Returns
        -------
        self
        """
        self.task_id_ = task_id

        if self.cl_strategy == "accumulate":
            if self.X_ref_ is None:
                self.X_ref_ = X.copy()
            else:
                self.X_ref_ = np.concatenate([self.X_ref_, X], axis=0)
            print(
                f"  [KNN] Tâche {task_id} — accumulate : "
                f"{len(X)} nouveaux + {len(self.X_ref_) - len(X)} existants = "
                f"{len(self.X_ref_)} références totales"
            )
        else:
            # refit : seule la tâche courante est conservée
            self.X_ref_ = X.copy()
            print(f"  [KNN] Tâche {task_id} — refit : {len(X)} références")

        # (Re)construire l'index KNN
        self.nn_ = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, len(self.X_ref_) - 1),
            metric=self.metric,
            algorithm="auto",
            n_jobs=-1,
        )
        self.nn_.fit(self.X_ref_)

        # Calcul du seuil sur Task 0 uniquement (pas de leakage)
        if task_id == 0 and self.threshold_ is None:
            scores = self._compute_scores(X)
            self.threshold_ = float(np.percentile(scores, self.anomaly_percentile))
            print(
                f"  [KNN] Seuil calculé sur Task 0 : {self.threshold_:.4f} "
                f"(percentile {self.anomaly_percentile})"
            )

        return self

    def _compute_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Calcule les scores d'anomalie pour chaque échantillon.

        Parameters
        ----------
        X : np.ndarray [N, n_features]

        Returns
        -------
        np.ndarray [N]
            Distance moyenne aux k plus proches voisins.
        """
        if self.nn_ is None:
            raise RuntimeError("KNNDetector non entraîné. Appeler fit_task() d'abord.")
        distances, _ = self.nn_.kneighbors(X)  # distances : [N, k]
        return distances.mean(axis=1)           # [N] — distance moyenne aux k voisins

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Retourne le score d'anomalie (distance moyenne aux k plus proches voisins).

        Un score élevé indique une anomalie probable.

        Parameters
        ----------
        X : np.ndarray [N, n_features]

        Returns
        -------
        np.ndarray [N], dtype=float32
        """
        return self._compute_scores(X).astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit le label binaire (0=normal, 1=anormal).

        Parameters
        ----------
        X : np.ndarray [N, n_features]

        Returns
        -------
        np.ndarray [N], dtype=int64

        Raises
        ------
        RuntimeError
            Si threshold_ n'a pas été calculé (fit_task sur Task 0 requis).
        """
        if self.threshold_ is None:
            raise RuntimeError(
                "Seuil non calculé. Appeler fit_task(X, task_id=0) sur Task 0 d'abord."
            )
        scores = self.anomaly_score(X)
        return (scores > self.threshold_).astype(np.int64)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcule l'accuracy binaire. Labels utilisés uniquement en évaluation.

        Parameters
        ----------
        X : np.ndarray [N, n_features]
        y : np.ndarray [N], valeurs ∈ {0, 1}

        Returns
        -------
        float
        """
        preds = self.predict(X)
        return float((preds == y.astype(np.int64)).mean())

    def count_parameters(self) -> int:
        """
        Retourne le nombre de points de référence × n_features (taille de X_ref_).

        Returns
        -------
        int
        """
        if self.X_ref_ is None:
            return 0
        return int(self.X_ref_.size)

    def summary(self) -> str:
        """Résumé du modèle pour affichage console."""
        n_ref = len(self.X_ref_) if self.X_ref_ is not None else 0
        threshold = f"{self.threshold_:.4f}" if self.threshold_ is not None else "—"
        return (
            f"KNNDetector | k={self.n_neighbors} | metric={self.metric} | "
            f"strategy={self.cl_strategy} | n_ref={n_ref} | threshold={threshold}"
        )

    def save(self, path: str | Path) -> None:
        """
        Sauvegarde le modèle au format pickle.

        Parameters
        ----------
        path : str | Path
        """
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[KNN] Modèle sauvegardé → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "KNNDetector":
        """Charge un modèle sauvegardé."""
        import pickle
        with open(Path(path), "rb") as f:
            return pickle.load(f)
```

---

## Critères d'acceptation

- [ ] `from src.models.unsupervised import KNNDetector` — aucune erreur d'import
- [ ] `fit_task(X, task_id=0)` — index KNN construit, seuil calculé sans erreur
- [ ] `fit_task(X, task_id=1)` avec `cl_strategy="accumulate"` — `len(X_ref_)` augmente
- [ ] `fit_task(X, task_id=1)` avec `cl_strategy="refit"` — `len(X_ref_)` = `len(X)` de Task 1
- [ ] `predict(X)` retourne `np.ndarray` de shape `[N]` avec valeurs ∈ {0, 1}
- [ ] `anomaly_score(X)` retourne `np.ndarray` de shape `[N]`, dtype `float32`
- [ ] `score(X, y)` retourne un float entre 0 et 1
- [ ] `count_parameters()` retourne `n_ref × n_features` (croissant si accumulate)
- [ ] `ruff check src/models/unsupervised/knn_detector.py` + `black --check` passent
- [ ] `save` + `load` round-trip : mêmes prédictions avant et après sérialisation

---

## Questions ouvertes

- `TODO(arnaud)` : `cl_strategy="accumulate"` est la valeur par défaut — mais sur 3 tâches × ~2000 échantillons, X_ref_ atteint ~6000 points × 4 features = ~192 Ko (FP32, RAM Python). Acceptable pour PC-only, mais confirmer que ce n'est pas une simulation fidèle d'un vrai MCU (qui ne pourrait pas stocker tout ça).
- `TODO(arnaud)` : l'évaluation CL du KNN avec `accumulate` ne devrait pas montrer d'oubli catastrophique par construction (les données passées restent dans X_ref_). Faut-il quand même reporter AF et BWT pour la complétude du tableau comparatif ?
- `TODO(dorra)` : `n_jobs=-1` dans NearestNeighbors utilise tous les cœurs CPU — désactiver pour une simulation plus fidèle d'un MCU mono-cœur ?
- `FIXME(gap1)` : tester sur un dataset avec vrai déséquilibre de classes (plus d'anormaux dans certains domaines) — Dataset 2 a un ratio faulty/normal à vérifier.
