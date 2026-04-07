# S5-12 — *(optionnel)* Implémenter `gmm_detector.py` (GMM EM offline, K petit)

| Champ | Valeur |
|-------|--------|
| **ID** | S5-12 |
| **Sprint** | Sprint 5 — Semaine 5 (13–20 mai 2026) |
| **Priorité** | 🟢 Basse (optionnel — backlog si temps disponible) |
| **Durée estimée** | 3h |
| **Dépendances** | S5-01 (structure + config YAML), S5-10 (`MahalanobisDetector` — pour comparer les deux approches probabilistes) |
| **Fichiers cibles** | `src/models/unsupervised/gmm_detector.py`, `configs/unsupervised_config.yaml` |
| **Complété le** | — |

---

## Objectif

Implémenter `GMMDetector`, une baseline optionnelle de détection d'anomalies basée sur un Gaussian Mixture Model (GMM) entraîné offline par algorithme EM. Cette tâche est dans le backlog : elle apporte une valeur scientifique (modèle probabiliste multimodal vs unimodal Mahalanobis) mais n'est pas sur le chemin critique du livrable sprint 5.

**Pourquoi GMM et pas HMM ?**
- GMM : entraînement EM offline viable, K=2–3 raisonnable en RAM, applicable aux deux datasets tabulaires
- HMM : complexité O(T×N²), Baum-Welch incompatible avec l'online learning, non applicable au Dataset 2 — relégué au backlog phase 2

**Limitation embarquée** : GMM avec covariances complètes stocke K×d×d floats (pour K=3, d=4 → 48 floats = 192 B @ FP32). Reste compatible 64 Ko, mais moins compact que Mahalanobis (80 B). Classé **PC-friendly**, **potentiellement embarqué** avec K ≤ 3 et covariances diagonales.

**Critère de succès** : `python -c "from src.models.unsupervised.gmm_detector import GMMDetector"` passe, et `fit_task` + `predict` sur données synthétiques retournent des prédictions binaires cohérentes. AUROC > 0.80 sur Dataset 2.

---

## Sous-tâches

### 1. Constantes du module

```python
# src/models/unsupervised/gmm_detector.py

# Valeurs par défaut — toujours passer par configs/unsupervised_config.yaml
N_COMPONENTS_DEFAULT: int = 2         # K composantes Gaussiennes (K=2–3 pour MCU)
COVARIANCE_TYPE_DEFAULT: str = "full"  # "full" | "diag" | "spherical"
MAX_ITER_DEFAULT: int = 100            # itérations EM max
N_INIT_DEFAULT: int = 3               # restart EM (robustesse convergence locale)
ANOMALY_PERCENTILE_DEFAULT: int = 95
CL_STRATEGY_DEFAULT: str = "refit"    # "refit" uniquement (EM offline)
```

### 2. Classe `GMMDetector`

```python
import numpy as np
from pathlib import Path
from sklearn.mixture import GaussianMixture


class GMMDetector:
    """
    Baseline optionnelle de détection d'anomalies par GMM (Gaussian Mixture Model).

    L'entraînement EM est offline (fit_task). L'inférence calcule la log-vraisemblance
    négative comme score d'anomalie : un score élevé → faible probabilité sous le GMM → anomalie.

    Empreinte mémoire @ FP32 pour K=2, d=4, covariances complètes :
    # MEM: 192 B @ FP32 / 48 B @ INT8 (K=2, d=4, full cov)
    # MEM: 96 B @ FP32 / 24 B @ INT8 (K=2, d=4, diag cov)

    En scénario CL (cl_strategy="refit"), le GMM est réentraîné sur chaque nouvelle tâche.
    Pas de mémoire inter-tâches.

    Parameters
    ----------
    config : dict
        Sous-section "gmm" de unsupervised_config.yaml.

    Attributes
    ----------
    gmm_ : sklearn.mixture.GaussianMixture | None
        GMM entraîné (None avant fit_task).
    threshold_ : float | None
        Seuil de décision (calculé sur Task 0 si anomaly_threshold est null).
    task_id_ : int
        Index de la dernière tâche entraînée.

    Notes
    -----
    PC-friendly (K=2–3), potentiellement embarqué avec covariances diagonales.
    Labels utilisés uniquement en évaluation, jamais pendant fit_task.
    """

    def __init__(self, config: dict) -> None:
        self.n_components: int = config.get("n_components", N_COMPONENTS_DEFAULT)
        self.covariance_type: str = config.get("covariance_type", COVARIANCE_TYPE_DEFAULT)
        self.max_iter: int = config.get("max_iter", MAX_ITER_DEFAULT)
        self.n_init: int = config.get("n_init", N_INIT_DEFAULT)
        self.anomaly_threshold: float | None = config.get("anomaly_threshold", None)
        self.anomaly_percentile: int = config.get("anomaly_percentile", ANOMALY_PERCENTILE_DEFAULT)
        self.cl_strategy: str = config.get("cl_strategy", CL_STRATEGY_DEFAULT)

        self.gmm_: GaussianMixture | None = None
        self.threshold_: float | None = self.anomaly_threshold
        self.task_id_: int = -1

    def fit_task(self, X: np.ndarray, task_id: int) -> "GMMDetector":
        """
        Entraîne le GMM sur les données d'une tâche par algorithme EM (offline).

        Le seuil de décision est calculé sur Task 0 uniquement.

        Parameters
        ----------
        X : np.ndarray [N, d]
            Données d'entraînement (non labelisées).
        task_id : int
            Index de la tâche (0-based).

        Returns
        -------
        self
        """
        self.task_id_ = task_id

        self.gmm_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=42,
        )
        self.gmm_.fit(X)
        print(
            f"  [GMM] Tâche {task_id} — K={self.n_components}, "
            f"cov_type={self.covariance_type}, "
            f"converged={self.gmm_.converged_}"
        )

        # Calcul du seuil sur Task 0 uniquement (pas de leakage inter-tâches)
        if task_id == 0 and self.threshold_ is None:
            scores = self._compute_nll(X)
            self.threshold_ = float(np.percentile(scores, self.anomaly_percentile))
            print(
                f"  [GMM] Seuil calculé sur Task 0 : {self.threshold_:.4f} "
                f"(percentile {self.anomaly_percentile})"
            )

        return self

    def _compute_nll(self, X: np.ndarray) -> np.ndarray:
        """
        Calcule la log-vraisemblance négative (NLL) par échantillon.

        Un NLL élevé → faible densité de probabilité → anomalie probable.

        Parameters
        ----------
        X : np.ndarray [N, d]

        Returns
        -------
        np.ndarray [N]
            Scores NLL (log-vraisemblance négative).
        """
        if self.gmm_ is None:
            raise RuntimeError("GMMDetector non entraîné. Appeler fit_task() d'abord.")
        log_prob = self.gmm_.score_samples(X)   # [N] — log-vraisemblance par échantillon
        return -log_prob                         # [N] — NLL (plus élevé = plus anormal)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Retourne le score d'anomalie (NLL — log-vraisemblance négative).

        Parameters
        ----------
        X : np.ndarray [N, d]

        Returns
        -------
        np.ndarray [N], dtype=float32
        """
        return self._compute_nll(X).astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit le label binaire (0=normal, 1=anormal).

        Parameters
        ----------
        X : np.ndarray [N, d]

        Returns
        -------
        np.ndarray [N], dtype=int64
        """
        if self.threshold_ is None:
            raise RuntimeError(
                "Seuil non calculé. Appeler fit_task(X, task_id=0) sur Task 0 d'abord."
            )
        scores = self.anomaly_score(X)
        return (scores > self.threshold_).astype(np.int64)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy binaire. Labels utilisés en évaluation uniquement."""
        preds = self.predict(X)
        return float((preds == y.astype(np.int64)).mean())

    def count_parameters(self) -> int:
        """
        Nombre de paramètres GMM : poids (K) + moyennes (K×d) + covariances.

        Covariances : K×d×d (full) | K×d (diag) | K (spherical).
        """
        if self.gmm_ is None:
            return 0
        K, d = self.n_components, self.gmm_.means_.shape[1]
        if self.covariance_type == "full":
            cov_params = K * d * d
        elif self.covariance_type == "diag":
            cov_params = K * d
        else:  # spherical
            cov_params = K
        return K + K * d + cov_params  # poids + moyennes + covariances

    def summary(self) -> str:
        """Résumé du modèle pour affichage console."""
        threshold = f"{self.threshold_:.4f}" if self.threshold_ is not None else "—"
        return (
            f"GMMDetector | K={self.n_components} | cov={self.covariance_type} | "
            f"threshold={threshold} | strategy={self.cl_strategy} | "
            f"params={self.count_parameters()}"
        )

    def save(self, path: str | Path) -> None:
        """Sauvegarde le modèle au format pickle."""
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[GMM] Modèle sauvegardé → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "GMMDetector":
        """Charge un modèle sauvegardé."""
        import pickle
        with open(Path(path), "rb") as f:
            return pickle.load(f)
```

### 3. Section `gmm` à ajouter dans `configs/unsupervised_config.yaml`

```yaml
gmm:
  n_components: 2              # K composantes — K=2 ou 3 pour compatibilité MCU
  covariance_type: "full"      # "full" | "diag" | "spherical" (diag si MCU)
  max_iter: 100                # itérations EM
  n_init: 3                    # restarts EM pour robustesse
  anomaly_threshold: null       # null → calculé automatiquement (percentile sur Task 0)
  anomaly_percentile: 95
  cl_strategy: "refit"         # "refit" uniquement (EM offline)
```

### 4. Estimation RAM embarquée

Pour évaluer la compatibilité STM32N6 avant toute implémentation C :

```python
# Estimation RAM GMM @ FP32 pour différentes configurations
configs = [
    ("K=2, d=4, full",       2, 4, "full"),
    ("K=3, d=4, full",       3, 4, "full"),
    ("K=2, d=4, diag",       2, 4, "diag"),
    ("K=2, d=7, full",       2, 7, "full"),   # Dataset 1 pump (7 features)
]
for label, K, d, cov in configs:
    weights = K
    means = K * d
    if cov == "full":
        covs = K * d * d
    elif cov == "diag":
        covs = K * d
    else:
        covs = K
    total = (weights + means + covs) * 4  # @ FP32
    print(f"{label:<25} → {total:>5} B  ({total/1024:.2f} Ko)")

# Résultats attendus :
# K=2, d=4, full        →   160 B  (0.16 Ko) ✅
# K=3, d=4, full        →   228 B  (0.22 Ko) ✅
# K=2, d=4, diag        →    64 B  (0.06 Ko) ✅
# K=2, d=7, full        →   468 B  (0.46 Ko) ✅
```

---

## Critères d'acceptation

> ⚠️ Cette tâche est **optionnelle**. Les critères ci-dessous sont des objectifs, non des bloquants pour la livraison sprint 5.

- [ ] `from src.models.unsupervised.gmm_detector import GMMDetector` — aucune erreur d'import
- [ ] `fit_task(X, task_id=0)` — GMM converge (`gmm_.converged_ == True`) et seuil calculé
- [ ] `fit_task(X, task_id=1)` — GMM réentraîné (refit), seuil inchangé
- [ ] `predict(X)` retourne `np.ndarray` de shape `[N]` avec valeurs ∈ {0, 1}
- [ ] `anomaly_score(X)` retourne `np.ndarray` de shape `[N]`, dtype `float32`
- [ ] `score(X, y)` retourne un float entre 0 et 1
- [ ] `covariance_type="diag"` : paramètres réduits vs `"full"` (K×d vs K×d×d covariances)
- [ ] `ruff check src/models/unsupervised/gmm_detector.py` + `black --check` passent
- [ ] `save` + `load` round-trip : mêmes prédictions avant et après sérialisation
- [ ] AUROC > 0.80 sur Dataset 2 (à valider dans une micro-expérience)
- [ ] RAM estimée < 500 B @ FP32 pour K ≤ 3, d ≤ 7

---

## Questions ouvertes

- `TODO(arnaud)` : GMM vs Mahalanobis — la distribution bimodale (normal/faulty) du Dataset 2 justifie-t-elle K=2 ? Ou le GMM surfit-il le train set de Task 0 et généralise-t-il mal aux tâches suivantes ?
- `TODO(arnaud)` : `covariance_type="diag"` — suffisant pour Dataset 2 (features peu corrélées après normalisation) ? Réduirait la RAM de 192 B à 64 B pour K=2, d=4.
- `TODO(dorra)` : export du GMM en C pour STM32N6 — les covariances diagonales permettent une implémentation sans BLAS (somme de carrés pondérés). Vaut-il la peine d'explorer ?
- `FIXME(gap2)` : si GMM est implémenté, mesurer `ram_peak_bytes` et comparer à l'estimation analytique. Documenter pour le manuscrit.
- `FIXME(gap1)` : GMM sur Dataset 1 (séries temporelles) — la distribution des features brutes n'est pas nécessairement gaussienne. Documenter si AUROC < 0.75 comme limitation méthodologique.
