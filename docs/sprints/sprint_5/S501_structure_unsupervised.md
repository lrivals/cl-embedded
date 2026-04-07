# S5-01 — Structure `src/models/unsupervised/` + config YAML

| Champ | Valeur |
|-------|--------|
| **ID** | S5-01 |
| **Sprint** | Sprint 5 — Semaine 5 (13–20 mai 2026) |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 1h |
| **Dépendances** | S1-03 (`monitoring_dataset.py` — pour connaître `N_FEATURES` et les domaines) |
| **Fichiers cibles** | `src/models/unsupervised/__init__.py`, `configs/unsupervised_config.yaml` |
| **Complété le** | 7 avril 2026 |

---

## Objectif

Initialiser le module `src/models/unsupervised/` et créer la configuration YAML partagée par les trois modèles non supervisés (K-Means, KNN, PCA). Ce fichier est le point d'entrée de tout le Sprint 5 : les tâches S5-02 à S5-05 en dépendent.

**Critère de succès** : `python -c "from src.models.unsupervised import KMeansDetector, KNNDetector, PCABaseline"` passe (après implémentation des modèles en S5-02/03/04), et `configs/unsupervised_config.yaml` est valide et chargeable sans erreur.

---

## Sous-tâches

### 1. Créer `src/models/unsupervised/__init__.py`

```python
# src/models/unsupervised/__init__.py
"""
Module baselines non supervisées pour la détection d'anomalies en scénario CL.

Ces modèles sont PC-only — ils ne sont pas soumis à la contrainte 64 Ko STM32N6.
Les labels ne sont utilisés qu'en évaluation (pas pendant l'entraînement).

Modèles disponibles
-------------------
KMeansDetector : K-Means avec sélection K dynamique via silhouette/elbow.
KNNDetector    : KNN distance-based anomaly detection.
PCABaseline    : PCA reconstruction error baseline.

Références
----------
docs/models/unsupervised_spec.md (à créer en S5-08)
"""

from .kmeans_detector import KMeansDetector
from .knn_detector import KNNDetector
from .pca_baseline import PCABaseline

__all__ = ["KMeansDetector", "KNNDetector", "PCABaseline"]
```

### 2. Créer `configs/unsupervised_config.yaml`

```yaml
# configs/unsupervised_config.yaml
# Configuration partagée pour les baselines non supervisées (Sprint 5)
# NE PAS modifier les valeurs directement dans le code source.
# Toute modification doit passer par ce fichier.

seed: 42
exp_id: "exp_005_unsupervised_dataset2"

data:
  csv_path: "data/raw/equipment_monitoring/equipment_monitoring.csv"
  normalizer_path: "data/processed/monitoring_normalizer.pkl"
  n_features: 4           # temperature, pressure, vibration, humidity (sans one-hot équipement)
  batch_size: 256         # batch pour l'évaluation (pas d'entraînement online)
  val_ratio: 0.2
  domains: ["pump", "turbine", "compressor"]

kmeans:
  k_method: "silhouette"  # "silhouette" | "elbow" | "fixed"
  k_fixed: 3              # utilisé uniquement si k_method == "fixed"
  k_min: 2                # borne basse pour la recherche K dynamique
  k_max: 10               # borne haute pour la recherche K dynamique
  anomaly_threshold: null  # null → calculé automatiquement (percentile sur Task 1)
  anomaly_percentile: 95  # percentile pour fixer le seuil de distance
  n_init: 10              # sklearn KMeans n_init
  max_iter: 300
  cl_strategy: "refit"    # "refit" | "accumulate" — TODO(arnaud) à confirmer

knn:
  n_neighbors: 5
  metric: "euclidean"     # "euclidean" | "manhattan" | "cosine"
  anomaly_threshold: null  # null → calculé automatiquement (percentile sur Task 1)
  anomaly_percentile: 95
  cl_strategy: "accumulate"  # "accumulate" | "refit" — TODO(arnaud) à confirmer

pca:
  n_components: 2         # dimensions retenues (à ajuster selon explained_variance)
  min_explained_variance: 0.95  # si n_components=null, sélection automatique
  anomaly_threshold: null  # null → calculé automatiquement (percentile sur Task 1)
  anomaly_percentile: 95
  cl_strategy: "refit"    # "refit" | "incremental" — TODO(arnaud) à confirmer
                          # "incremental" → sklearn IncrementalPCA — TODO(dorra)

evaluation:
  metrics: ["aa", "af", "bwt", "auroc"]  # auroc = AUC-ROC (labels éval uniquement)
  n_latency_runs: 100     # runs pour mesure latence
```

### 3. Vérifier la compatibilité avec `monitoring_dataset.py`

Le loader `src/data/monitoring_dataset.py` retourne des tenseurs avec `N_FEATURES = 4` features numériques (temperature, pressure, vibration, humidity). S'assurer que `data.n_features: 4` dans le YAML est cohérent avec `NUMERIC_FEATURES` défini dans le loader.

```python
# Vérification rapide (à exécuter manuellement)
from src.data.monitoring_dataset import get_cl_dataloaders
import yaml

with open("configs/unsupervised_config.yaml") as f:
    config = yaml.safe_load(f)

tasks = get_cl_dataloaders(
    csv_path=config["data"]["csv_path"],
    normalizer_path=config["data"]["normalizer_path"],
    batch_size=config["data"]["batch_size"],
    val_ratio=config["data"]["val_ratio"],
    seed=config["seed"],
)

x_sample, y_sample = next(iter(tasks[0]["train_loader"]))
assert x_sample.shape[1] == config["data"]["n_features"], (
    f"N_FEATURES mismatch : loader retourne {x_sample.shape[1]}, "
    f"config attend {config['data']['n_features']}"
)
print(f"[OK] N_FEATURES = {x_sample.shape[1]}, domaines = {[t['domain'] for t in tasks]}")
```

---

## Critères d'acceptation

- [x] `src/models/unsupervised/__init__.py` créé et syntaxiquement valide (`ruff check` passe)
- [x] `configs/unsupervised_config.yaml` chargeable via `yaml.safe_load` sans erreur
- [x] `data.n_features: 4` cohérent avec `monitoring_dataset.py`
- [x] Les trois sections `kmeans`, `knn`, `pca` sont présentes dans le YAML
- [x] `anomaly_threshold: null` dans les trois sections (calcul automatique via percentile)
- [x] `black --check src/models/unsupervised/__init__.py` passe

## Notes d'implémentation

- `normalizer_path` corrigé vers `configs/monitoring_normalizer.yaml` (format YAML cohérent avec `hdc_config.yaml` et `ewc_config.yaml`, et non `.pkl` comme indiqué dans la spec initiale).
- Section `memory` ajoutée au YAML pour cohérence avec les autres configs (PC-only, non contraignante).
- `domain_order` en casse mixte (`["Pump", "Turbine", "Compressor"]`) pour correspondre exactement à `DOMAIN_ORDER` dans `monitoring_dataset.py`.

---

## Questions ouvertes

- `TODO(arnaud)` : config unique `unsupervised_config.yaml` ou trois configs séparés (`kmeans_config.yaml`, etc.) ? La config unique simplifie le script d'entraînement S5-05.
- `TODO(arnaud)` : `cl_strategy` pour K-Means — `refit` (recommencer sur toutes les données vues) ou `accumulate` (conserver les centroides des tâches passées) ? Choix architectural majeur pour l'évaluation CL.
- `TODO(dorra)` : `sklearn.decomposition.IncrementalPCA` vs PCA standard — pertinent pour simuler un vrai online learning sur MCU ?
- `FIXME(gap1)` : ces modèles non supervisés seront comparés aux modèles supervisés M1/M2/M3 — s'assurer que les splits val/test sont identiques (même seed=42, même normalizer).
