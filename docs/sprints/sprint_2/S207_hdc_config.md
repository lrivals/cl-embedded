# S2-07 — Config YAML pour HDC + refactoring configs

| Champ | Valeur |
|-------|--------|
| **ID** | S2-07 |
| **Sprint** | Sprint 2 — Semaine 2 (22–29 avril 2026) |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 1h |
| **Dépendances** | S2-01 (`base_vectors.py` — pour `N_FEATURES` et `base_vectors_path`), S2-02 (`hdc_classifier.py` — pour `feature_bounds`) |
| **Fichiers cibles** | `configs/hdc_config.yaml` |
| **Complété le** | 6 avril 2026 |

---

## Objectif

Compléter et valider `configs/hdc_config.yaml`. Le fichier squelette existe déjà (créé à S2-01) mais contient des valeurs `null` non renseignées pour `feature_bounds`. Ce sprint finalise la config pour que `scripts/train_hdc.py` puisse s'exécuter sans modification du code source.

**Règle absolue** : toute modification de paramètre passe par ce fichier YAML — jamais directement dans le code source.

**Critère de succès** : `python -c "import yaml; yaml.safe_load(open('configs/hdc_config.yaml'))"` passe sans erreur, et `hdc_classifier.py` charge la config sans `KeyError` ni `None` check manquant.

---

## Sous-tâches

### 1. Calculer les `feature_bounds` (script one-shot)

Les bornes de quantification doivent être calculées sur **Task 1 (pumps uniquement)** pour respecter la contrainte online-learning : le modèle ne doit pas observer les distributions futures.

```python
# Script à exécuter une fois en cellule notebook ou en REPL :
# notebooks/01_data_exploration.ipynb → Section "HDC Feature Bounds"

import yaml
import numpy as np
from src.data.monitoring_dataset import load_monitoring_dataset

config = yaml.safe_load(open("configs/hdc_config.yaml"))
tasks = load_monitoring_dataset(config)

# Task 0 = pumps
pump_task = tasks[0]
all_x = []
for x_batch, _ in pump_task["train_loader"]:
    all_x.append(x_batch.numpy())
X_pump = np.concatenate(all_x, axis=0)

# Features : temperature, pressure, vibration, humidity (indices 0-3 dans le tenseur X)
FEATURE_NAMES = ["temperature", "pressure", "vibration", "humidity"]
for i, name in enumerate(FEATURE_NAMES):
    print(f"{name}: [{X_pump[:, i].min():.4f}, {X_pump[:, i].max():.4f}]")
```

> Reporter les valeurs obtenues dans `feature_bounds` ci-dessous (§2).

### 2. Config complète cible

Voici la version cible de `configs/hdc_config.yaml` après complétion :

```yaml
# configs/hdc_config.yaml
# Hyperparamètres HDC (Hyperdimensional Computing) — Dataset 2 (Equipment Monitoring)
# NE PAS modifier les valeurs directement — utiliser ce fichier comme seul point de vérité.

exp_id: "exp_002_hdc_dataset2"

hdc:
  D: 1024            # Dimension des hypervecteurs (puissance de 2 pour SIMD)
                     # MEM: D × N_LEVELS × 1 B = 10 Ko @ INT8 (H_level)
                     #      D × N_FEATURES × 1 B = 4 Ko @ INT8 (H_pos)
  n_levels: 10       # Niveaux de quantification par feature
  seed: 42           # Seed génération vecteurs de base — FIXER après 1ère init
  base_vectors_path: "configs/hdc_base_vectors.npz"   # généré par S2-01

data:
  dataset: "equipment_monitoring"
  path: "data/raw/equipment_monitoring/"
  domain_column: "equipment"
  domain_order: ["pump", "turbine", "compressor"]
  label_column: "faulty"
  feature_columns: ["temperature", "pressure", "vibration", "humidity"]
  categorical_columns: ["equipment"]
  n_features: 4          # Features numériques uniquement (sans one-hot équipement)
                         # TODO(arnaud) : 4 ou 6 ? Voir S2-01 §1 pour la discussion.
  n_classes: 2           # faulty / normal
  test_split: 0.2
  normalizer_path: "configs/monitoring_normalizer.yaml"

  # Bornes pour quantification — calculées sur Task 1 (pumps) uniquement
  # IMPORTANT : bornes sur données normalisées Z-score (après monitoring_normalizer.yaml)
  feature_bounds:
    temperature:  [-3.0, 3.0]   # À remplacer par les valeurs réelles (§1)
    pressure:     [-3.0, 3.0]   # À remplacer par les valeurs réelles (§1)
    vibration:    [-3.0, 3.0]   # À remplacer par les valeurs réelles (§1)
    humidity:     [-3.0, 3.0]   # À remplacer par les valeurs réelles (§1)

training:
  # HDC est one-pass par design — pas d'epochs, pas d'optimizer
  # Ces champs sont présents pour cohérence avec ewc_config.yaml
  optimizer: null        # HDC : pas de gradient
  epochs_per_task: 1     # 1 passe par tâche (online learning strict)
  batch_size: 1          # update sample-by-sample sur MCU
  seed: 42

evaluation:
  seed: 42
  metrics: ["aa", "af", "bwt", "ram_peak_bytes", "n_params"]
  output_dir: "experiments/exp_002_hdc_dataset2/results/"

memory:
  target_ram_bytes: 65536       # 64 Ko — contrainte STM32N6
  expected_ram_bytes: 12288     # ~12 Ko estimé (prototypes INT32 + buffer encodage)
                                # MEM: 2 classes × 1024 dims × 4 B = 8 Ko (prototypes FP32)
                                #      + 4 features × 4 B = 16 B (buffer encodage)
                                #      + overhead Python ~4 Ko
  warn_if_above_bytes: 52000    # Alerte si > 50 Ko (laisse 14 Ko de marge)
```

### 3. Points de vérification après complétion

Vérifier la cohérence avec `ewc_config.yaml` sur les points suivants :

| Champ | ewc_config.yaml | hdc_config.yaml | OK ? |
|-------|----------------|----------------|------|
| `data.domain_order` | `["pump", "turbine", "compressor"]` | identique | à vérifier |
| `data.test_split` | `0.2` | `0.2` | ✅ |
| `data.normalizer_path` | `configs/monitoring_normalizer.yaml` | identique | à vérifier |
| `evaluation.metrics` | `["aa", "af", "bwt", ...]` | identique | à vérifier |
| `memory.target_ram_bytes` | `65536` | `65536` | ✅ |

### 4. Documenter le choix `n_features: 4` vs `6`

Ajouter une note dans le fichier expliquant la décision architecturale :

```yaml
# NOTE ARCHITECTURALE — n_features
# La spec HDC (docs/models/hdc_spec.md §2.1) mentionne 6 features (4 numériques + 2 one-hot).
# Mais monitoring_dataset.py retourne X avec 4 features numériques uniquement
# (l'encodage one-hot équipement est utilisé uniquement pour le split de domaine,
# pas comme feature d'entrée du modèle).
# Décision retenue : n_features = 4 pour cohérence avec le loader existant.
# TODO(arnaud) : confirmer si one-hot doit être inclus comme feature HDC.
```

---

## Critères d'acceptation

- [x] `yaml.safe_load(open("configs/hdc_config.yaml"))` — aucune erreur YAML
- [x] Aucune valeur `null` dans `feature_bounds` (toutes les bornes sont des nombres)
- [x] `n_features: 4` aligné avec `base_vectors.py::N_FEATURES = 4`
- [x] Section `training` présente pour cohérence avec `ewc_config.yaml` (même structure)
- [x] Section `memory` avec `expected_ram_bytes` et annotations `# MEM:` commentées
- [x] Chargeable par `HDCClassifier.__init__(config)` sans `KeyError`
- [x] Commentaire architectural sur `n_features: 4 vs 6` présent

---

## Interface attendue par `scripts/train_hdc.py` (S2-03)

```python
import yaml
from src.models.hdc import HDCClassifier

config = yaml.safe_load(open("configs/hdc_config.yaml"))
model = HDCClassifier(config)
# model.config["hdc"]["D"] == 1024
# model.config["data"]["feature_bounds"]["temperature"] == [min_val, max_val]
# model.config["hdc"]["base_vectors_path"] == "configs/hdc_base_vectors.npz"
```

---

## Questions ouvertes

- `TODO(arnaud)` : `n_features: 4` (sans one-hot) ou `6` (avec one-hot équipement) ? Impacte directement `H_pos.shape` et la RAM.
- `TODO(dorra)` : les `feature_bounds` doivent-ils être calculés sur les données **brutes** ou **normalisées Z-score** ? Actuellement : normalisées (après `monitoring_normalizer.yaml`). Sur MCU, la normalisation Z-score serait pré-calculée en Flash.
- `FIXME(gap2)` : vérifier que `expected_ram_bytes: 12288` est bien mesuré à l'exécution après S2-03.
