# S2511–S2512 — Configs YAML : RUL et Multi-classe

| Champ | Valeur |
|-------|--------|
| **Sprint** | 25 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Terminé |
| **Durée estimée** | S2511 : 1h / S2512 : 1h = 2h total |
| **Dépendances** | S2506 ✅ (EWCMlpRegressor — paramètres à configurer), S2507 ✅ (EWCMlpMulticlass) |
| **Fichiers cibles** | `configs/cmapss_rul_config.yaml`, `configs/pronostia_rul_config.yaml`, `configs/cwru_multiclass_config.yaml`, `configs/paderborn_multiclass_config.yaml` |
| **Référence** | `configs/cmapss_config.yaml` (pattern CMAPSS binaire), `configs/cwru_by_fault_config.yaml` (pattern CWRU), `configs/ewc_config.yaml` (hyperparamètres EWC) |

---

## Contexte

Chaque expérience Sprint 25 requiert une config YAML dédiée qui spécifie à la fois le `task_mode` (natif) et les hyperparamètres du modèle correspondant. Les configs YAML sont la **seule source d'hyperparamètres** — aucune valeur ne doit être codée en dur dans les scripts.

Convention de nommage : `{dataset}_{task_mode}_config.yaml` pour distinguer clairement des configs binaires existantes.

---

## S2511 — Configs RUL : CMAPSS et Pronostia

### `configs/cmapss_rul_config.yaml`

```yaml
# cmapss_rul_config.yaml — Config EWC Régression RUL sur CMAPSS
# Sprint 25 — mode natif RUL (ne pas confondre avec cmapss_config.yaml — binaire)
# Ne PAS modifier les hyperparamètres ici — créer une copie pour chaque variante.

# --- Identité ---
task_mode: rul          # "binary" | "rul" | "multiclass"
model: ewc_regression
dataset: cmapss
sprint: 25

# --- Dataset CMAPSS ---
data_dir: "data/raw/CMAPSS Jet Engine Simulated Data/"
feature_subset_path: "configs/cmapss_feature_subset.yaml"
domain_order:
  - "FD001"
  - "FD002"
  - "FD003"
  - "FD004"
n_tasks: 4
val_ratio: 0.2
rul_cap: 125            # clip RUL à 125 cycles (cohérence CMAPSS_RUL_CAP)
seed: 42

# --- Architecture EWCMlpRegressor ---
INPUT_DIM: 5            # top-5 features sélectionnées par mutual info
HIDDEN_DIMS:
  - 32
  - 16
DROPOUT: 0.2
OUTPUT_DIM: 1           # régression — 1 neurone de sortie

# --- Hyperparamètres EWC ---
EWC_LR: 0.01
EWC_LAMBDA: 400.0       # même valeur que mode binaire (réglage à affiner exp_S25_01)
FISHER_N_SAMPLES: 200   # nombre d'exemples pour le calcul de la Fisher diagonale
FISHER_EMA_DECAY: 0.99  # EWC Online : decay de la Fisher accumulée

# --- Entraînement ---
N_EPOCHS_PER_TASK: 20
BATCH_SIZE: 32
OPTIMIZER: sgd          # SGD recommandé pour compatibilité MCU
MOMENTUM: 0.9

# --- Métriques (rul_metrics.py) ---
primary_metric: rmse    # métrique de suivi principale
report_metrics:
  - rmse
  - mae
  - horizon_score
  - avg_forgetting_rmse

# --- Budget mémoire ---
ram_budget_bytes: 262144   # 256 Ko (NUCLEO-F439ZI)
```

### `configs/pronostia_rul_config.yaml`

```yaml
# pronostia_rul_config.yaml — Config EWC Régression RUL sur Pronostia
# Sprint 25 — mode natif RUL par condition opératoire.

# --- Identité ---
task_mode: rul
model: ewc_regression
dataset: pronostia
sprint: 25

# --- Dataset Pronostia ---
npy_dir: "data/raw/Pronostia dataset/binaries"
normalizer_path: "configs/pronostia_normalizer.yaml"
condition_order:
  - 1   # 1 800 rpm, 4 000 N
  - 2   # 1 650 rpm, 4 200 N
  - 3   # 1 500 rpm, 5 000 N
n_tasks: 3
val_ratio: 0.2
window_duration_s: 0.1  # WINDOW_DURATION_S = 2560 / 25600 Hz
seed: 42

# --- Architecture EWCMlpRegressor ---
INPUT_DIM: 13           # 12 features statistiques + position temporelle (Pronostia)
HIDDEN_DIMS:
  - 32
  - 16
DROPOUT: 0.2
OUTPUT_DIM: 1

# --- Hyperparamètres EWC ---
EWC_LR: 0.005           # LR plus faible — RUL Pronostia en secondes (valeurs élevées)
EWC_LAMBDA: 400.0
FISHER_N_SAMPLES: 200
FISHER_EMA_DECAY: 0.99

# --- Entraînement ---
N_EPOCHS_PER_TASK: 20
BATCH_SIZE: 32
OPTIMIZER: sgd
MOMENTUM: 0.9

# --- Métriques ---
primary_metric: rmse
report_metrics:
  - rmse
  - mae
  - avg_forgetting_rmse

# --- Budget mémoire ---
ram_budget_bytes: 262144
```

---

## S2512 — Configs Multi-classe : CWRU et Paderborn

### `configs/cwru_multiclass_config.yaml`

```yaml
# cwru_multiclass_config.yaml — Config EWC Multi-classe sur CWRU (10 classes, 3 tâches)
# Sprint 25 — mode natif multi-classe.
# Ne PAS confondre avec cwru_by_fault_config.yaml (binaire domain-incremental).

# --- Identité ---
task_mode: multiclass
model: ewc_multiclass
dataset: cwru
scenario: by_fault_type  # tâches : ball → inner_race → outer_race
sprint: 25

# --- Dataset CWRU ---
data_path: "data/raw/CWRU Bearing Dataset/feature_time_48k_2048_load_1.csv"
scenario: by_fault_type
n_tasks: 3
n_classes: 10            # 10 classes totales (Normal + 3 types × 3 sévérités)
val_ratio: 0.2
seed: 42

# Mapping classes (référence : MULTICLASS_LABEL_MAP dans cwru_dataset.py)
class_names:
  0: "Normal"
  1: "Ball_007"
  2: "Ball_014"
  3: "Ball_021"
  4: "IR_007"
  5: "IR_014"
  6: "IR_021"
  7: "OR_007"
  8: "OR_014"
  9: "OR_021"

# --- Architecture EWCMlpMulticlass ---
INPUT_DIM: 9             # 9 features statistiques CWRU (max, min, mean, sd, rms, skew, kurt, crest, form)
HIDDEN_DIMS:
  - 32
  - 16
DROPOUT: 0.2
N_CLASSES: 10

# --- Hyperparamètres EWC ---
EWC_LR: 0.01
EWC_LAMBDA: 400.0
FISHER_N_SAMPLES: 200
FISHER_EMA_DECAY: 0.99

# --- Entraînement ---
N_EPOCHS_PER_TASK: 30   # plus d'époques — 10 classes plus difficiles que 2
BATCH_SIZE: 32
OPTIMIZER: sgd
MOMENTUM: 0.9

# --- Métriques (multiclass_metrics.py) ---
primary_metric: f1_macro
report_metrics:
  - f1_macro
  - confusion_matrix
  - per_class_accuracy
  - avg_forgetting_f1

# --- Critère de validation ---
f1_macro_min_task1: 0.70   # critère exp_S25_03 (S2517)

# --- Budget mémoire ---
ram_budget_bytes: 262144
```

### `configs/paderborn_multiclass_config.yaml`

```yaml
# paderborn_multiclass_config.yaml — Config EWC Multi-classe sur Paderborn (3 états)
# Sprint 25 — mode natif multi-classe bearing.

# --- Identité ---
task_mode: multiclass
model: ewc_multiclass
dataset: paderborn
sprint: 25

# --- Dataset Paderborn ---
data_dir: "data/raw/paderborn/"
feature_subset_path: "configs/paderborn_feature_subset.yaml"
n_tasks: 3
n_classes: 3
val_ratio: 0.2
seed: 42

# Mapping états bearing
class_names:
  0: "K001_healthy"
  1: "KA04_outer_race"
  2: "KI04_inner_race"

# --- Architecture EWCMlpMulticlass ---
INPUT_DIM: 9             # à confirmer selon paderborn_loader.py
HIDDEN_DIMS:
  - 32
  - 16
DROPOUT: 0.2
N_CLASSES: 3

# --- Hyperparamètres EWC ---
EWC_LR: 0.01
EWC_LAMBDA: 400.0
FISHER_N_SAMPLES: 200
FISHER_EMA_DECAY: 0.99

# --- Entraînement ---
N_EPOCHS_PER_TASK: 20
BATCH_SIZE: 32
OPTIMIZER: sgd
MOMENTUM: 0.9

# --- Métriques ---
primary_metric: f1_macro
report_metrics:
  - f1_macro
  - confusion_matrix
  - avg_forgetting_f1

# --- Budget mémoire ---
ram_budget_bytes: 262144
```

---

## Vérification

```bash
# Charger chaque config et vérifier les clés obligatoires
python -c "
import yaml
from pathlib import Path

configs = [
    'configs/cmapss_rul_config.yaml',
    'configs/pronostia_rul_config.yaml',
    'configs/cwru_multiclass_config.yaml',
    'configs/paderborn_multiclass_config.yaml',
]
required_keys = {'task_mode', 'model', 'dataset', 'sprint', 'EWC_LR', 'EWC_LAMBDA', 'primary_metric'}

for cfg_path in configs:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    missing = required_keys - set(cfg.keys())
    assert not missing, f'{cfg_path} : clés manquantes {missing}'
    print(f'{cfg_path} OK ✅  (task_mode={cfg[\"task_mode\"]}, model={cfg[\"model\"]})')
"

# Vérifier la cohérence INPUT_DIM avec les loaders
python -c "
import yaml
from pathlib import Path

# CMAPSS : INPUT_DIM doit correspondre aux features sélectionnées
cfg = yaml.safe_load(Path('configs/cmapss_rul_config.yaml').read_text())
subset = yaml.safe_load(Path('configs/cmapss_feature_subset.yaml').read_text())
n_features = len(subset.get('selected_features', subset.get('features', [])))
assert cfg['INPUT_DIM'] == n_features, f'INPUT_DIM={cfg[\"INPUT_DIM\"]} != {n_features} features CMAPSS'
print(f'CMAPSS INPUT_DIM cohérent : {n_features} ✅')
"
```

---

## Résultats d'implémentation

| Fichier | Statut | Notes |
|---------|:------:|-------|
| `configs/cmapss_rul_config.yaml` | ✅ | INPUT_DIM=5 confirmé (cmapss_feature_subset.yaml) |
| `configs/pronostia_rul_config.yaml` | ✅ | INPUT_DIM=13 |
| `configs/cwru_multiclass_config.yaml` | ✅ | INPUT_DIM=9, N_CLASSES=10 |
| `configs/paderborn_multiclass_config.yaml` | ✅ | INPUT_DIM=5 corrigé (spec=9 → réel=5, PADERBORN_N_FEATURES_SELECTED) |

---

## Questions ouvertes

- `TODO(arnaud)` : `EWC_LAMBDA=400.0` est la valeur calibrée sur le mode binaire Monitoring. Faut-il recalibrer pour la régression MSE (unités différentes — cycles vs 0/1) ? Prévoir une exp de sensibilité λ ∈ {100, 400, 1000} ?
- `TODO(dorra)` : Le `INPUT_DIM` de Paderborn doit être confirmé en lisant `paderborn_loader.py` (valeur approximative mise à 9 ici — à corriger si différent).
- `FIXME(gap2)` : `ram_budget_bytes: 262144` (256 Ko) — cohérent avec NUCLEO-F439ZI. Ne pas réduire à 65536 (64 Ko) qui était la cible STM32N6 abandonnée.
