# S12-02 — Configs YAML CWRU (3 fichiers)

| Champ | Valeur |
|-------|--------|
| **ID** | S12-02 |
| **Sprint** | Sprint 12 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 1h |
| **Dépendances** | S12-01 (`cwru_dataset.py` validé) |
| **Fichiers cibles** | `configs/cwru_single_task_config.yaml`, `configs/cwru_by_fault_config.yaml`, `configs/cwru_by_severity_config.yaml` |

---

## Objectif

Créer les trois fichiers de configuration YAML pour les scénarios CWRU, en respectant la convention du projet : tous les hyperparamètres sont définis dans `configs/`, jamais en dur dans le code source. Les constantes doivent satisfaire la contrainte RAM ≤ 64 Ko du STM32N6.

---

## Contenu des configs

### `cwru_single_task_config.yaml`

```yaml
# Scénario : toutes les fenêtres CWRU, pas de split CL
DATASET: cwru
SCENARIO: no_split

# Données
CSV_PATH: data/raw/CWRU Bearing Dataset/feature_time_48k_2048_load_1.csv
WINDOW_SIZE: 2048        # points @ 48 kHz
N_FEATURES: 9            # features statistiques extraites
LABEL_COL: label         # colonne label binaire {0, 1}

# Split
TEST_SPLIT: 0.2
RANDOM_STATE: 42

# Entraînement commun
BATCH_SIZE: 1            # online strict (MCU)
N_EPOCHS: 1

# EWC
EWC_LAMBDA: 400
EWC_LR: 0.001

# HDC
HDC_DIM: 512             # MEM: 2048 B @ FP32 / 512 B @ INT8
HDC_LEVELS: 10

# TinyOL
TINYOL_HIDDEN: 8
TINYOL_HEAD_LR: 0.01

# RAM budget
RAM_BUDGET_BYTES: 65536  # 64 Ko
```

### `cwru_by_fault_config.yaml`

```yaml
# Scénario : 3 tâches domain-incremental par type de défaut
DATASET: cwru
SCENARIO: by_fault_type

CSV_PATH: data/raw/CWRU Bearing Dataset/feature_time_48k_2048_load_1.csv
WINDOW_SIZE: 2048
N_FEATURES: 9
N_TASKS_FAULT: 3         # Ball → Inner Race → Outer Race
LABEL_COL: label

TEST_SPLIT: 0.2
RANDOM_STATE: 42

BATCH_SIZE: 1
N_EPOCHS: 1

EWC_LAMBDA: 400
EWC_LR: 0.001

HDC_DIM: 512
HDC_LEVELS: 10

TINYOL_HIDDEN: 8
TINYOL_HEAD_LR: 0.01

RAM_BUDGET_BYTES: 65536
```

### `cwru_by_severity_config.yaml`

```yaml
# Scénario : 3 tâches domain-incremental par sévérité de défaut
DATASET: cwru
SCENARIO: by_severity

CSV_PATH: data/raw/CWRU Bearing Dataset/feature_time_48k_2048_load_1.csv
WINDOW_SIZE: 2048
N_FEATURES: 9
N_TASKS_SEVERITY: 3      # 0.007" → 0.014" → 0.021"
LABEL_COL: label

TEST_SPLIT: 0.2
RANDOM_STATE: 42

BATCH_SIZE: 1
N_EPOCHS: 1

EWC_LAMBDA: 400
EWC_LR: 0.001

HDC_DIM: 512
HDC_LEVELS: 10

TINYOL_HIDDEN: 8
TINYOL_HEAD_LR: 0.01

RAM_BUDGET_BYTES: 65536
```

---

## Critères d'acceptation

- [x] Les 3 fichiers chargent sans erreur avec `yaml.safe_load()`
- [x] `WINDOW_SIZE=2048`, `N_FEATURES=9` présents dans les 3 configs
- [x] `N_TASKS_FAULT=3` dans `cwru_by_fault_config.yaml`
- [x] `N_TASKS_SEVERITY=3` dans `cwru_by_severity_config.yaml`
- [x] `RAM_BUDGET_BYTES=65536` présent dans les 3 configs
- [x] Toutes les clés en `UPPER_SNAKE_CASE`

## Statut

✅ Terminé — 24 avril 2026
