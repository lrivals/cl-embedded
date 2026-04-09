# Datasets — Description et protocoles d'utilisation

---

## Dataset 1 — Large Industrial Pump Maintenance Dataset

| Propriété | Valeur |
|-----------|--------|
| **Source** | Kaggle — `TODO(arnaud): slug exact à renseigner` |
| **Chemin local** | `data/raw/pump_maintenance/Large Industrial_Pump_Maintenance_Dataset/Large_Industrial_Pump_Maintenance_Dataset.csv` |
| **Type** | Séries temporelles multivariées |
| **Nature** | Simulé (réaliste) |
| **Tâche ML** | Classification binaire (Maintenance_Flag: 0/1) |
| **N échantillons** | 20 000 |

### Variables (noms bruts CSV → convention pipeline)

| Colonne CSV brute | Rename pipeline | Type | Description |
|-------------------|-----------------|------|-------------|
| `Pump_ID` | — (non utilisé) | int | Identifiant de la pompe |
| `Temperature` | `temperature` | float (°C) | État thermique de la machine |
| `Vibration` | `vibration` | float | Indicateur de défaut mécanique |
| `Pressure` | `pressure` | float | Pression interne du système |
| `Flow_Rate` | `flow_rate` | float | Débit (5e canal — absent de la spec initiale) |
| `RPM` | `rpm` | float | Vitesse de rotation |
| `Operational_Hours` | `operational_hours` | float | Proxy temporel (remplace `timestamp` absent) |
| `Maintenance_Flag` | `maintenance_required` | int (0/1) | Label binaire |

> **Note** : le CSV ne contient pas de colonne `timestamp`. L'axe temporel est reconstruit par tri croissant de `Operational_Hours`. Le renommage vers les noms snake_case est effectué à l'import dans `pump_dataset.py`.

### Scénario CL associé

**Domain-Incremental avec drift temporel** : les données évoluent chronologiquement selon un gradient de dégradation (pompe saine → usure progressive → pré-panne). Les tâches sont définies par découpage temporel, sans frontière explicite.

### Pipeline de preprocessing

```python
# src/data/pump_dataset.py

WINDOW_SIZE = 32        # fenêtre glissante
STEP_SIZE = 16          # chevauchement 50%
N_FEATURES = 25         # features extraites par fenêtre (voir tinyol_spec.md)
N_TASKS = 3             # découpage chronologique en 3 périodes

FEATURE_COLUMNS = ["temperature", "vibration", "pressure", "rpm"]
LABEL_COLUMN = "maintenance_required"
TIMESTAMP_COLUMN = "timestamp"

# Features par canal (6 features × 4 canaux = 24 + 1 = 25)
FEATURES_PER_CHANNEL = ["mean", "std", "rms", "kurtosis", "peak", "crest_factor"]
```

### Chargement recommandé

```python
from src.data.pump_dataset import PumpMaintenanceDataset, CLStreamSplitter

# 1. Chargement brut
dataset = PumpMaintenanceDataset("data/raw/pump_maintenance/")

# 2. Feature engineering (fenêtres + features statistiques)
features, labels = dataset.extract_features(window_size=32, step_size=16)

# 3. Split en stream CL (ordre chronologique obligatoire)
cl_stream = CLStreamSplitter(features, labels, n_tasks=3, strategy="chronological")

# 4. Normalisation (calculée sur Task 1 uniquement)
normalizer = cl_stream.fit_normalizer(task_id=0)  # IMPORTANT : fit sur T1 seulement
cl_stream.apply_normalizer(normalizer)
normalizer.save("configs/pump_normalizer.yaml")    # pour portage MCU
```

---

## Dataset 2 — Industrial Equipment Monitoring Dataset

| Propriété | Valeur |
|-----------|--------|
| **Source** | Kaggle |
| **Chemin local** | `data/raw/equipment_monitoring/` |
| **Type** | Tabulaire statique |
| **Nature** | Simulé |
| **Tâche ML** | Classification binaire (faulty: 0/1) |

### Variables

| Colonne | Type | Description | Encodage |
|---------|------|-------------|---------|
| `temperature` | float (°C) | Température mesurée | Z-score |
| `pressure` | float (bar) | Pression | Z-score |
| `vibration` | float | Niveau de vibration | Z-score |
| `humidity` | float (%) | Humidité ambiante | Z-score |
| `equipment` | string | Type d'équipement | One-hot (drop first) |
| `location` | string | Localisation | Label encoding |
| `faulty` | int (0/1) | État (0=normal, 1=défaillant) | — (label) |

### Scénario CL associé

**Domain-Incremental par type d'équipement** : entraînement séquentiel sur T1 = pumps, T2 = turbines, T3 = compressors. Les frontières de tâches sont explicites (task label disponible à l'entraînement, pas à l'inférence).

### Pipeline de preprocessing

```python
# src/data/monitoring_dataset.py

DOMAIN_COLUMN = "equipment"
DOMAIN_ORDER = ["pump", "turbine", "compressor"]   # ordre fixe pour reproductibilité
LABEL_COLUMN = "faulty"
FEATURE_COLUMNS = ["temperature", "pressure", "vibration", "humidity"]
CATEGORICAL_COLUMNS = ["equipment", "location"]

# Encodage : drop "location" si cardinalité > 10 (peu utile)
# One-hot "equipment" avec drop_first=True → 2 features binaires
N_FEATURES_FINAL = 6   # 4 numériques + 2 one-hot
```

### Chargement recommandé

```python
from src.data.monitoring_dataset import EquipmentMonitoringDataset, CLStreamSplitter

# 1. Chargement et preprocessing
dataset = EquipmentMonitoringDataset("data/raw/equipment_monitoring/")
X, y, task_ids = dataset.preprocess(domain_order=["pump", "turbine", "compressor"])

# 2. Split par domaine
cl_stream = CLStreamSplitter.from_domain_ids(X, y, task_ids)

# 3. Normalisation sur Task 1 (pumps)
normalizer = cl_stream.fit_normalizer(task_id=0)
cl_stream.apply_normalizer(normalizer)
normalizer.save("configs/monitoring_normalizer.yaml")
```

---

---

## Dataset 3 — Battery Remaining Useful Life (RUL)

| Propriété | Valeur |
|-----------|--------|
| **Source** | Kaggle |
| **Chemin local** | `data/raw/Battery Remaining Useful Life (RUL)/Battery_RUL.csv` |
| **Type** | Tabulaire séquentiel (cycles de charge/décharge) |
| **Nature** | Simulé / mesuré sur banc (lithium-ion) |
| **Tâche ML** | Détection d'anomalie binaire (`faulty = RUL ≤ ANOMALY_THRESHOLD`) |
| **N échantillons** | ~15 064 cycles |

### Colonnes brutes — Battery RUL

| Colonne CSV brute | Rename pipeline | Type | Description |
|-------------------|-----------------|------|-------------|
| `Cycle_Index` | `cycle_index` | int | Proxy temporel (tri croissant, non utilisé comme feature) |
| `Discharge Time (s)` | `discharge_time_s` | float | Durée de décharge complète |
| `Decrement 3.6-3.4V (s)` | `decrement_3640v_s` | float | Durée dans la fenêtre de tension critique |
| `Max. Voltage Dischar. (V)` | `max_voltage_discharge_v` | float | Tension maximale en décharge |
| `Min. Voltage Charg. (V)` | `min_voltage_charge_v` | float | Tension minimale en charge |
| `Time at 4.15V (s)` | `time_at_415v_s` | float | Durée à tension de charge maximale |
| `Time constant current (s)` | `time_constant_current_s` | float | Phase courant constant (CCCV) |
| `Charging time (s)` | `charging_time_s` | float | Durée totale de charge |
| `RUL` | `rul` | int (0–1112) | Remaining Useful Life — converti en label binaire |

> **Label binaire** : `faulty = int(rul <= ANOMALY_THRESHOLD)`. Valeur par défaut : `ANOMALY_THRESHOLD = 200` (≈18% du RUL max). Configurable dans `configs/unsupervised_config.yaml`.

### Scénario CL — Battery RUL

**Domain-Incremental temporel** : les 15 064 cycles sont découpés en 3 tiers chronologiques égaux par `cycle_index`. Tâche 0 = début de vie (batterie saine, RUL élevé), Tâche 2 = fin de vie (dégradée, RUL → 0). Même architecture de scénario que Dataset 1 — Pump.

### Pipeline de preprocessing — Battery RUL

```python
# src/data/battery_rul_dataset.py

N_FEATURES: int = 7        # colonnes numériques hors cycle_index et rul
N_TASKS: int = 3           # tiers chronologiques par cycle_index
ANOMALY_THRESHOLD: int = 200   # RUL ≤ 200 → faulty=1
VAL_RATIO: float = 0.2
```

### Chargement recommandé — Battery RUL

```python
from src.data.battery_rul_dataset import get_battery_dataloaders
import yaml

cfg = yaml.safe_load(open("configs/unsupervised_config.yaml"))["battery_rul"]
tasks = get_battery_dataloaders(cfg)
# → [{task_id: 0, train_loader, val_loader, n_train, n_val}, ...]
```

---

## Dataset 4 — FEMTO PRONOSTIA (IEEE PHM Challenge 2012)

| Propriété | Valeur |
|-----------|--------|
| **Référence** | Nectoux et al. (2012) — IEEE PHM Data Challenge |
| **Chemin local** | `data/raw/Pronostia dataset/` |
| **Type** | Séries temporelles brutes (vibrations + température) |
| **Nature** | **Données réelles** (tests run-to-failure sur banc FEMTO-ST) |
| **Tâche ML** | Détection d'anomalie binaire (derniers 20% de vie = faulty=1) |
| **Accès** | Public — IEEE PHM Challenge 2012 |
| **N fichiers (Learning_set)** | 8 384 fichiers acc + 850 fichiers temp (6 roulements) |
| **Pertinence Gap 1** | **Seul dataset réel industriel** du projet — contribution directe au Gap 1 |

> **Utilisation** : uniquement le **Learning_set** (6 roulements avec run-to-failure complet, ~588 Mo CSV ou 883 Mo `.npy`). Les Test_set et Full_Test_Set (>2 Go) ne sont pas utilisés dans les expériences Python.

### Structure des fichiers

```
data/raw/Pronostia dataset/
├── Learning_set/
│   ├── Bearing1_1/   ← Condition 1 (1800 rpm, 4000 N)
│   ├── Bearing1_2/   ← Condition 1
│   ├── Bearing2_1/   ← Condition 2 (1650 rpm, 4200 N)
│   ├── Bearing2_2/   ← Condition 2
│   ├── Bearing3_1/   ← Condition 3 (1500 rpm, 5000 N)
│   └── Bearing3_2/   ← Condition 3
└── binaries/
    ├── Bearing1_1.npy   # shape: (n_files, 2560, 2)
    └── ...              # acc_X et acc_Y pré-chargés
```

**Format des fichiers acc** (6 colonnes, sans header) : `heure, min, sec, ms, acc_X, acc_Y`

### Features extraites — Pronostia

Feature engineering par fenêtre (1 fichier acc = 2560 points) :
6 statistiques × 2 axes (acc_X, acc_Y) = **12 features** par fenêtre.

Statistiques calculées : mean, std, rms, kurtosis, peak, crest_factor.

> **Label binaire** : les 20% dernières fenêtres temporelles de chaque bearing = `faulty=1` (dégradation avancée). Paramètre `ANOMALY_LAST_PCT = 0.20` dans `configs/unsupervised_config.yaml`.

### Scénario CL — Pronostia

**Domain-Incremental par condition opératoire** :

| Tâche | Bearings | Condition | Vitesse | Charge |
|-------|----------|-----------|---------|--------|
| Task 0 | Bearing1_1, Bearing1_2 | Condition 1 | 1800 rpm | 4000 N |
| Task 1 | Bearing2_1, Bearing2_2 | Condition 2 | 1650 rpm | 4200 N |
| Task 2 | Bearing3_1, Bearing3_2 | Condition 3 | 1500 rpm | 5000 N |

### Pipeline de preprocessing — Pronostia

```python
# src/data/pronostia_dataset.py

N_FEATURES: int = 12           # 6 stats × 2 axes (acc_X, acc_Y)
WINDOW_SIZE: int = 2560        # = 1 fichier acc complet
ANOMALY_LAST_PCT: float = 0.20 # last 20% fenêtres d'un bearing = faulty=1
N_TASKS: int = 3               # 3 conditions opératoires
PROCESSED_PATH = "data/processed/pronostia/features.csv"  # gitignored, re-calculable
```

### Chargement recommandé — Pronostia

```python
from src.data.pronostia_dataset import preprocess_to_csv, get_pronostia_dataloaders
import yaml

# 1. Préprocessing (no-op si CSV déjà généré)
preprocess_to_csv()  # lit binaries/*.npy, extrait 12 features, sauvegarde CSV

# 2. Chargement
cfg = yaml.safe_load(open("configs/unsupervised_config.yaml"))["pronostia"]
tasks = get_pronostia_dataloaders(cfg)
# → [{task_id: 0, condition: "1800rpm", train_loader, val_loader, n_train, n_val}, ...]
```

---

## Comparaison et rôles dans le projet

| Critère | Dataset 1 — Pump | Dataset 2 — Monitoring | Dataset 3 — Battery RUL | Dataset 4 — Pronostia |
|---------|-----------------|----------------------|------------------------|----------------------|
| Nature | Simulé | Simulé | Simulé/mesuré | **Réel** ✅ |
| Type | Séries temporelles | Tabulaire statique | Tabulaire séquentiel | Séries temporelles brutes |
| Réalisme CL | ✅ Drift temporel | ⚠️ Frontières artificielles | ✅ Drift temporel | ✅ Conditions opératoires réelles |
| Feature engineering | ✅ (sliding window) | ❌ (prêt) | ❌ (prêt) | ✅ (sliding window .npy) |
| N features | 25 | 4 | 7 | 12 |
| N tâches | 3 | 3 | 3 | 3 |
| Modèles associés | M1 TinyOL | M2 EWC, M3 HDC | Non-supervisés (S5-16) | Non-supervisés (S5-16) |
| Pertinence Gap 1 | ⚠️ Simulé | ⚠️ Simulé | ⚠️ Simulé | **✅ Réel — contribution directe** |
| Expériences | exp_001, exp_002 | exp_003 à exp_008 | exp_009 | exp_010 |

---

## Téléchargement des datasets Kaggle

```bash
# Avec l'API Kaggle (kaggle.json requis dans ~/.kaggle/)
kaggle datasets download -d [slug_dataset_1] -p data/raw/pump_maintenance/
kaggle datasets download -d [slug_dataset_2] -p data/raw/equipment_monitoring/

# Décompression
unzip data/raw/pump_maintenance/*.zip -d data/raw/pump_maintenance/
unzip data/raw/equipment_monitoring/*.zip -d data/raw/equipment_monitoring/
```

> TODO : renseigner les slugs Kaggle exacts une fois les datasets identifiés précisément.

---

## .gitignore — Données

Les données brutes et processées ne sont jamais committées :

```gitignore
data/raw/
data/processed/
data/*.csv
data/*.zip
```

Seuls les fichiers de configuration (normaliseurs, stats) sont versionnés :

```
configs/pump_normalizer.yaml       ← committable
configs/monitoring_normalizer.yaml ← committable
```
