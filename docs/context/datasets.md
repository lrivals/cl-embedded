# Datasets — Description et protocoles d'utilisation

---

## Dataset 1 — Large Industrial Pump Maintenance Dataset

| Propriété | Valeur |
|-----------|--------|
| **Source** | Kaggle |
| **Chemin local** | `data/raw/pump_maintenance/` |
| **Type** | Séries temporelles multivariées |
| **Nature** | Simulé (réaliste) |
| **Tâche ML** | Classification binaire (maintenance_required: 0/1) |

### Variables

| Colonne | Type | Description |
|---------|------|-------------|
| `timestamp` | datetime | Horodatage de la mesure |
| `temperature` | float (°C) | État thermique de la machine |
| `vibration` | float | Indicateur de défaut mécanique |
| `pressure` | float | Pression interne du système |
| `rpm` | float | Vitesse de rotation |
| `maintenance_required` | int (0/1) | Label binaire |

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

## Comparaison et rôles dans le projet

| Critère | Dataset 1 — Pump | Dataset 2 — Monitoring |
|---------|-----------------|----------------------|
| Réalisme CL | ✅ Drift temporel naturel | ⚠️ Frontières artificielles |
| Complexité pipeline | ⚠️ Feature engineering requis | ✅ Données prêtes |
| Prototypage rapide | ⚠️ | ✅ Idéal pour valider baselines |
| Modèles associés | M1 TinyOL | M2 EWC, M3 HDC |
| Scénario CL | Domain-incr. temporel | Domain-incr. par type équipement |
| Démarrage recommandé | Phase 2 (semaine 3–4) | Phase 1 (semaine 1–2) |

---

## Référence scientifique — FEMTO PRONOSTIA

> Ce dataset n'est PAS utilisé dans les expériences Python de ce dépôt.  
> Il sert uniquement d'argument scientifique de positionnement dans le manuscrit.

| Propriété | Valeur |
|-----------|--------|
| **Référence** | Nectoux et al. (2012) |
| **Type** | Dégradation de roulements (vibrations + température) |
| **Nature** | Données réelles (tests run-to-failure accélérés) |
| **Accès** | Public — IEEE PHM Challenge 2012 |
| **Pertinence Gap 1** | Seul dataset industriel de dégradation réel cité dans le corpus |

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
