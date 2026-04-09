# S5-15 — EDA et loaders : Battery RUL + Pronostia

| Champ | Valeur |
|-------|--------|
| **ID** | S5-15 |
| **Sprint** | Sprint 5 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 7.5h |
| **Dépendances** | S5-01 (structure + config YAML), S5-05 (`train_unsupervised.py`) |
| **Fichiers cibles** | `src/data/battery_rul_dataset.py`, `src/data/pronostia_dataset.py`, `docs/context/datasets.md`, `configs/battery_rul_normalizer.yaml`, `configs/pronostia_normalizer.yaml`, `notebooks/01_data_exploration.ipynb` |

---

## Objectif

Intégrer deux nouveaux datasets dans le pipeline du projet pour enrichir l'évaluation des méthodes non-supervisées (S5-16) :

- **Battery RUL** : dataset tabulaire de dégradation de batteries lithium-ion (15 K cycles, RUL explicite décroissant 1112 → 0). Scénario CL domain-incremental temporel.
- **Pronostia** (FEMTO / IEEE PHM 2012) : données réelles de roulements en run-to-failure (vibrations 3 axes + température). Utilisé jusqu'ici comme référence scientifique uniquement — promu en dataset actif via le Learning_set (6 roulements, ~588 Mo) et les binaires `.npy` pré-calculés.

**Critères de succès** :
- `from src.data.battery_rul_dataset import get_battery_dataloaders` retourne 3 dicts sans erreur
- `from src.data.pronostia_dataset import get_pronostia_dataloaders` retourne 3 dicts (après `preprocess_to_csv()`)
- Figures EDA générées dans `notebooks/figures/eda/battery_rul/` et `notebooks/figures/eda/pronostia/`
- `docs/context/datasets.md` mis à jour (Dataset 3 = Battery RUL, Dataset 4 = Pronostia actif)

---

## Tâches

| ID | Tâche | Priorité | Fichier cible | Durée est. | Dépendances |
|----|-------|:---:|---------------|:---:|-------------|
| S5-15a | Mettre à jour `docs/context/datasets.md` — Dataset 3 Battery RUL (spec complète) + Dataset 4 Pronostia (passer de "référence" à "actif", Learning_set uniquement) | 🔴 | `docs/context/datasets.md` | 0.5h | — |
| S5-15b | Implémenter `battery_rul_dataset.py` — load CSV, renommage colonnes snake_case, labellisation `RUL ≤ ANOMALY_THRESHOLD → faulty=1`, split 3 tâches chronologiques, normalizer Z-score YAML | 🔴 | `src/data/battery_rul_dataset.py` | 2h | S5-15a |
| S5-15c | Implémenter `pronostia_dataset.py` — `preprocess_to_csv()` depuis `.npy` Learning_set, extraction 12 features (6 stats × 2 axes acc_X/acc_Y), `label_by_rul()` (last 20% life = faulty=1), split 3 conditions opératoires, normalizer YAML | 🔴 | `src/data/pronostia_dataset.py` | 3h | S5-15a |
| S5-15d | EDA Battery RUL dans `notebooks/01_data_exploration.ipynb` — courbe RUL vs cycle, distribution des 7 features, histogramme par label (normal/anomalie), matrice de corrélations | 🟡 | `notebooks/01_data_exploration.ipynb` | 1h | S5-15b |
| S5-15e | EDA Pronostia dans `notebooks/01_data_exploration.ipynb` — courbe de dégradation RMS par bearing, distribution features par condition opératoire, proportion anomalie par tâche | 🟡 | `notebooks/01_data_exploration.ipynb` | 1h | S5-15c |

**Livrable** : 2 loaders fonctionnels, normalizers YAML générés, figures EDA, `docs/context/datasets.md` mis à jour.

---

## Notes d'implémentation

### Battery RUL — `src/data/battery_rul_dataset.py`

#### Colonnes brutes → convention pipeline

| Colonne CSV brute | Rename pipeline | Description |
|-------------------|-----------------|-------------|
| `Cycle_Index` | `cycle_index` | Proxy temporel (tri croissant, non utilisé comme feature) |
| `Discharge Time (s)` | `discharge_time_s` | Durée de décharge complète |
| `Decrement 3.6-3.4V (s)` | `decrement_3640v_s` | Durée dans la fenêtre de tension critique |
| `Max. Voltage Dischar. (V)` | `max_voltage_discharge_v` | Tension maximale en décharge |
| `Min. Voltage Charg. (V)` | `min_voltage_charge_v` | Tension minimale en charge |
| `Time at 4.15V (s)` | `time_at_415v_s` | Durée à tension de charge maximale |
| `Time constant current (s)` | `time_constant_current_s` | Phase courant constant (CCCV) |
| `Charging time (s)` | `charging_time_s` | Durée totale de charge |
| `RUL` | `rul` | Label continu (0–1112) — converti en binaire |

#### Constantes du module

```python
# src/data/battery_rul_dataset.py
# Toujours modifier via configs/unsupervised_config.yaml — ne pas hardcoder ici

RAW_FILENAME = "Battery_RUL.csv"
N_FEATURES: int = 7            # colonnes numériques hors cycle_index et rul
N_TASKS: int = 3               # découpage chronologique par tiers de cycles
ANOMALY_THRESHOLD: int = 200   # RUL ≤ 200 → faulty=1 (≈ 18% du max)
VAL_RATIO: float = 0.2
```

#### Interface publique

```python
def get_battery_dataloaders(
    config: dict,
    data_root: str = "data/raw/Battery Remaining Useful Life (RUL)/",
    normalizer_path: str = "configs/battery_rul_normalizer.yaml",
) -> list[dict]:
    """
    Charge Battery RUL et retourne 3 dicts domain-incremental.

    Returns
    -------
    list[dict]
        [{task_id, train_loader, val_loader, n_train, n_val}, ...]
        task_id 0 = premiers cycles (batterie saine)
        task_id 2 = derniers cycles (batterie dégradée)
    """
```

```python
# MEM: N_FEATURES × 4 B @ FP32 par sample = 28 B → négligeable
```

#### Scénario CL

- **Type** : Domain-incremental temporel (identique à Dataset 1 — Pump)
- **Découpage** : 3 tiers égaux par `cycle_index` (tiers 0 = batterie neuve, tiers 2 = fin de vie)
- **Normalisation** : Z-score fitté sur Task 0 uniquement (même contrainte que `pump_dataset.py`)
- **Label** : `faulty = int(rul <= ANOMALY_THRESHOLD)` — binaire, calculé avant le split

---

### Pronostia — `src/data/pronostia_dataset.py`

#### Structure des données

```
data/raw/Pronostia dataset/Learning_set/
├── Bearing1_1/   ← Condition 1 (1800 rpm, 4000 N)
├── Bearing1_2/   ← Condition 1
├── Bearing2_1/   ← Condition 2 (1650 rpm, 4200 N)
├── Bearing2_2/   ← Condition 2
├── Bearing3_1/   ← Condition 3 (1500 rpm, 5000 N)
└── Bearing3_2/   ← Condition 3

data/raw/Pronostia dataset/binaries/
├── Bearing1_1.npy   # shape: (n_files, 2560, 2) — acc_X, acc_Y
├── Bearing1_2.npy
└── ...  (6 fichiers au total)
```

#### Features extraites (par fichier acc = 1 fenêtre de 2560 points)

Pour chaque axe (acc_X, acc_Y) : mean, std, rms, kurtosis, peak, crest_factor → **12 features** par fenêtre.

```python
# MEM: 12 × 4 B @ FP32 = 48 B par sample
N_FEATURES: int = 12
WINDOW_SIZE: int = 2560   # = 1 fichier acc complet (~0.1s à 25.6 kHz)
ANOMALY_LAST_PCT: float = 0.20   # last 20% des fenêtres d'un bearing = faulty=1
N_TASKS: int = 3   # 3 conditions opératoires
```

#### Constantes du module

```python
LEARNING_SET_PATH = "data/raw/Pronostia dataset/Learning_set/"
BINARIES_PATH = "data/raw/Pronostia dataset/binaries/"
PROCESSED_PATH = "data/processed/pronostia/features.csv"  # gitignored, re-calculable
N_FEATURES: int = 12
ANOMALY_LAST_PCT: float = 0.20
N_TASKS: int = 3
VAL_RATIO: float = 0.2
```

#### Interface publique

```python
def preprocess_to_csv(
    binaries_path: str = BINARIES_PATH,
    output_path: str = PROCESSED_PATH,
    force_recompute: bool = False,
) -> None:
    """
    Charge les .npy du Learning_set, extrait 12 features par fenêtre,
    labellise les 20% dernières fenêtres de chaque bearing (faulty=1),
    sauvegarde dans data/processed/pronostia/features.csv.
    No-op si output_path existe déjà (sauf force_recompute=True).
    """

def get_pronostia_dataloaders(
    config: dict,
    processed_path: str = PROCESSED_PATH,
    normalizer_path: str = "configs/pronostia_normalizer.yaml",
) -> list[dict]:
    """
    Charge le CSV préprocessé et retourne 3 dicts domain-incremental.

    Returns
    -------
    list[dict]
        [{task_id, condition, train_loader, val_loader, n_train, n_val}, ...]
        task_id 0 = Condition 1 (1800 rpm), 1 = Condition 2, 2 = Condition 3
    """
```

#### Scénario CL

- **Type** : Domain-incremental par condition opératoire (3 conditions de charge/vitesse)
- **Tâche 0** : Condition 1 — Bearing1_1, Bearing1_2 (1800 rpm, 4000 N)
- **Tâche 1** : Condition 2 — Bearing2_1, Bearing2_2 (1650 rpm, 4200 N)
- **Tâche 2** : Condition 3 — Bearing3_1, Bearing3_2 (1500 rpm, 5000 N)
- **Normalisation** : Z-score fitté sur Task 0 uniquement (Bearing1_1 + Bearing1_2)
- **Label** : `faulty = 1` pour les 20% dernières fenêtres temporelles de chaque bearing (run-to-failure)

#### Note sur les binaires `.npy`

Les fichiers `.npy` du dossier `binaries/` sont des arrays 3D de shape `(n_files, 2560, 2)` :
- `n_files` = nombre de fichiers `acc_XXXXX.csv` de ce bearing
- `2560` = points par fichier acc
- `2` = colonnes acc_X (index 4) et acc_Y (index 5) des CSV bruts

Utiliser les `.npy` plutôt que les 8384 CSV bruts évite ~30s de chargement.

---

## Questions ouvertes

- `TODO(arnaud)` : `ANOMALY_THRESHOLD = 200` pour Battery RUL (≈18% du max RUL) — est-ce pertinent comme seuil de criticité ? Alternativement, utiliser `RUL ≤ 50` (dernier 4.5%) pour ne cibler que les phases proches de la panne.
- `TODO(arnaud)` : Pronostia — le split par condition (3 tâches) est-il le bon scénario CL, ou vaut-il mieux splitter par bearing (6 tâches) pour avoir plus de granularité ?
- `TODO(arnaud)` : `ANOMALY_LAST_PCT = 0.20` pour Pronostia — les 20% derniers fichiers d'un bearing correspondent à une durée variable (Bearing1_1 dure 2h16, Bearing3_2 dure 5h11). Faut-il labelliser par durée absolue (ex. dernier 30 min) plutôt que par proportion ?
- `FIXME(gap1)` : Pronostia est le **seul dataset réel industriel** du projet — à valoriser explicitement dans le manuscrit comme contribution au Gap 1 (données industrielles réelles de dégradation).
- `FIXME(gap2)` : vérifier que `ram_peak_bytes` (tracemalloc) reste < 64 Ko pour les 5 détecteurs sur N_FEATURES=7 (Battery) et N_FEATURES=12 (Pronostia).
