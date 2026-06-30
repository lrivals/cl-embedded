# S2201–S2204 — CMAPSS : EDA + Loader + Feature Engineering + Config

| Champ | Valeur |
|-------|--------|
| **Sprint** | 22 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Terminé |
| **Durée estimée** | 30 min + 2h + 2h + 30 min = 5h |
| **Dépendances** | Sprint 21 ✅ — scripts `train_ewc.py`, `train_hdc.py`, `train_tinyol.py` opérationnels |
| **Fichiers cibles** | `src/data/cmapss_loader.py`, `configs/cmapss_config.yaml`, `notebooks/eda_cmapss.ipynb` |
| **Référence** | `src/data/monitoring_dataset.py` (pattern loader), `Hurtado2023CLPdM`, `DeLange2021Survey` |

---

## Contexte

CMAPSS (NASA C-MAPSS Turbofan Engine Degradation Simulation) est le premier des 2 nouveaux datasets ajoutés en Sprint 22 pour combler **Gap 1** (validation sur données industrielles réelles diversifiées). Il contient 4 sous-datasets (FD001–FD004) correspondant à des conditions opératoires différentes — naturellement adaptés à un scénario **domain-incremental** CL.

La particularité CMAPSS : le label natif est le **RUL continu** (Remaining Useful Life). Il faut le **binariser** en `faulty = 1 si RUL ≤ 30` pour compatibilité avec les scripts existants (EWC, HDC, TinyOL, Mahalanobis).

Aucun fichier `cmapss_loader.py` n'existe encore dans `src/data/`. Le pattern à suivre est `src/data/monitoring_dataset.py`.

---

## S2201 — Download CMAPSS

**Source** : NASA Prognostics Center of Excellence Data Repository / Kaggle (`cmapss-jet-engine-simulated-data`)

**Structure cible** :
```
data/raw/cmapss/
├── train_FD001.txt
├── train_FD002.txt
├── train_FD003.txt
├── train_FD004.txt
├── test_FD001.txt
├── test_FD002.txt
├── test_FD003.txt
├── test_FD004.txt
└── RUL_FD001.txt  (... RUL_FD004.txt)
```

**Format des fichiers** : espace-séparé, sans en-tête. Colonnes (26 au total) :
```
unit_nr | time_cycles | op_setting_1 | op_setting_2 | op_setting_3 | s1..s21
```

**Vérification** :
```bash
wc -l data/raw/cmapss/train_FD001.txt   # ~20631 lignes attendues
head -1 data/raw/cmapss/train_FD001.txt | wc -w  # 26 colonnes
```

---

## S2202 — `src/data/cmapss_loader.py`

### Fonctionnalités requises

1. Lecture des 4 fichiers `train_FDxxx.txt` via `pandas.read_csv(sep=' ')`
2. Calcul du **RUL** par unité moteur : `RUL = max_cycles - time_cycles`
3. **RUL capping** à 125 (cap=125) : `RUL = min(RUL, 125)` — évite les valeurs extrêmes en début de vie
4. **Binarisation** : `faulty = 1 si RUL ≤ 30` (seuil de défaillance imminente)
5. **Feature selection top-5** par mutual info avec `faulty` (sklearn)
6. Normalisation MinMax (fit sur FD001 uniquement, appliqué à FD001–FD004)
7. Interface `get_cl_dataloaders()` identique à `monitoring_dataset.py`

### Structure du module

```python
"""
cmapss_loader.py — Loader PyTorch pour CMAPSS (NASA Turbofan Degradation).

Scénario CL : Domain-Incremental
    Task 1 = FD001 → Task 2 = FD002 → Task 3 = FD003 → Task 4 = FD004

RUL binarisé : faulty = 1 si RUL ≤ 30 (seuil défaillance imminente).
RUL capping : cap = 125 cycles (défini dans CMAPSS_RUL_CAP).
Normalisation MinMax fixée sur FD001 uniquement.

Usage :
    from src.data.cmapss_loader import get_cl_dataloaders
    tasks = get_cl_dataloaders(data_dir=Path("data/raw/cmapss/"),
                               config_path=Path("configs/cmapss_config.yaml"))
"""

# Constantes (toutes ici, jamais en dur dans les fonctions)
CMAPSS_RUL_CAP: int = 125
CMAPSS_FAULTY_THRESHOLD: int = 30  # RUL ≤ 30 → faulty = 1
CMAPSS_N_FEATURES_RAW: int = 21    # capteurs s1–s21
CMAPSS_N_FEATURES_SELECTED: int = 5

# Noms des 21 capteurs (index dans le fichier texte brut, colonnes 5–25)
SENSOR_NAMES: list[str] = [
    "T2", "T24", "T30", "T50", "P2", "P15", "P30",
    "Nf", "Nc", "epr", "Ps30", "phi", "NRf", "NRc",
    "BPR", "farB", "htBleed", "Nf_dmd", "PCNfR_dmd", "W31", "W32",
]

# Ordre des domaines CL (4 tâches)
DOMAIN_ORDER: list[str] = ["FD001", "FD002", "FD003", "FD004"]
```

### Fonctions à implémenter

```python
def _load_raw(data_dir: Path, subset: str) -> pd.DataFrame:
    """Lit train_FDxxx.txt, attribue les noms de colonnes, calcule RUL cappé et faulty."""
    ...

def compute_feature_selection(data_dir: Path, n_features: int = 5) -> list[str]:
    """
    Calcule le top-N par mutual_info_classif sur FD001 (fit uniquement).
    Retourne les noms de capteurs sélectionnés.
    Sauvegarde dans configs/cmapss_feature_subset.yaml.
    """
    ...

def get_cl_dataloaders(
    data_dir: Path,
    config_path: Path,
    feature_names: list[str] | None = None,
) -> list[dict]:
    """
    Retourne une liste de dicts par tâche CL :
    [{"task_id": 1, "domain": "FD001", "train_loader": ..., "val_loader": ...,
      "n_train": int, "n_val": int, "class_weights": Tensor}]
    """
    ...
```

### Vérification

```bash
python -c "
from pathlib import Path
from src.data.cmapss_loader import get_cl_dataloaders
tasks = get_cl_dataloaders(Path('data/raw/cmapss/'), Path('configs/cmapss_config.yaml'))
for t in tasks:
    x, y = next(iter(t['train_loader']))
    print(f\"{t['domain']} — x:{x.shape}, y:{y.shape}, faulty_rate:{y.mean():.3f}\")
assert x.shape[1] == 5, 'N_FEATURES != 5'
print('cmapss_loader OK')
"
```

---

## S2203 — Notebook EDA `notebooks/eda_cmapss.ipynb`

### Sections requises

1. **Distribution RUL brut vs cappé** (FD001) — histogramme avec ligne `cap=125`
2. **Drift domaine** : distribution features top-5 par FD (boxplot 4 sous-datasets côte à côte)
3. **Matrice de corrélation** : 21 capteurs × RUL (heatmap seaborn)
4. **Ranking mutual info** : barplot top-5 vs bottom-5 capteurs
5. **Taux de défaut** par FD après binarisation (tableau + barplot)
6. **Justification du seuil RUL ≤ 30** : discussion domaine + lien `TODO(arnaud)` pour validation

```python
# Cellule à inclure obligatoirement — justification Gap 1
print("CMAPSS contribue à Gap 1 : validation sur données industrielles réelles")
print("FD001→FD004 = conditions opératoires différentes (scénario domain-incremental)")
print("Binarisation RUL ≤ 30 : choix à valider avec Arnaud — voir TODO(arnaud)")
```

---

## S2204 — `configs/cmapss_config.yaml`

```yaml
# configs/cmapss_config.yaml
# CMAPSS — NASA C-MAPSS Turbofan (4 sous-datasets domain-incremental)
# Binarisation RUL ≤ 30, cap 125, top-5 features mutual info
# NE PAS modifier : créer une copie pour chaque variante expérimentale.

exp_id: "exp_S22_cmapss_base"

model:
  architecture: "mlp"
  input_dim: 5             # top-5 capteurs sélectionnés par mutual info
  hidden_dims: [32, 16]
  output_dim: 1
  activation: "relu"
  dropout: 0.2

training:
  optimizer: "sgd"
  learning_rate: 0.01
  momentum: 0.9
  epochs_per_task: 10
  batch_size: 32
  seed: 42

ewc:
  lambda: 1000
  gamma: 0.9
  n_fisher_samples: 200

data:
  dataset: "cmapss"
  data_dir: "data/raw/cmapss/"
  rul_cap: 125
  faulty_threshold: 30     # TODO(arnaud): valider seuil RUL ≤ 30
  n_features_selected: 5
  feature_subset_path: "configs/cmapss_feature_subset.yaml"
  domain_order: ["FD001", "FD002", "FD003", "FD004"]
  label_column: "faulty"
  test_split: 0.2
  normalizer: "minmax"     # fit sur FD001 uniquement

evaluation:
  seed: 42
  metrics: ["aa", "af", "bwt", "ram_peak_bytes", "n_params", "inference_latency_ms"]
  output_dir: "experiments/"

memory:
  target_ram_bytes: 262144   # 256 Ko NUCLEO-F439ZI
  expected_ram_bytes: 9356   # identique monitoring (même architecture input_dim=5)
```

---

## Questions ouvertes

- `TODO(arnaud)` : Valider le seuil `RUL ≤ 30` comme critère de défaillance imminente. Alternative : RUL ≤ 15 (plus conservateur) ou RUL ≤ 50 (plus permissif) ?
- `TODO(arnaud)` : Les capteurs constants dans FD001 (s1, s5, s6, s10, s16, s18, s19) doivent-ils être exclus avant la sélection mutual info ?
- `FIXME(gap1)` : Documenter dans `docs/datasets_analysis.md` en quoi CMAPSS complète Gap 1 par rapport à CWRU et Monitoring.
