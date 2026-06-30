# S2210–S2213 — Paderborn : EDA + Loader + Feature Engineering + Config

| Champ | Valeur |
|-------|--------|
| **Sprint** | 22 |
| **Priorité** | 🔴 (S2210, S2211) / 🟡 (S2212, S2213) |
| **Statut** | ✅ Complété (2026-06-01) |
| **Durée estimée** | 1h + 3h + 2h + 30 min = 6h30 |
| **Dépendances** | Sprint 21 ✅ — pipeline CL PC opérationnel |
| **Fichiers cibles** | `src/data/paderborn_loader.py`, `configs/paderborn_config.yaml`, `notebooks/eda_paderborn.ipynb` |
| **Référence** | `src/data/pronostia_dataset.py` (pattern signaux bruts → features), `Benatti2019HDC`, `Capogrosso2023TinyML` |

---

## Contexte

Le dataset Paderborn University (KAt-DataCenter) contient des signaux de **courant moteur + vibration accéléromètre** de roulements en différents états de défaut. C'est le second dataset ajouté en Sprint 22 pour Gap 1, apportant une **diversité de signal** par rapport à CWRU et Pronostia : le courant moteur est un capteur distinct de la vibration mécanique directe.

**Risque principal** : les fichiers bruts représentent ~2 Go. Télécharger uniquement 3 conditions (K001 healthy, OR fault, IR fault) pour rester sous 500 Mo.

Le scénario CL est **sain → défaut OR → défaut IR** (domain-incremental, 3 tâches).

Pattern de référence pour l'extraction de features depuis signaux bruts : `src/data/pronostia_dataset.py`.

---

## S2210 — Download Paderborn (subset K001 + OR + IR)

**Source** : [Paderborn University KAt-DataCenter](https://mb.uni-paderborn.de/kat/forschung/kat-datacenter)  
**Chemin cible** : `data/raw/paderborn/`

**Sous-sélection stricte** (< 500 Mo) :

| Condition | Fichiers | État |
|-----------|----------|------|
| K001 | K001_*.mat (healthy, artificially damaged) | Sain (y=0) |
| KA04 | KA04_*.mat (outer race fault, artificial) | OR fault (y=1) |
| KI04 | KI04_*.mat (inner race fault, artificial) | IR fault (y=1) |

**Structure cible** :
```
data/raw/paderborn/
├── K001/   # healthy — several .mat files (speed × load conditions)
│   ├── K001_1_1.mat
│   └── ...
├── KA04/   # outer race fault
│   ├── KA04_1_1.mat
│   └── ...
└── KI04/   # inner race fault
    ├── KI04_1_1.mat
    └── ...
```

**Vérification** :
```bash
du -sh data/raw/paderborn/   # doit être < 500 Mo
ls data/raw/paderborn/K001/ | wc -l   # nombre de fichiers .mat
```

---

## S2211 — `src/data/paderborn_loader.py`

### Fonctionnalités requises

1. **Lecture des `.mat`** via `scipy.io.loadmat()` — extraire le canal `vibration` (accéléromètre) et optionnellement `current` (courant moteur)
2. **Fenêtrage** : découpe en fenêtres de taille fixe (ex. 1024 samples) avec recouvrement 50%
3. **Feature engineering FFT** par fenêtre :
   - `rms` : √(mean(x²))
   - `kurtosis` : kurtosis statistique
   - `crest_factor` : max(|x|) / rms
   - `energy_band_1..4` : énergie spectrale dans 4 bandes fréquentielles (0-1kHz, 1-2kHz, 2-5kHz, 5-10kHz)
4. **Feature selection top-5** par mutual info sur K001 + KA04 (fit uniquement)
5. **Label binaire** : `faulty = 0` pour K001, `faulty = 1` pour KA04 et KI04
6. Interface `get_cl_dataloaders()` identique à `monitoring_dataset.py`

### Structure du module

```python
"""
paderborn_loader.py — Loader PyTorch pour Paderborn Bearing Dataset.

Scénario CL : Domain-Incremental
    Task 1 = K001 (healthy) → Task 2 = KA04 (OR fault) → Task 3 = KI04 (IR fault)

Feature engineering : FFT window-based (rms, kurtosis, crest_factor, energy_band_1..4)
Feature selection : top-5 mutual info (fit sur K001+KA04 uniquement).

Usage :
    from src.data.paderborn_loader import get_cl_dataloaders
    tasks = get_cl_dataloaders(data_dir=Path("data/raw/paderborn/"),
                               config_path=Path("configs/paderborn_config.yaml"))
"""

# Constantes
PADERBORN_WINDOW_SIZE: int = 1024
PADERBORN_OVERLAP: float = 0.5
PADERBORN_SAMPLING_RATE: int = 64_000   # Hz
PADERBORN_N_FEATURES_RAW: int = 7       # rms + kurtosis + crest + 4 bandes
PADERBORN_N_FEATURES_SELECTED: int = 5

# Bandes fréquentielles (Hz)
FREQ_BANDS: list[tuple[float, float]] = [
    (0, 1_000),
    (1_000, 2_000),
    (2_000, 5_000),
    (5_000, 10_000),
]

# Conditions CL
DOMAIN_ORDER: list[str] = ["K001", "KA04", "KI04"]
DOMAIN_LABELS: dict[str, int] = {"K001": 0, "KA04": 1, "KI04": 1}
```

### Fonctions à implémenter

```python
def _extract_windows(signal: np.ndarray, window_size: int, overlap: float) -> np.ndarray:
    """Découpe un signal 1D en fenêtres chevauchantes. Retourne [N_windows, window_size]."""
    ...

def _compute_features(windows: np.ndarray, fs: int) -> np.ndarray:
    """
    Calcule les 7 features time-freq par fenêtre.
    Retourne [N_windows, 7] : [rms, kurtosis, crest_factor, eb1, eb2, eb3, eb4].
    """
    from scipy.stats import kurtosis as scipy_kurtosis
    from numpy.fft import rfft, rfftfreq
    ...

def _load_condition(data_dir: Path, condition: str) -> tuple[np.ndarray, np.ndarray]:
    """Charge tous les .mat d'une condition, extrait features, retourne (X[N,7], y[N])."""
    ...

def compute_feature_selection(data_dir: Path, n_features: int = 5) -> list[str]:
    """Top-N mutual info sur K001+KA04. Sauvegarde configs/paderborn_feature_subset.yaml."""
    ...

def get_cl_dataloaders(
    data_dir: Path,
    config_path: Path,
    feature_names: list[str] | None = None,
) -> list[dict]:
    """Interface standard — même signature que monitoring_dataset.get_cl_dataloaders()."""
    ...
```

### Vérification

```bash
python -c "
from pathlib import Path
from src.data.paderborn_loader import get_cl_dataloaders
tasks = get_cl_dataloaders(Path('data/raw/paderborn/'), Path('configs/paderborn_config.yaml'))
for t in tasks:
    x, y = next(iter(t['train_loader']))
    print(f\"{t['domain']} — x:{x.shape}, y:{y.shape}, faulty_rate:{y.mean():.3f}\")
assert x.shape[1] == 5, 'N_FEATURES != 5'
print('paderborn_loader OK')
"
```

---

## S2212 — Notebook EDA `notebooks/eda_paderborn.ipynb`

### Sections requises

1. **Spectre FFT** : signal brut K001 vs KA04 vs KI04 — 3 colonnes côte à côte (1 fenêtre type)
2. **Distribution features** : violinplot 7 features × 3 conditions (avant sélection)
3. **Drift inter-condition** : t-SNE 2D coloré par condition K001/KA04/KI04
4. **Ranking mutual info** : barplot top-5 vs autres
5. **Taux de défaut** par condition après binarisation (vérification sanity : K001→0%, KA04+KI04→100%)
6. **Comparaison avec CWRU** : même famille (vibration roulement) — noter la différence signal courant moteur

```python
# Cellule obligatoire — lien avec littérature
print("Paderborn vs CWRU : même phénomène (défaut roulement)")
print("Originalité Paderborn : signal courant moteur en plus de la vibration")
print("Référence : Benatti2019HDC (signaux EEG de complexité similaire)")
```

---

## S2213 — `configs/paderborn_config.yaml`

```yaml
# configs/paderborn_config.yaml
# Paderborn Bearing Dataset — scénario domain-incremental (sain → OR → IR)
# Feature engineering FFT, top-5 mutual info
# NE PAS modifier : créer une copie pour chaque variante expérimentale.

exp_id: "exp_S22_paderborn_base"

model:
  architecture: "mlp"
  input_dim: 5             # top-5 features FFT sélectionnées
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
  dataset: "paderborn"
  data_dir: "data/raw/paderborn/"
  window_size: 1024
  overlap: 0.5
  sampling_rate: 64000     # Hz
  n_features_selected: 5
  feature_subset_path: "configs/paderborn_feature_subset.yaml"
  domain_order: ["K001", "KA04", "KI04"]
  domain_labels: {K001: 0, KA04: 1, KI04: 1}
  label_column: "faulty"
  test_split: 0.2
  normalizer: "minmax"     # fit sur K001 uniquement

evaluation:
  seed: 42
  metrics: ["aa", "af", "bwt", "ram_peak_bytes", "n_params", "inference_latency_ms"]
  output_dir: "experiments/"

memory:
  target_ram_bytes: 262144   # 256 Ko NUCLEO-F439ZI
  expected_ram_bytes: 9356   # même architecture input_dim=5
```

---

## Questions ouvertes

- `TODO(arnaud)` : Utiliser canal vibration seul ou combiner vibration + courant moteur ? (7 features brutes restent < 20 comme spécifié dans le sprint overview)
- `TODO(arnaud)` : Le fenêtrage 1024 samples @ 64 kHz = 16 ms de signal. Est-ce cohérent avec les conditions d'acquisition Paderborn (régime stationnaire) ?
- `FIXME(gap1)` : La comparaison Paderborn vs CWRU (même famille défaut roulement, signal différent) est un argument de diversité Gap 1 — à documenter dans `docs/datasets_analysis.md`.
