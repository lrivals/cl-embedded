# Architecture du projet — CL-Embedded

> Ce document décrit l'architecture complète du dépôt, les responsabilités de chaque module,
> et les conventions d'interface entre composants.

---

## Arborescence complète

```
cl-embedded/
│
├── CLAUDE.md                          ← Context Claude Code (LIRE EN PREMIER)
├── README.md                          ← Documentation GitHub publique
├── LICENSE                            ← MIT
├── .gitignore
├── pyproject.toml                     ← Dépendances + config linting
├── requirements.txt                   ← Dépendances pip
│
├── configs/                           ← Hyperparamètres (versionnés, jamais hardcodés)
│   ├── ewc_config.yaml
│   ├── hdc_config.yaml
│   ├── tinyol_config.yaml
│   ├── pump_normalizer.yaml           ← Stats normalisation Dataset 1
│   └── monitoring_normalizer.yaml     ← Stats normalisation Dataset 2
│
├── data/                              ← GITIGNORED
│   ├── raw/
│   │   ├── pump_maintenance/
│   │   └── equipment_monitoring/
│   └── processed/
│       ├── pump_features.npz
│       └── monitoring_features.npz
│
├── docs/
│   ├── models/
│   │   ├── tinyol_spec.md             ← Spec complète M1
│   │   ├── ewc_mlp_spec.md            ← Spec complète M2
│   │   └── hdc_spec.md                ← Spec complète M3
│   ├── context/
│   │   ├── project_overview.md        ← Vue macro du projet
│   │   ├── hardware_constraints.md    ← STM32N6 + règles de portabilité
│   │   ├── datasets.md                ← Description datasets + protocoles
│   │   └── triple_gap.md              ← Positionnement scientifique
│   └── roadmap.md                     ← Sprints + backlog
│
├── skills/                            ← Guides prompting Claude (ne pas modifier sans raison)
│   ├── sprint_generation.md
│   ├── model_implementation.md
│   ├── cl_evaluation.md
│   ├── embedded_portability.md
│   ├── latex_manuscript.md
│   └── code_review.md
│
├── src/                               ← Code source principal
│   ├── __init__.py
│   │
│   ├── data/                          ← Loaders et preprocessing
│   │   ├── __init__.py
│   │   ├── pump_dataset.py            ← Dataset 1 : fenêtrage + features
│   │   ├── monitoring_dataset.py      ← Dataset 2 : tabulaire + split domaine
│   │   └── cl_stream.py               ← Abstraction générique stream CL
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_cl_model.py           ← Classe abstraite commune aux 3 modèles
│   │   │
│   │   ├── tinyol/                    ← M1
│   │   │   ├── __init__.py
│   │   │   ├── autoencoder.py         ← Backbone (encodeur + décodeur)
│   │   │   └── oto_head.py            ← Tête OtO + boucle online
│   │   │
│   │   ├── ewc/                       ← M2
│   │   │   ├── __init__.py
│   │   │   ├── ewc_mlp.py             ← MLP + perte EWC Online
│   │   │   └── fisher.py              ← Calcul Fisher diagonale
│   │   │
│   │   └── hdc/                       ← M3
│   │       ├── __init__.py
│   │       ├── hdc_classifier.py      ← Encodage + prototypes + inférence
│   │       └── base_vectors.py        ← Génération + sauvegarde vecteurs de base
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── cl_trainer.py              ← Boucle d'entraînement CL générique
│   │   ├── scenarios.py               ← Construction des streams CL
│   │   └── baselines.py               ← Fine-tuning naïf + joint training
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                 ← AA, AF, BWT, FWT
│   │   ├── memory_profiler.py         ← tracemalloc + rapport RAM
│   │   └── plots.py                   ← Visualisations (accuracy matrix, etc.)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── reproducibility.py         ← set_seed()
│       ├── quantization.py            ← UINT8 encode/decode embeddings
│       ├── feature_engineering.py     ← Features statistiques (fenêtres)
│       └── config_loader.py           ← Chargement YAML + validation
│
├── experiments/                       ← Résultats reproductibles (versionnés)
│   ├── exp_001_ewc_dataset2/
│   │   ├── config_snapshot.yaml       ← Config exacte utilisée
│   │   └── results/
│   │       ├── metrics.json
│   │       ├── accuracy_matrix.png
│   │       └── forgetting_curve.png
│   ├── exp_002_hdc_dataset2/
│   ├── exp_003_tinyol_dataset1/
│   └── exp_004_tinyol_uint8/
│
├── notebooks/                         ← Exploration et visualisation
│   ├── 01_data_exploration.ipynb      ← EDA des deux datasets
│   ├── 02_baseline_comparison.ipynb   ← EWC vs HDC vs Fine-tuning
│   ├── 03_cl_evaluation.ipynb         ← Évaluation TinyOL + Dataset 1
│   └── 04_final_comparison.ipynb      ← Tableau comparatif 3 modèles
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    ← Fixtures pytest partagées
│   ├── test_ewc.py
│   ├── test_hdc.py
│   ├── test_tinyol.py
│   ├── test_metrics.py
│   └── test_memory_profiler.py
│
└── scripts/                           ← Points d'entrée CLI
    ├── train_ewc.py
    ├── train_hdc.py
    ├── train_tinyol.py
    ├── pretrain_tinyol.py             ← Pré-entraînement backbone autoencoder
    ├── evaluate_all.py                ← Évaluation comparative tous modèles
    ├── profile_memory.py              ← Profiling RAM systématique
    └── export_onnx.py                 ← Export et validation ONNX
```

---

## Interfaces entre composants

### Interface CLDataset (abstraite)

Tous les loaders implementent cette interface :

```python
class CLDataset:
    def get_task_data(task_id: int) -> Tuple[np.ndarray, np.ndarray]
    def n_tasks: int
    def n_features: int
    def n_classes: int
    def task_names: List[str]
```

### Interface BaseCLModel (abstraite)

Tous les modèles implementent :

```python
class BaseCLModel:
    def predict(x: Tensor) -> Tensor                    # inférence
    def update(x: Tensor, y: Tensor) -> float           # mise à jour online
    def on_task_end(task_id: int, dataloader) -> None   # fin de tâche (Fisher, etc.)
    def estimate_ram_bytes(dtype: str) -> int           # budget mémoire
    def count_parameters() -> int
    def save(path: str) -> None
    def load(path: str) -> None
```

### Interface CLTrainer

```python
class CLTrainer:
    def run(model: BaseCLModel, stream: CLDataset, cfg: dict) -> dict
    # Retourne : {"acc_matrix": ..., "memory": ..., "config": ...}
```

---

## Dépendances

```toml
# pyproject.toml

[project]
name = "cl-embedded"
python_requires = ">=3.10"

dependencies = [
    "torch>=2.0",
    "numpy>=1.24",
    "scikit-learn>=1.3",
    "pandas>=2.0",
    "pyyaml>=6.0",
    "onnx>=1.14",
    "onnxruntime>=1.16",
    "matplotlib>=3.7",
    "seaborn>=0.12",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "black>=23.0",
    "ruff>=0.1",
    "jupyter>=1.0",
]
```

---

## Conventions Git

### Branches
- `main` — stable, résultats reproductibles uniquement
- `dev` — développement en cours
- `exp/[nom]` — branches d'expérimentation (fusionner dans dev si concluant)

### Messages de commit
```
feat(ewc): implement Fisher diagonal computation
fix(hdc): correct prototype binarization for edge case
exp(001): add EWC baseline results on dataset2
docs: update roadmap sprint 2
refactor(data): unify CLDataset interface
```

### Ce qui est toujours committé
- `src/`, `tests/`, `scripts/`, `configs/`, `docs/`, `skills/`
- `experiments/*/config_snapshot.yaml`
- `experiments/*/results/metrics.json`
- `experiments/*/results/*.png`

### Ce qui n'est jamais committé
- `data/` (toutes les données)
- `experiments/*/checkpoint.pt` (trop lourd)
- `.env`, clés API, chemins absolus
