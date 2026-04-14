# S5-18 — Expériences granulaires Dataset 1 : scénario Pump par Pump_ID

| Champ | Valeur |
|-------|--------|
| **ID** | S5-18 |
| **Sprint** | Sprint 5 — Extension (≥ 12 avril 2026) |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 6h |
| **Dépendances** | S3-02 (`pump_dataset.py` existant), S5-10 (Mahalanobis existant), S5-15 (EDA par Pump_ID existante) |
| **Fichiers cibles** | `src/data/pump_dataset.py`, `configs/pump_by_id_config.yaml`, `experiments/exp_012–015/`, `src/evaluation/plots.py`, `notebooks/03_cl_evaluation.ipynb` |
| **Complété le** | — |
| **Statut** | ⬜ À faire |

---

## Objectif

Implémenter un nouveau scénario CL **domain-incremental par Pump_ID** sur le Dataset 1 (Pump
Maintenance). Au lieu des 3 tâches chronologiques (sain → usure → pré-panne) du scénario
actuel, chaque tâche correspond à un identifiant de pompe distinct (Pump_ID ∈ {1, 2, 3, 4, 5}),
donnant 5 tâches.

**Questions scientifiques ciblées** :
- Les pompes individuelles ont-elles des profils de panne distincts (distributions de features
  différentes) qui rendent le scénario plus ou moins difficile ?
- La granularité plus fine (5 vs 3 tâches) augmente-t-elle l'oubli catastrophique ?
- Certains modèles (HDC additif, Mahalanobis refit) gèrent-ils mieux ce type de shift inter-pompe ?

**Critère de succès** : 4 expériences (exp_012 à exp_015) exécutées, métriques AA/AF/BWT/RAM
enregistrées, plots générés dans `notebooks/figures/cl_evaluation/pump_by_id/`.

---

## Sous-tâches

### 1. Nouveau loader `get_pump_dataloaders_by_id()` dans `src/data/pump_dataset.py`

Ajouter une fonction qui retourne une liste de 5 dicts (un par Pump_ID), avec la même
interface que `get_pump_dataloaders()` :

```python
def get_pump_dataloaders_by_id(
    csv_path: str,
    normalizer_path: str,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> list[dict]:
    """
    Crée un scénario CL domain-incremental où chaque tâche correspond à un Pump_ID distinct.

    Parameters
    ----------
    csv_path : str
        Chemin complet vers le CSV pump maintenance.
    normalizer_path : str
        Chemin vers pump_normalizer.yaml (normaliseur ajusté sur Pump_ID=1).
    batch_size : int
        Taille de batch pour les DataLoaders.
    val_ratio : float
        Fraction validation (split stratifié sur Maintenance_Flag).
    seed : int
        Seed reproductibilité.

    Returns
    -------
    list[dict]
        Liste de 5 dicts, chacun avec keys : task_id, pump_id, train_loader, val_loader,
        n_train, n_val.
    """
```

**Points d'implémentation** :
- Trier les Pump_IDs dans l'ordre croissant (1 → 2 → 3 → 4 → 5)
- Appliquer le même feature engineering que `get_pump_dataloaders()` (fenêtrage 32/16,
  6 stats × 4 canaux + temporal_position)
- Normaliser avec le même `pump_normalizer.yaml` (ajusté sur Pump_ID=1, tâche 0)
- Split train/val stratifié par `Maintenance_Flag` au sein de chaque Pump_ID

### 2. Config YAML `configs/pump_by_id_config.yaml`

```yaml
# pump_by_id_config.yaml — Scénario CL par Pump_ID (5 tâches)
data:
  csv_path: /home/leonard/Documents/ENAC/cl-embedded/data/raw/pump_maintenance/Large Industrial_Pump_Maintenance_Dataset/Large_Industrial_Pump_Maintenance_Dataset.csv
  normalizer_path: configs/pump_normalizer.yaml
  task_split: by_pump_id        # nouveau mode : un Pump_ID par tâche
  pump_id_col: Pump_ID
  label_col: Maintenance_Flag
  feature_cols: [temperature, vibration, pressure, rpm]
  window_size: 32
  step_size: 16
  val_ratio: 0.2

# Reproduire les sections model/training des configs existantes (tinyol_config.yaml,
# ewc_pump_config.yaml, hdc_pump_config.yaml, unsupervised_config.yaml)
# selon le modèle utilisé à l'exécution.
```

### 3. Expériences exp_012 à exp_015

Chaque expérience réutilise les scripts et modèles existants avec le nouveau loader :

| Exp | Modèle | Script | Config supplémentaire |
|-----|--------|--------|----------------------|
| **exp_012_tinyol_pump_by_id** | TinyOL (backbone pré-entraîné sur Pump_ID=1 normales) | `scripts/train_tinyol.py` | `tinyol_config.yaml` + `pump_by_id_config.yaml` |
| **exp_013_ewc_pump_by_id** | EWC Online + MLP | `scripts/train_ewc.py` | `ewc_pump_config.yaml` + `pump_by_id_config.yaml` |
| **exp_014_hdc_pump_by_id** | HDC (D=1024, n_levels=10) | `scripts/train_hdc.py` | `hdc_pump_config.yaml` + `pump_by_id_config.yaml` |
| **exp_015_mahalanobis_pump_by_id** | Mahalanobis (refit) | `scripts/train_unsupervised.py` | `unsupervised_config.yaml` (mahalanobis block) + `pump_by_id_config.yaml` |

Structure de sortie pour chaque expérience :
```
experiments/exp_0XX_<model>_pump_by_id/
├── config_snapshot.yaml       # copie des configs fusionnées
├── checkpoints/               # ou results/ pour non supervisé
│   └── metrics.json           # AA, AF, BWT, ram_peak_bytes, latency_ms
└── acc_matrix.npy             # matrice 5×5
```

### 4. Nouveau plot `plot_performance_by_pump_id_bar()` dans `src/evaluation/plots.py`

Barplot groupé : axe X = Pump_ID (1–5), barres = accuracy finale par modèle après
entraînement complet sur les 5 tâches. Révèle quels Pump_ID sont les plus "difficiles"
à retenir.

```python
def plot_performance_by_pump_id_bar(
    results: dict[str, dict],   # {model_name: {pump_id: final_accuracy}}
    pump_ids: list[int],
    title: str = "Accuracy finale par Pump_ID",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Parameters
    ----------
    results : dict
        Clés = noms de modèles, valeurs = dict {pump_id (int): accuracy (float)}.
    pump_ids : list[int]
        Liste des Pump_IDs (ex. [1, 2, 3, 4, 5]).
    """
```

### 5. Plots réutilisés (pas de nouvelles fonctions nécessaires)

Appeler dans `notebooks/03_cl_evaluation.ipynb` (nouvelle section "Scénario Pump par ID") :

| Fonction | Module | Usage pour ce scénario |
|----------|--------|------------------------|
| `plot_accuracy_matrix(acc_matrix, task_names=["P1","P2","P3","P4","P5"])` | `plots.py` | Matrice 5×5 par Pump_ID pour chaque modèle |
| `plot_forgetting_curve(acc_matrix)` | `plots.py` | Courbe d'oubli 5 tâches (quel Pump_ID est oublié en premier ?) |
| `plot_model_radar(results, ...)` | `plots.py` | Comparaison 4 modèles sur ce scénario |
| `plot_boxplots_by_pump_id(df, ...)` | `eda_plots.py` | Distribution features par Pump_ID — déjà dans notebook section 2.4-ter |
| `plot_fault_rate_heatmap_pump(df, ...)` | `eda_plots.py` | Taux de panne Pump_ID × fenêtres temporelles — déjà dans notebook |
| `plot_cl_evolution(task_arrays, pca2d)` | `feature_space_plots.py` | Évolution de l'espace feature tâche par tâche (5 panels) |
| `plot_clustering_with_correctness(...)` | `feature_space_plots.py` | Correctness des prédictions dans l'espace PCA |

**Figures à sauvegarder dans** `notebooks/figures/cl_evaluation/pump_by_id/` :
- `pump_by_id_acc_matrix_{model}.png` (× 4 modèles)
- `pump_by_id_forgetting_curve.png` (4 modèles superposés)
- `pump_by_id_performance_bar.png` (nouveau plot S5-18.4)
- `pump_by_id_radar_comparison.png`
- `pump_by_id_cl_evolution_{model}.png` (× 4)

---

## Critères d'acceptation

- [ ] `get_pump_dataloaders_by_id()` retourne bien 5 dicts avec `pump_id` dans chaque dict
- [ ] `configs/pump_by_id_config.yaml` créé et lisible par les scripts existants
- [ ] exp_012 à exp_015 : 4 répertoires `experiments/` créés avec `metrics.json`
- [ ] `plot_performance_by_pump_id_bar()` ajouté dans `plots.py` avec docstring NumPy
- [ ] Cellules notebook ajoutées dans `notebooks/03_cl_evaluation.ipynb` (section "Pump par ID")
- [ ] Figures sauvegardées dans `notebooks/figures/cl_evaluation/pump_by_id/`
- [ ] RAM peak enregistrée pour chaque modèle (comparaison 3 vs 5 tâches)

---

## Artefacts produits

| Fichier | Chemin | Commitable | Utilisé par |
|---------|--------|:----------:|-------------|
| Loader additionnel | `src/data/pump_dataset.py` (fonction ajoutée) | ✅ | Scripts d'entraînement |
| Config YAML | `configs/pump_by_id_config.yaml` | ✅ | Scripts d'entraînement |
| Résultats exp_012–015 | `experiments/exp_012–015/` | ❌ (gitignore data) | Notebook |
| Nouveau plot | `src/evaluation/plots.py` (fonction ajoutée) | ✅ | Notebook |
| Notebook mis à jour | `notebooks/03_cl_evaluation.ipynb` | ✅ | Rapport |
| Figures PNG | `notebooks/figures/cl_evaluation/pump_by_id/*.png` | ❌ (gitignore figures) | Rapport |
| Sprint doc | `docs/sprints/sprint_5/S518_pump_granular_experiments.md` | ✅ | Ce fichier |

---

## Commandes de vérification

```bash
# Vérifier le nouveau loader
python -c "
from src.data.pump_dataset import get_pump_dataloaders_by_id
tasks = get_pump_dataloaders_by_id(
    csv_path='data/raw/pump_maintenance/Large Industrial_Pump_Maintenance_Dataset/Large_Industrial_Pump_Maintenance_Dataset.csv',
    normalizer_path='configs/pump_normalizer.yaml'
)
assert len(tasks) == 5, f'Expected 5 tasks, got {len(tasks)}'
for t in tasks:
    assert 'pump_id' in t and 'train_loader' in t
    print(f'Pump_ID={t[\"pump_id\"]} : {t[\"n_train\"]} train, {t[\"n_val\"]} val')
print('OK — loader Pump par ID validé')
"

# Lancer une expérience (exemple EWC)
python scripts/train_ewc.py \
  --config configs/ewc_pump_config.yaml \
  --data_config configs/pump_by_id_config.yaml \
  --exp_dir experiments/exp_013_ewc_pump_by_id

# Vérifier le nouveau plot
python -c "
from src.evaluation.plots import plot_performance_by_pump_id_bar
import matplotlib; matplotlib.use('Agg')
results = {'EWC': {1: 0.7, 2: 0.6, 3: 0.8, 4: 0.5, 5: 0.7}, 'HDC': {1: 0.6, 2: 0.5, 3: 0.7, 4: 0.4, 5: 0.6}}
fig = plot_performance_by_pump_id_bar(results, pump_ids=[1,2,3,4,5])
print('plot_performance_by_pump_id_bar OK')
"

# Tests existants toujours verts
pytest tests/ -v
```

---

## Questions ouvertes

- `TODO(arnaud)` : Le normaliser actuel (`pump_normalizer.yaml`) est ajusté sur Pump_ID=1 normales
  uniquement. Pour ce scénario, faut-il normaliser sur l'ensemble des Pump_IDs ou conserver la
  normalisation sur Pump_ID=1 comme référence de "tâche initiale" ?
- `TODO(fred)` : Les Pump_ID correspondent-ils à des pompes physiquement distinctes (âge, modèle,
  site) ou sont-ils juste des identifiants séquentiels ? La réponse change l'interprétation du
  domain shift inter-tâches.
- `FIXME(gap1)` : Valider que le scénario par Pump_ID modélise bien un domain shift industriel
  réaliste (vs scénario temporel) — à argumenter dans le manuscrit.
