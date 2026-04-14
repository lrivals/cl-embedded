# S5-19 — Expériences granulaires Dataset 2 : scénario Monitoring par location

| Champ | Valeur |
|-------|--------|
| **ID** | S5-19 |
| **Sprint** | Sprint 5 — Extension (≥ 12 avril 2026) |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 6h |
| **Dépendances** | S1-03 (`monitoring_dataset.py` existant), S5-10 (Mahalanobis existant), S5-15 (EDA equipment × location existante) |
| **Fichiers cibles** | `src/data/monitoring_dataset.py`, `configs/monitoring_by_location_config.yaml`, `experiments/exp_016–019/`, `src/evaluation/plots.py`, `notebooks/03_cl_evaluation.ipynb` |
| **Complété le** | — |
| **Statut** | ⬜ À faire |

---

## Objectif

Implémenter un nouveau scénario CL **domain-incremental par location géographique** sur le
Dataset 2 (Equipment Monitoring). Au lieu des 3 tâches par type d'équipement (Pump → Turbine →
Compressor), chaque tâche correspond à un site géographique distinct :
Atlanta → Chicago → Houston → New York → San Francisco (5 tâches).

Chaque tâche contient les données des **3 types d'équipements** pour ce site. La question
devient : *le modèle apprend-il à détecter des anomalies dans un contexte géographique
(conditions ambiantes, historique de maintenance) plutôt que par type de machine ?*

**Questions scientifiques ciblées** :
- Les sites géographiques induisent-ils un domain shift plus ou moins prononcé que les types
  d'équipements ? (Hyp. : conditions climatiques et opérationnelles différentes par ville)
- Certains sites ont-ils des taux de panne systématiquement plus élevés (cf. `fault_rate_heatmap`)
  et sont-ils plus difficiles à apprendre ?
- Un modèle entraîné sur Pump+Turbine+Compressor à Atlanta généralise-t-il à Chicago, ou
  l'effet site domine-t-il l'effet équipement ?
- Comparaison directe scénario equipment (3 tâches) vs scénario location (5 tâches) : quel
  découpage produit le plus d'oubli ?

**Critère de succès** : 4 expériences (exp_016 à exp_019) exécutées, métriques AA/AF/BWT/RAM
enregistrées, plots générés dans `notebooks/figures/cl_evaluation/monitoring_by_location/`.

---

## Sous-tâches

### 1. Nouveau loader `get_cl_dataloaders_by_location()` dans `src/data/monitoring_dataset.py`

Ajouter une fonction retournant 5 dicts (un par site), avec la même interface que
`get_cl_dataloaders()` :

```python
def get_cl_dataloaders_by_location(
    csv_path: str,
    normalizer_path: str,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    seed: int = 42,
    location_order: list[str] | None = None,
) -> list[dict]:
    """
    Crée un scénario CL domain-incremental où chaque tâche correspond à une location distincte.
    Chaque tâche contient les 3 types d'équipements pour ce site.

    Parameters
    ----------
    csv_path : str
        Chemin complet vers le CSV equipment monitoring.
    normalizer_path : str
        Chemin vers monitoring_normalizer.yaml (ajusté sur Pump domain).
    batch_size : int
        Taille de batch.
    val_ratio : float
        Fraction validation (stratifié sur faulty).
    seed : int
        Seed reproductibilité.
    location_order : list[str] | None
        Ordre des locations (défaut: alphabétique = Atlanta, Chicago, Houston,
        New York, San Francisco).

    Returns
    -------
    list[dict]
        Liste de 5 dicts, chacun avec keys : task_id, location, train_loader,
        val_loader, n_train, n_val.
    """
```

**Points d'implémentation** :
- Ordre par défaut : alphabétique (`["Atlanta", "Chicago", "Houston", "New York", "San Francisco"]`)
- Même features numériques que l'existant : `[temperature, pressure, vibration, humidity]`
- Normalisation avec `monitoring_normalizer.yaml` (ajusté sur Pump domain task 0 — garder
  comme référence même si la tâche 0 est maintenant Atlanta)
- Split train/val stratifié par `faulty` au sein de chaque location

### 2. Config YAML `configs/monitoring_by_location_config.yaml`

```yaml
# monitoring_by_location_config.yaml — Scénario CL par location géographique (5 tâches)
data:
  csv_path: /home/leonard/Documents/ENAC/cl-embedded/data/raw/equipment_monitoring/Industrial_Equipment_Monitoring_Dataset/equipment_anomaly_data.csv
  normalizer_path: configs/monitoring_normalizer.yaml
  task_split: by_location          # nouveau mode
  location_col: location
  equipment_col: equipment         # conservé pour analyse croisée
  label_col: faulty
  feature_cols: [temperature, pressure, vibration, humidity]
  location_order:
    - Atlanta
    - Chicago
    - Houston
    - New York
    - San Francisco
  val_ratio: 0.2

# Sections model/training : reprendre ewc_config.yaml / hdc_config.yaml /
# tinyol_monitoring_config.yaml / unsupervised_config.yaml selon le modèle.
```

### 3. Expériences exp_016 à exp_019

| Exp | Modèle | Script | Config supplémentaire |
|-----|--------|--------|----------------------|
| **exp_016_ewc_monitoring_by_location** | EWC Online + MLP | `scripts/train_ewc.py` | `ewc_config.yaml` + `monitoring_by_location_config.yaml` |
| **exp_017_hdc_monitoring_by_location** | HDC (D=1024) | `scripts/train_hdc.py` | `hdc_config.yaml` + `monitoring_by_location_config.yaml` |
| **exp_018_tinyol_monitoring_by_location** | TinyOL (backbone pré-entraîné sur Pump normales) | `scripts/train_tinyol.py` | `tinyol_monitoring_config.yaml` + `monitoring_by_location_config.yaml` |
| **exp_019_mahalanobis_monitoring_by_location** | Mahalanobis (refit) | `scripts/train_unsupervised.py` | `unsupervised_config.yaml` (mahalanobis block) + `monitoring_by_location_config.yaml` |

Structure de sortie pour chaque expérience :
```
experiments/exp_0XX_<model>_monitoring_by_location/
├── config_snapshot.yaml       # copie des configs fusionnées
├── checkpoints/               # ou results/ pour non supervisé
│   └── metrics.json           # AA, AF, BWT, ram_peak_bytes, latency_ms
└── acc_matrix.npy             # matrice 5×5
```

### 4. Nouveau plot `plot_performance_heatmap_equipment_location()` dans `src/evaluation/plots.py`

Heatmap `3 equipment × 5 locations` où chaque cellule contient l'accuracy finale d'un
modèle sur ce sous-groupe. Révèle si certains croisements sont systématiquement difficiles
à détecter.

```python
def plot_performance_heatmap_equipment_location(
    results_by_cell: dict[str, dict[tuple[str, str], float]],
    equipment_types: list[str],
    locations: list[str],
    title: str = "Accuracy par équipement × location",
    figsize: tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Parameters
    ----------
    results_by_cell : dict
        Clés = noms de modèles, valeurs = dict {(equipment, location): accuracy (float)}.
    equipment_types : list[str]
        Ex. ["Pump", "Turbine", "Compressor"].
    locations : list[str]
        Ex. ["Atlanta", "Chicago", "Houston", "New York", "San Francisco"].

    Returns
    -------
    plt.Figure
        Figure avec une heatmap par modèle (grille 2×2 pour 4 modèles).
    """
```

### 5. Plots réutilisés (pas de nouvelles fonctions nécessaires)

Appeler dans `notebooks/03_cl_evaluation.ipynb` (nouvelle section "Scénario Monitoring par location") :

| Fonction | Module | Usage pour ce scénario |
|----------|--------|------------------------|
| `plot_accuracy_matrix(acc_matrix, task_names=["ATL","CHI","HOU","NYC","SFO"])` | `plots.py` | Matrice 5×5 par location pour chaque modèle |
| `plot_forgetting_curve(acc_matrix)` | `plots.py` | Courbe d'oubli 5 tâches (quelle location est oubliée ?) |
| `plot_model_radar(results, ...)` | `plots.py` | Comparaison 4 modèles sur ce scénario |
| `plot_fault_rate_heatmap_equipment(df, ...)` | `eda_plots.py` | Taux de panne equipment × location — déjà dans notebook section 7-bis |
| `plot_boxplots_by_equipment_location(df, ...)` | `eda_plots.py` | Distribution features par location — déjà dans notebook |
| `plot_correlation_by_equipment(df, ...)` | `eda_plots.py` | Corrélations features par type d'équipement (contexte inter-location) |
| `plot_feature_space_2d(X_proj, y, domain_ids=location_ids)` | `feature_space_plots.py` | Scatter PCA coloré par location (visualise le domain shift) |
| `plot_cl_evolution(task_arrays, pca2d)` | `feature_space_plots.py` | Évolution de l'espace feature tâche par tâche (5 panels) |

**Figures à sauvegarder dans** `notebooks/figures/cl_evaluation/monitoring_by_location/` :
- `by_location_acc_matrix_{model}.png` (× 4 modèles)
- `by_location_forgetting_curve.png` (4 modèles superposés)
- `by_location_performance_heatmap_{model}.png` (× 4) — nouveau plot S5-19.4
- `by_location_radar_comparison.png`
- `by_location_feature_space_pca.png` (PCA coloré par location, avant tout entraînement)
- `by_location_cl_evolution_{model}.png` (× 4)

**Figure comparative clé** — à générer en fin de section :
- `comparison_equipment_vs_location.png` : barplot côte à côte AA/AF pour chaque modèle,
  scénario équipement (3 tâches) vs scénario location (5 tâches). Utilise `plot_metrics_comparison()`.

---

## Critères d'acceptation

- [ ] `get_cl_dataloaders_by_location()` retourne bien 5 dicts avec `location` dans chaque dict
- [ ] `configs/monitoring_by_location_config.yaml` créé et lisible par les scripts existants
- [ ] exp_016 à exp_019 : 4 répertoires `experiments/` créés avec `metrics.json`
- [ ] `plot_performance_heatmap_equipment_location()` ajouté dans `plots.py` avec docstring NumPy
- [ ] Cellules notebook ajoutées dans `notebooks/03_cl_evaluation.ipynb` (section "Monitoring par location")
- [ ] Figure comparative `comparison_equipment_vs_location.png` générée
- [ ] Figures sauvegardées dans `notebooks/figures/cl_evaluation/monitoring_by_location/`
- [ ] RAM peak enregistrée pour chaque modèle (comparaison 3 vs 5 tâches sur Dataset 2)

---

## Artefacts produits

| Fichier | Chemin | Commitable | Utilisé par |
|---------|--------|:----------:|-------------|
| Loader additionnel | `src/data/monitoring_dataset.py` (fonction ajoutée) | ✅ | Scripts d'entraînement |
| Config YAML | `configs/monitoring_by_location_config.yaml` | ✅ | Scripts d'entraînement |
| Résultats exp_016–019 | `experiments/exp_016–019/` | ❌ (gitignore data) | Notebook |
| Nouveau plot | `src/evaluation/plots.py` (fonction ajoutée) | ✅ | Notebook |
| Notebook mis à jour | `notebooks/03_cl_evaluation.ipynb` | ✅ | Rapport |
| Figures PNG | `notebooks/figures/cl_evaluation/monitoring_by_location/*.png` | ❌ (gitignore figures) | Rapport |
| Sprint doc | `docs/sprints/sprint_5/S519_monitoring_granular_experiments.md` | ✅ | Ce fichier |

---

## Commandes de vérification

```bash
# Vérifier le nouveau loader
python -c "
from src.data.monitoring_dataset import get_cl_dataloaders_by_location
tasks = get_cl_dataloaders_by_location(
    csv_path='data/raw/equipment_monitoring/Industrial_Equipment_Monitoring_Dataset/equipment_anomaly_data.csv',
    normalizer_path='configs/monitoring_normalizer.yaml'
)
assert len(tasks) == 5, f'Expected 5 tasks, got {len(tasks)}'
for t in tasks:
    assert 'location' in t and 'train_loader' in t
    print(f'Location={t[\"location\"]} : {t[\"n_train\"]} train, {t[\"n_val\"]} val')
print('OK — loader Monitoring par location validé')
"

# Lancer une expérience (exemple EWC)
python scripts/train_ewc.py \
  --config configs/ewc_config.yaml \
  --data_config configs/monitoring_by_location_config.yaml \
  --exp_dir experiments/exp_016_ewc_monitoring_by_location

# Vérifier le nouveau plot
python -c "
from src.evaluation.plots import plot_performance_heatmap_equipment_location
import matplotlib; matplotlib.use('Agg')
results = {
    'EWC': {('Pump','Atlanta'): 0.95, ('Turbine','Chicago'): 0.88, ('Compressor','Houston'): 0.91}
}
fig = plot_performance_heatmap_equipment_location(
    results,
    equipment_types=['Pump','Turbine','Compressor'],
    locations=['Atlanta','Chicago','Houston','New York','San Francisco']
)
print('plot_performance_heatmap_equipment_location OK')
"

# Tests existants toujours verts
pytest tests/ -v
```

---

## Questions ouvertes

- `TODO(arnaud)` : L'ordre des locations (alphabétique) est-il pertinent pour le scénario CL,
  ou faut-il un ordre basé sur la similarité des distributions (cluster hiérarchique) pour
  maximiser le domain shift progressif ?
- `TODO(fred)` : Les 5 locations (Atlanta, Chicago, Houston, New York, San Francisco) sont-elles
  des sites industriels réels avec des conditions climatiques et d'exploitation distinctes, ou
  des labels artificiels du dataset Kaggle ? La réponse impacte la justification scientifique
  du scénario dans le manuscrit.
- `TODO(arnaud)` : Faut-il ajouter un scénario combiné `equipment × location` (15 tâches =
  3 × 5) pour pousser la granularité maximale ? Ou est-ce trop fragmenté (≈ 500 échantillons
  par tâche) ?
- `FIXME(gap1)` : Les locations Kaggle ne sont pas validées industriellement — à distinguer
  clairement du gap 1 (validation sur FEMTO PRONOSTIA) dans le manuscrit.
