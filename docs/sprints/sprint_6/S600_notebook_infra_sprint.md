# Sprint 6 (Phase 1) — Infrastructure notebooks & expériences granulaires complètes

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 6 — Phase 1 Extension (≥ 15 avril 2026) |
| **Priorité globale** | 🔴 Critique — bloque les Sprints 7, 8, 9 |
| **Durée estimée totale** | ~20h |
| **Dépendances** | Sprint 5 terminé (tous les modèles implémentés), S5-18/S5-19 (configs pump_by_id et monitoring_by_location existantes) |

> ⚠️ **Note numérotation** : Le dossier `sprint_6/` contenait déjà `S601_stm32_env_setup.md` (Phase 2 MCU). Ce sprint MCU est renommé Sprint 10 dans `roadmap_phase2.md`. Ce fichier S600 est le sprint Phase 1 Extension pour l'infrastructure notebooks.

---

## Objectif

Mettre en place l'infrastructure technique nécessaire pour les 28 notebooks d'évaluation (Sprints 7 et 8) : utilitaire de sauvegarde figures avec sous-dossiers, loader temporel pour Dataset 1, fonctions de visualisation manquantes, et lancement des 18 expériences manquantes.

**Critère de succès** : `save_figure()` supporte les chemins imbriqués, le loader `get_pump_dataloaders_by_temporal_window()` retourne 4 tâches, les 18 expériences (exp_012–029) sont exécutées et leurs `metrics.json` + `acc_matrix.npy` sont disponibles dans `experiments/`.

---

## Tâches

| ID | Tâche | Priorité | Fichier cible | Durée est. | Dépendances |
|----|-------|:---:|---------------|:---:|-------------|
| S6-01 | Vérifier et corriger `save_figure()` pour création auto de sous-dossiers (`parents=True, exist_ok=True`) | 🔴 | `src/evaluation/plots.py` | 1h | — |
| S6-02 | Implémenter `get_pump_dataloaders_by_temporal_window()` — 4 tâches par quartiles de 5 000 entrées triées par `Operational_Hours` | 🔴 | `src/data/pump_dataset.py` | 3h | — |
| S6-03 | Créer `configs/pump_by_temporal_window_config.yaml` (compatible TinyOL, EWC, HDC, unsupervised) | 🔴 | `configs/pump_by_temporal_window_config.yaml` | 1h | S6-02 |
| S6-04 | Implémenter `plot_performance_by_pump_id_bar()` dans `plots.py` | 🔴 | `src/evaluation/plots.py` | 2h | — |
| S6-05 | Implémenter `plot_performance_heatmap_equipment_location()` dans `plots.py` | 🔴 | `src/evaluation/plots.py` | 2h | — |
| S6-06 | Run exp_012–015 : pump_by_id pour TinyOL, EWC, HDC, Mahalanobis | 🔴 | `experiments/exp_012–015/` | 2h | `pump_by_id_config.yaml` (existant) |
| S6-07 | Run exp_016–019 : monitoring_by_location pour EWC, HDC, TinyOL, Mahalanobis | 🔴 | `experiments/exp_016–019/` | 2h | `monitoring_by_location_config.yaml` (existant) |
| S6-08 | Run exp_020–021 : pump_by_id pour KMeans et DBSCAN | 🟡 | `experiments/exp_020–021/` | 1h | S6-06 |
| S6-09 | Run exp_022–023 : monitoring_by_location pour KMeans et DBSCAN | 🟡 | `experiments/exp_022–023/` | 1h | S6-07 |
| S6-10 | Run exp_024–029 : pump_by_temporal_window pour les 6 modèles (TinyOL, EWC, HDC, Mahalanobis, KMeans, DBSCAN) | 🔴 | `experiments/exp_024–029/` | 3h | S6-02, S6-03 |
| S6-11 | Tests unitaires du loader `get_pump_dataloaders_by_temporal_window()` | 🟡 | `tests/test_pump_dataset.py` | 2h | S6-02 |

---

## Détail des tâches critiques

### S6-01 — Refactoring `save_figure()`

Vérifier que dans `src/evaluation/plots.py`, la fonction `save_figure()` (ou équivalent) appelle bien `Path(output_path).parent.mkdir(parents=True, exist_ok=True)` avant `plt.savefig()`. Si ce n'est pas le cas, l'ajouter.

Pattern cible pour les notebooks :
```python
FIGURES_DIR = Path("../notebooks/figures/cl_evaluation")
fig_path = FIGURES_DIR / "ewc" / "monitoring" / "by_equipment" / "acc_matrix.png"
save_figure(fig, fig_path)  # doit créer les sous-dossiers automatiquement
```

### S6-02 — Loader `get_pump_dataloaders_by_temporal_window()`

```python
def get_pump_dataloaders_by_temporal_window(
    csv_path: str | Path,
    normalizer_path: str | Path,
    n_tasks: int = 4,
    entries_per_task: int = 5000,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> list[dict]:
    """
    Scénario CL domain-incremental par fenêtres temporelles.

    Découpe les 20 000 entrées en n_tasks quartiles de entries_per_task entrées
    chacune, triées par Operational_Hours.
      T1 : lignes 0–4999 (Operational_Hours les plus basses)
      T2 : lignes 5000–9999
      T3 : lignes 10000–14999
      T4 : lignes 15000–19999

    Applique le même feature engineering que get_pump_dataloaders() :
    fenêtrage WINDOW_SIZE=32, STEP_SIZE=16, 6 stats × 4 canaux + temporal_position.

    Normalisation Z-score ajustée sur T1 uniquement (pump_normalizer.yaml).
    """
```

Points clés :
- Trier par `Operational_Hours` avant le split (pas d'ordre garanti dans le CSV)
- Appliquer le feature engineering (fenêtrage + stats) sur chaque tranche séparément
- Normaliser avec le même `pump_normalizer.yaml` ajusté sur T1

### S6-03 — Config `pump_by_temporal_window_config.yaml`

```yaml
# pump_by_temporal_window_config.yaml — Scénario CL par fenêtres temporelles (4 tâches)
# T1: 0–4999 entrées (Operational_Hours les + basses)
# T2: 5000–9999 | T3: 10000–14999 | T4: 15000–19999

data:
  csv_path: "data/raw/pump_maintenance/Large_Industrial_Pump_Maintenance_Dataset/Large_Industrial_Pump_Maintenance_Dataset.csv"
  normalizer_path: "configs/pump_normalizer.yaml"
  task_split: by_temporal_window
  n_tasks: 4
  entries_per_task: 5000
  temporal_col: Operational_Hours
  label_col: Maintenance_Flag
  feature_cols: [temperature, vibration, pressure, rpm]
  window_size: 32
  step_size: 16
  val_ratio: 0.2
  dataset: pump_maintenance
  label_column: maintenance_required
  feature_columns: [temperature, vibration, pressure, rpm]
```

### S6-04 — `plot_performance_by_pump_id_bar()`

Barplot groupé avec :
- Axe X : Pump_ID (1, 2, 3, 4, 5)
- Barres colorées par modèle (TinyOL, EWC, HDC, Mahalanobis, KMeans, DBSCAN)
- Valeur : accuracy finale sur la tâche correspondant à ce Pump_ID (dernière ligne de acc_matrix)

```python
def plot_performance_by_pump_id_bar(
    results: dict[str, dict[int, float]],
    pump_ids: list[int],
    title: str = "Accuracy finale par Pump_ID",
    figsize: tuple[int, int] = (10, 5),
    output_path: str | Path | None = None,
) -> plt.Figure:
```

### S6-05 — `plot_performance_heatmap_equipment_location()`

Heatmap 2D (equipment type × location) avec la performance moyenne par cellule. Utile pour détecter des patterns géographiques ou par type d'équipement sur Dataset 2.

```python
def plot_performance_heatmap_equipment_location(
    results: dict[str, dict[tuple[str, str], float]],
    # {model: {(equipment, location): accuracy}}
    equipment_types: list[str],
    locations: list[str],
    model_name: str,
    title: str = "Performance par Équipement × Location",
    output_path: str | Path | None = None,
) -> plt.Figure:
```

### S6-06 à S6-09 — Expériences manquantes

Utiliser les scripts existants avec les configs existantes :

```bash
# Scénario pump_by_id (modèles supervisés — exp_012–014)
python scripts/train_tinyol.py --config configs/tinyol_config.yaml --data_config configs/pump_by_id_config.yaml --exp_dir experiments/exp_012_tinyol_pump_by_id
python scripts/train_ewc.py --config configs/ewc_pump_config.yaml --data_config configs/pump_by_id_config.yaml --exp_dir experiments/exp_013_ewc_pump_by_id
python scripts/train_hdc.py --config configs/hdc_pump_config.yaml --data_config configs/pump_by_id_config.yaml --exp_dir experiments/exp_014_hdc_pump_by_id

# Scénario pump_by_id (modèles non supervisés — exp_015, 020, 021)
python scripts/train_unsupervised.py --config configs/unsupervised_config.yaml --data_config configs/pump_by_id_config.yaml --model mahalanobis --exp_dir experiments/exp_015_mahalanobis_pump_by_id
python scripts/train_unsupervised.py --config configs/unsupervised_config.yaml --data_config configs/pump_by_id_config.yaml --model kmeans --exp_dir experiments/exp_020_kmeans_pump_by_id
python scripts/train_unsupervised.py --config configs/unsupervised_config.yaml --data_config configs/pump_by_id_config.yaml --model dbscan --exp_dir experiments/exp_021_dbscan_pump_by_id

# Scénario monitoring_by_location (modèles supervisés — exp_016–018)
python scripts/train_ewc.py --config configs/ewc_config.yaml --data_config configs/monitoring_by_location_config.yaml --exp_dir experiments/exp_016_ewc_monitoring_by_location
python scripts/train_hdc.py --config configs/hdc_config.yaml --data_config configs/monitoring_by_location_config.yaml --exp_dir experiments/exp_017_hdc_monitoring_by_location
python scripts/train_tinyol.py --config configs/tinyol_monitoring_config.yaml --data_config configs/monitoring_by_location_config.yaml --exp_dir experiments/exp_018_tinyol_monitoring_by_location

# Scénario monitoring_by_location (modèles non supervisés — exp_019, 022, 023)
python scripts/train_unsupervised.py --config configs/unsupervised_config.yaml --data_config configs/monitoring_by_location_config.yaml --model mahalanobis --exp_dir experiments/exp_019_mahalanobis_monitoring_by_location
python scripts/train_unsupervised.py --config configs/unsupervised_config.yaml --data_config configs/monitoring_by_location_config.yaml --model kmeans --exp_dir experiments/exp_022_kmeans_monitoring_by_location
python scripts/train_unsupervised.py --config configs/unsupervised_config.yaml --data_config configs/monitoring_by_location_config.yaml --model dbscan --exp_dir experiments/exp_023_dbscan_monitoring_by_location

# Scénario pump_by_temporal_window (tous modèles — exp_024–029)
python scripts/train_tinyol.py --config configs/tinyol_config.yaml --data_config configs/pump_by_temporal_window_config.yaml --exp_dir experiments/exp_024_tinyol_pump_temporal
python scripts/train_ewc.py --config configs/ewc_pump_config.yaml --data_config configs/pump_by_temporal_window_config.yaml --exp_dir experiments/exp_025_ewc_pump_temporal
python scripts/train_hdc.py --config configs/hdc_pump_config.yaml --data_config configs/pump_by_temporal_window_config.yaml --exp_dir experiments/exp_026_hdc_pump_temporal
python scripts/train_unsupervised.py --model mahalanobis --data_config configs/pump_by_temporal_window_config.yaml --exp_dir experiments/exp_027_mahalanobis_pump_temporal
python scripts/train_unsupervised.py --model kmeans --data_config configs/pump_by_temporal_window_config.yaml --exp_dir experiments/exp_028_kmeans_pump_temporal
python scripts/train_unsupervised.py --model dbscan --data_config configs/pump_by_temporal_window_config.yaml --exp_dir experiments/exp_029_dbscan_pump_temporal
```

---

## Table des expériences planifiées

| Exp | Modèle | Scénario | Dataset | Statut |
|-----|--------|----------|---------|--------|
| exp_012 | TinyOL | pump_by_pump_id (5 tâches P1→P5) | Dataset 1 | ⬜ |
| exp_013 | EWC | pump_by_pump_id (5 tâches P1→P5) | Dataset 1 | ⬜ |
| exp_014 | HDC | pump_by_pump_id (5 tâches P1→P5) | Dataset 1 | ⬜ |
| exp_015 | Mahalanobis | pump_by_pump_id (5 tâches P1→P5) | Dataset 1 | ⬜ |
| exp_016 | EWC | monitoring_by_location (5 tâches ATL→SFO) | Dataset 2 | ⬜ |
| exp_017 | HDC | monitoring_by_location (5 tâches ATL→SFO) | Dataset 2 | ⬜ |
| exp_018 | TinyOL | monitoring_by_location (5 tâches ATL→SFO) | Dataset 2 | ⬜ |
| exp_019 | Mahalanobis | monitoring_by_location (5 tâches ATL→SFO) | Dataset 2 | ⬜ |
| exp_020 | KMeans | pump_by_pump_id (5 tâches P1→P5) | Dataset 1 | ⬜ |
| exp_021 | DBSCAN | pump_by_pump_id (5 tâches P1→P5) | Dataset 1 | ⬜ |
| exp_022 | KMeans | monitoring_by_location (5 tâches ATL→SFO) | Dataset 2 | ⬜ |
| exp_023 | DBSCAN | monitoring_by_location (5 tâches ATL→SFO) | Dataset 2 | ⬜ |
| exp_024 | TinyOL | pump_by_temporal_window (4 tâches Q1→Q4) | Dataset 1 | ⬜ |
| exp_025 | EWC | pump_by_temporal_window (4 tâches Q1→Q4) | Dataset 1 | ⬜ |
| exp_026 | HDC | pump_by_temporal_window (4 tâches Q1→Q4) | Dataset 1 | ⬜ |
| exp_027 | Mahalanobis | pump_by_temporal_window (4 tâches Q1→Q4) | Dataset 1 | ⬜ |
| exp_028 | KMeans | pump_by_temporal_window (4 tâches Q1→Q4) | Dataset 1 | ⬜ |
| exp_029 | DBSCAN | pump_by_temporal_window (4 tâches Q1→Q4) | Dataset 1 | ⬜ |

---

## Critères d'acceptation

- [ ] `save_figure(fig, nested/path/figure.png)` crée les sous-dossiers automatiquement
- [ ] `get_pump_dataloaders_by_temporal_window()` retourne exactement 4 dicts avec clé `temporal_window`
- [ ] `configs/pump_by_temporal_window_config.yaml` lu sans erreur par `train_ewc.py`, `train_hdc.py`, `train_unsupervised.py`
- [ ] `plot_performance_by_pump_id_bar()` + `plot_performance_heatmap_equipment_location()` importables depuis `src.evaluation.plots`
- [ ] 18 répertoires `experiments/exp_012–029/` créés, chacun avec `metrics.json` + `acc_matrix.npy`
- [ ] `pytest tests/ -v` passe (notamment `test_pump_dataset.py`)

---

## Commandes de vérification

```bash
# Vérifier le nouveau loader
python -c "
from src.data.pump_dataset import get_pump_dataloaders_by_temporal_window
tasks = get_pump_dataloaders_by_temporal_window(
    csv_path='data/raw/pump_maintenance/Large_Industrial_Pump_Maintenance_Dataset/Large_Industrial_Pump_Maintenance_Dataset.csv',
    normalizer_path='configs/pump_normalizer.yaml'
)
assert len(tasks) == 4, f'Expected 4 tasks, got {len(tasks)}'
for t in tasks: print(f'Window {t[\"temporal_window\"]}: {t[\"n_train\"]} train, {t[\"n_val\"]} val')
print('OK')
"

# Vérifier les nouvelles fonctions de visualisation
python -c "
from src.evaluation.plots import plot_performance_by_pump_id_bar, plot_performance_heatmap_equipment_location
print('Imports OK')
"

# Tests
pytest tests/ -v
```

---

## Livrable sprint 6

Infrastructure complète pour les Sprints 7 et 8 :
- `save_figure()` supporte les chemins imbriqués
- Loader temporel opérationnel
- 2 nouvelles fonctions de visualisation
- 18 expériences exp_012–029 exécutées

---

## Questions ouvertes

- `TODO(arnaud)` : Pour le scénario pump_by_temporal_window, faut-il trier les 20 000 entrées globalement par `Operational_Hours` avant découpage, ou respecter l'ordre par Pump_ID au sein de chaque quartile ? L'interprétation scientifique diffère.
- `TODO(fred)` : Les quartiles temporels (0–5000, 5001–10000, etc.) correspondent-ils à des phases d'exploitation réelles (rodage, fonctionnement nominal, usure avancée) ? Cette information changerait le nommage des tâches dans les notebooks.
- `FIXME(gap1)` : Les AA très basses sur Dataset 1 (~0.50) persistent-elles sur le scénario temporel à 4 tâches ? Si oui, le dataset est peut-être trop homogène pour distinguer les phases d'usure.

---

> **⚠️ Après l'implémentation de ce sprint** : exécuter TOUTES les commandes de la section "Commandes de vérification" et vérifier que les 18 répertoires d'expériences sont bien créés avec leurs `metrics.json`. Mettre à jour `docs/roadmap_phase1.md` en marquant les tâches S6-01 à S6-11 comme ✅ et enregistrer les métriques AA/AF/BWT/RAM de chaque expérience dans la section "Résultats d'expériences".
