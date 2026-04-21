# S8-14 — Infrastructure & Expériences Baseline Single-Task (Dataset 1 Pump)

> **Sprint 8 — Pré-requis obligatoire à S8-16**  
> Durée estimée : 3h  
> Statut : ✅ Terminé

---

## Objectif

Mettre en place l'infrastructure nécessaire pour entraîner et évaluer les 6 modèles du projet sur le Dataset 1 (Large Industrial Pump Maintenance) **sans découpage en tâches CL**.  
Toutes les données sont traitées comme une seule tâche unifiée avec un split train/test global.

Ces expériences servent de **baseline ultime** pour les notebooks pump : elles répondent à la question *"Quelle est la performance maximale sur le dataset pump sans contrainte CL ?"*

---

## Distinction avec les scénarios CL existants

| Critère | Scénarios CL (existant) | Single-task baseline (nouveau) |
|---------|------------------------|-------------------------------|
| Découpage | 3T chrono / 5T by_id / 4T temporal | **Pas de découpage** |
| Features | 25 stats (sliding window) | **Mêmes 25 features** |
| Évaluation | acc_matrix + AF/BWT | Accuracy/F1/AUC globaux |
| Usage | Scénarios CL | Référence hors-CL |

> **Important** : Le Dataset 1 affiche AA ≈ 0.50 dans tous les scénarios CL (backbone trop faible, faible drift inter-tâches). La baseline single-task révèle si ce problème est structurel (données) ou lié à la contrainte CL.

---

## Sous-tâches

### S8-14a — Config `configs/pump_single_task_config.yaml`

Nouveau fichier YAML avec les clés :

```yaml
# Baseline single-task — Dataset 1 Pump Maintenance
# MEM: pas de tâches séquentielles, split train/test global uniquement

dataset:
  name: pump
  task_split: "no_split"          # pas de découpage temporel ou par pump_id
  test_ratio: 0.2
  val_ratio: 0.1                  # sur le train uniquement
  seed: 42
  window_size: 32                 # identique aux scénarios CL
  step_size: 16

paths:
  csv_path: "data/raw/pump_maintenance/pump_maintenance.csv"
  normalizer_path: "configs/pump_normalizer.yaml"
```

### S8-14b — Loader `get_pump_dataloaders_single_task()`

Ajouter dans `src/data/pump_dataset.py` :

```python
def get_pump_dataloaders_single_task(
    csv_path: Path,
    normalizer_path: Path,
    batch_size: int = 32,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 42,
    window_size: int = 32,
    step_size: int = 16,
) -> dict:
    """
    Retourne un dict unique (pas une liste) avec toutes les données pump.
    Pas de découpage temporel ou par pump_id — baseline hors-CL.

    Returns
    -------
    dict avec clés : train_loader, val_loader, test_loader, n_train, n_val, n_test
    """
```

- Fusionne tous les pumps (Pump_ID 1–5) et toutes les périodes opérationnelles
- Extraction des 25 features statistiques via sliding window (identique aux autres loaders)
- Normalisation Z-score fittée sur le train split (stratifié sur `maintenance_required`)
- Retourne **un seul dict** (signal que ce n'est pas du CL)

### S8-14c — Expériences exp_036 à exp_041

| Exp ID | Modèle | Script | Config principale |
|--------|--------|--------|-------------------|
| exp_036 | EWC | `train_ewc.py` | `ewc_config.yaml` |
| exp_037 | HDC | `train_hdc.py` | `hdc_config.yaml` |
| exp_038 | TinyOL | `train_tinyol.py` | `tinyol_config.yaml` |
| exp_039 | KMeans | `train_unsupervised.py` | `unsupervised_config.yaml` |
| exp_040 | Mahalanobis | `train_unsupervised.py` | `unsupervised_config.yaml` |
| exp_041 | DBSCAN | `train_unsupervised.py` | `unsupervised_config.yaml` |

Commandes d'exécution :

```bash
python scripts/train_ewc.py \
  --config configs/ewc_config.yaml \
  --data_config configs/pump_single_task_config.yaml \
  --exp_dir experiments/exp_036_ewc_pump_single_task \
  --skip-baselines

python scripts/train_hdc.py \
  --config configs/hdc_config.yaml \
  --data_config configs/pump_single_task_config.yaml \
  --exp_dir experiments/exp_037_hdc_pump_single_task \
  --skip-baselines

python scripts/train_tinyol.py \
  --config configs/tinyol_config.yaml \
  --data_config configs/pump_single_task_config.yaml \
  --exp_dir experiments/exp_038_tinyol_pump_single_task \
  --skip-baselines

python scripts/train_unsupervised.py \
  --config configs/unsupervised_config.yaml \
  --model kmeans \
  --data_config configs/pump_single_task_config.yaml \
  --exp_dir experiments/exp_039_kmeans_pump_single_task

python scripts/train_unsupervised.py \
  --config configs/unsupervised_config.yaml \
  --model mahalanobis \
  --data_config configs/pump_single_task_config.yaml \
  --exp_dir experiments/exp_040_mahalanobis_pump_single_task

python scripts/train_unsupervised.py \
  --config configs/unsupervised_config.yaml \
  --model dbscan \
  --data_config configs/pump_single_task_config.yaml \
  --exp_dir experiments/exp_041_dbscan_pump_single_task
```

---

## Sorties attendues par expérience

```
experiments/exp_036_ewc_pump_single_task/
├── config_snapshot.yaml
└── results/
    └── metrics_single_task.json   ← accuracy, f1, auc_roc, ram_peak_bytes, inference_latency_ms, n_params
```

> **Pas** d'`acc_matrix.npy` — il n'y a pas de tâches séquentielles.

---

## Hypothèse scientifique à vérifier

Si l'accuracy single-task est également ≈ 0.50, cela confirme que le problème est **structurel** (les features ne discriminent pas bien `maintenance_required` sur ce dataset), et non lié à la contrainte CL.  
→ FIXME(gap1) : envisager features alternatives ou dataset FEMTO-Bearing (données industrielles certifiées).

---

## Critères de succès

- [x] `configs/pump_single_task_config.yaml` créé et valide
- [x] `get_pump_dataloaders_single_task()` implémentée (`src/data/pump_dataset.py:845`) — ⚠️ tests unitaires à compléter dans `tests/test_pump_dataset.py`
- [x] 6 expériences (exp_036–041) exécutées sans erreur
- [x] Chaque `metrics_single_task.json` contient les 6 métriques obligatoires
- [x] `pytest tests/` passe entièrement (pas de régression)

---

## Dépendances

- Pré-requis : aucun (première tâche du sprint pour les baselines)
- Bloqué par : aucun
- Débloque : **S8-15** (notebook baseline pump)
