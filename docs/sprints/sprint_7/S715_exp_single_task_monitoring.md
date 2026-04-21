# S7-15 — Infrastructure & Expériences Baseline Single-Task (Dataset 2 Monitoring)

> **Sprint 7 — Pré-requis obligatoire à S7-16**  
> Durée estimée : 3h  
> Statut : 🟢 Terminé — exp_030–035 exécutées, `metrics_single_task.json` présent pour chaque modèle

---

## Objectif

Mettre en place l'infrastructure nécessaire pour entraîner et évaluer les 6 modèles du projet sur le Dataset 2 (Equipment Monitoring) **sans découpage en tâches CL**.  
Toutes les données sont traitées comme une seule tâche unifiée avec un split train/test global.

Ces expériences servent de **baseline ultime** pour les notebooks de comparaison : elles répondent à la question *"Quelle est la performance maximale sans contrainte CL ?"*

---

## Distinction avec `train_joint()`

| Critère | `train_joint()` (existant) | Single-task baseline (nouveau) |
|---------|---------------------------|-------------------------------|
| Structure | Plusieurs tâches concaténées | **Pas de structure de tâches** |
| Évaluation | `acc_matrix[T-1, :]` par tâche | Accuracy/F1/AUC globaux |
| Métriques CL | AF, BWT = NaN | **Non applicables** |
| Usage | Upper-bound dans scénario CL | Référence hors-CL |

---

## Sous-tâches

### S7-15a — Config `configs/monitoring_single_task_config.yaml`

Nouveau fichier YAML avec les clés :

```yaml
# Baseline single-task — Dataset 2 Equipment Monitoring
# MEM: pas de tâches séquentielles, split train/test global uniquement

dataset:
  name: monitoring
  task_split: "no_split"          # pas de découpage par équipement ou location
  test_ratio: 0.2
  val_ratio: 0.1                  # sur le train uniquement, pour early stopping
  seed: 42
  features: ["temperature", "pressure", "vibration", "humidity"]
  label: "faulty"

paths:
  csv_path: "data/raw/equipment_monitoring/equipment_monitoring.csv"
  normalizer_path: "configs/monitoring_normalizer.yaml"
```

### S7-15b — Loader `get_monitoring_dataloaders_single_task()`

Ajouter dans `src/data/monitoring_dataset.py` :

```python
def get_monitoring_dataloaders_single_task(
    csv_path: Path,
    normalizer_path: Path,
    batch_size: int = 32,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Retourne un dict unique (pas une liste) avec toutes les données monitoring.
    Pas de découpage par domaine — baseline hors-CL.

    Returns
    -------
    dict avec clés : train_loader, val_loader, test_loader, n_train, n_val, n_test
    """
```

- Fusionne tous les équipements (Pump + Turbine + Compressor)
- Normalisation Z-score fittée sur le train split (stratifié sur `faulty`)
- Retourne **un seul dict** (signal que ce n'est pas du CL)

### S7-15c — Expériences exp_030 à exp_035

| Exp ID | Modèle | Script | Config principale |
|--------|--------|--------|-------------------|
| exp_030 | EWC | `train_ewc.py` | `ewc_config.yaml` |
| exp_031 | HDC | `train_hdc.py` | `hdc_config.yaml` |
| exp_032 | TinyOL | `train_tinyol.py` | `tinyol_config.yaml` |
| exp_033 | KMeans | `train_unsupervised.py` | `unsupervised_config.yaml` |
| exp_034 | Mahalanobis | `train_unsupervised.py` | `unsupervised_config.yaml` |
| exp_035 | DBSCAN | `train_unsupervised.py` | `unsupervised_config.yaml` |

Commandes d'exécution :

```bash
python scripts/train_ewc.py \
  --config configs/ewc_config.yaml \
  --data_config configs/monitoring_single_task_config.yaml \
  --exp_dir experiments/exp_030_ewc_monitoring_single_task \
  --skip-baselines

python scripts/train_hdc.py \
  --config configs/hdc_config.yaml \
  --data_config configs/monitoring_single_task_config.yaml \
  --exp_dir experiments/exp_031_hdc_monitoring_single_task \
  --skip-baselines

python scripts/train_tinyol.py \
  --config configs/tinyol_config.yaml \
  --data_config configs/monitoring_single_task_config.yaml \
  --exp_dir experiments/exp_032_tinyol_monitoring_single_task \
  --skip-baselines

python scripts/train_unsupervised.py \
  --config configs/unsupervised_config.yaml \
  --model kmeans \
  --data_config configs/monitoring_single_task_config.yaml \
  --exp_dir experiments/exp_033_kmeans_monitoring_single_task

python scripts/train_unsupervised.py \
  --config configs/unsupervised_config.yaml \
  --model mahalanobis \
  --data_config configs/monitoring_single_task_config.yaml \
  --exp_dir experiments/exp_034_mahalanobis_monitoring_single_task

python scripts/train_unsupervised.py \
  --config configs/unsupervised_config.yaml \
  --model dbscan \
  --data_config configs/monitoring_single_task_config.yaml \
  --exp_dir experiments/exp_035_dbscan_monitoring_single_task
```

---

## Sorties attendues par expérience

```
experiments/exp_030_ewc_monitoring_single_task/
├── config_snapshot.yaml
└── results/
    └── metrics_single_task.json   ← accuracy, f1, auc_roc, ram_peak_bytes, inference_latency_ms, n_params
```

> **Pas** d'`acc_matrix.npy` — il n'y a pas de tâches séquentielles.  
> `metrics_single_task.json` remplace `metrics.json` pour distinguer des scénarios CL.

---

## Critères de succès

- [ ] `configs/monitoring_single_task_config.yaml` créé et valide
- [ ] `get_monitoring_dataloaders_single_task()` passe les tests unitaires
- [ ] 6 expériences (exp_030–035) exécutées sans erreur
- [ ] Chaque `metrics_single_task.json` contient les 6 métriques obligatoires
- [ ] `pytest tests/` passe entièrement (pas de régression)

---

## Dépendances

- Pré-requis : aucun (première tâche du sprint pour les baselines)
- Bloqué par : aucun
- Débloque : **S7-16** (notebook baseline monitoring)
