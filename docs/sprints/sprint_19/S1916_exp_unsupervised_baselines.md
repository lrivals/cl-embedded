# S1916 — Expériences M4 baselines Python : KMeans, DBSCAN, KNN, PCA — 3 datasets

| Champ | Valeur |
|-------|--------|
| **Sprint** | 19 |
| **exp_IDs** | `exp_S19_13` à `exp_S19_24` (4 modèles × 3 datasets) |
| **Priorité** | 🟡 Important |
| **Statut** | ✅ Terminé (KMeans + DBSCAN × 3 datasets — KNN/PCA non lancés) |
| **Durée estimée** | 4h |
| **Dépendances** | Aucune (modèles Python, pas de firmware requis) |
| **Fichiers cibles** | `experiments/exp_S19_13/` à `experiments/exp_S19_24/` |

---

## Contexte

Les modèles M4 (KMeansDetector, DBSCANDetector, KNNDetector, PCABaseline) sont des baselines non-supervisées Python implémentées dans `src/models/unsupervised/`. Ils servent de référence comparative pour les modèles CL (EWC, HDC, TinyOL, Mahalanobis) sur les mêmes datasets. Cette tâche couvre l'ensemble de la matrice 4×3.

Ces modèles tournent sur PC Python (pas de firmware C), mais peuvent utiliser des données streamées depuis le board via `scripts/sensor_stream.py` pour un contexte "board-connected".

---

## Nomenclature complète de cette tâche

| exp_ID | Modèle | Dataset | Script |
|--------|--------|---------|--------|
| `exp_S19_13` | kmeans | cwru | `train_kmeans.py` |
| `exp_S19_14` | kmeans | monitoring | `train_kmeans.py` |
| `exp_S19_15` | kmeans | pronostia | `train_kmeans.py` |
| `exp_S19_16` | dbscan | cwru | `train_dbscan.py` |
| `exp_S19_17` | dbscan | monitoring | `train_dbscan.py` |
| `exp_S19_18` | dbscan | pronostia | `train_dbscan.py` |
| `exp_S19_19` | knn | cwru | `train_unsupervised.py` |
| `exp_S19_20` | knn | monitoring | `train_unsupervised.py` |
| `exp_S19_21` | knn | pronostia | `train_unsupervised.py` |
| `exp_S19_22` | pca | cwru | `train_unsupervised.py` |
| `exp_S19_23` | pca | monitoring | `train_unsupervised.py` |
| `exp_S19_24` | pca | pronostia | `train_unsupervised.py` |

---

## Configs réutilisées

| Modèle | CWRU | Monitoring | Pronostia |
|--------|------|-----------|-----------|
| KMeans | `cwru_by_fault_config.yaml` | `monitoring_by_location_config.yaml` | `kmeans_pronostia_by_condition_config.yaml` |
| DBSCAN | `cwru_by_fault_config.yaml` | `monitoring_by_location_config.yaml` | `dbscan_pronostia_by_condition_config.yaml` |
| KNN | `cwru_by_fault_config.yaml` | `monitoring_by_location_config.yaml` | `pronostia_config.yaml` |
| PCA | `cwru_by_fault_config.yaml` | `monitoring_by_location_config.yaml` | `pronostia_config.yaml` |

---

## Procédure

### KMeans

```bash
# CWRU
python scripts/train_kmeans.py \
    --config configs/cwru_by_fault_config.yaml \
    --output experiments/exp_S19_13

# Monitoring
python scripts/train_kmeans.py \
    --config configs/monitoring_by_location_config.yaml \
    --output experiments/exp_S19_14

# Pronostia
python scripts/train_kmeans.py \
    --config configs/kmeans_pronostia_by_condition_config.yaml \
    --output experiments/exp_S19_15
```

### DBSCAN

```bash
python scripts/train_dbscan.py --config configs/cwru_by_fault_config.yaml         --output experiments/exp_S19_16
python scripts/train_dbscan.py --config configs/monitoring_by_location_config.yaml --output experiments/exp_S19_17
python scripts/train_dbscan.py --config configs/dbscan_pronostia_by_condition_config.yaml --output experiments/exp_S19_18
```

### KNN et PCA (via train_unsupervised.py)

```bash
# KNN
python scripts/train_unsupervised.py --model knn --config configs/cwru_by_fault_config.yaml         --output experiments/exp_S19_19
python scripts/train_unsupervised.py --model knn --config configs/monitoring_by_location_config.yaml --output experiments/exp_S19_20
python scripts/train_unsupervised.py --model knn --config configs/pronostia_config.yaml             --output experiments/exp_S19_21

# PCA
python scripts/train_unsupervised.py --model pca --config configs/cwru_by_fault_config.yaml         --output experiments/exp_S19_22
python scripts/train_unsupervised.py --model pca --config configs/monitoring_by_location_config.yaml --output experiments/exp_S19_23
python scripts/train_unsupervised.py --model pca --config configs/pronostia_config.yaml             --output experiments/exp_S19_24
```

---

## Métriques

Ces modèles sont des détecteurs d'anomalie non-supervisés — les métriques sont différentes des modèles CL supervisés :

| Métrique | Module | Description |
|----------|--------|-------------|
| `auroc` | `src/evaluation/anomaly_metrics.py` | AUC ROC (principal) |
| `f1_score` | `src/evaluation/anomaly_metrics.py` | F1 au seuil optimal |
| `precision` / `recall` | `src/evaluation/anomaly_metrics.py` | PR curve |
| `ram_peak_bytes` | `src/evaluation/memory_profiler.py` | via tracemalloc |
| `inference_latency_ms` | `src/evaluation/memory_profiler.py` | sur 100 runs |

> `acc_final` et `avg_forgetting` ne s'appliquent pas directement aux modèles non-supervisés sans mécanisme CL — remplacer par `auroc` comme métrique principale.

---

## Format JSON attendu (exemple KMeans/CWRU)

```json
{
  "exp_id": "S19_13",
  "model": "kmeans",
  "dataset": "cwru",
  "platform": "pc_python",
  "date": "2026-06-XX",
  "auroc": null,
  "f1_score": null,
  "precision": null,
  "recall": null,
  "ram_peak_bytes": null,
  "inference_latency_ms": null,
  "n_clusters": 4,
  "config_snapshot": "configs/cwru_by_fault_config.yaml"
}
```

---

## Vérification

- [ ] 12 dossiers `experiments/exp_S19_{13..24}/` créés
- [ ] Chaque `results.json` contient `auroc` et `f1_score`
- [ ] `platform: "pc_python"` (pas `nucleo_f439zi`)
- [ ] Aucune donnée brute commitée dans `experiments/`

---

## Questions ouvertes

- `TODO(arnaud)` : Faut-il inclure les baselines M4 Python dans la comparaison finale Gap 1 (validation industrielle) ?
