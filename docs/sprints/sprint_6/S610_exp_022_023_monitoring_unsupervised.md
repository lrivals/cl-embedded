# S6-09 — Run exp_022–023 : monitoring_by_location pour KMeans et DBSCAN

| Champ | Valeur |
|-------|--------|
| **ID** | S6-09 |
| **Sprint** | Sprint 6 — Phase 1 Extension (≥ 15 avril 2026) |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 1h |
| **Dépendances** | S6-07 (exp_016–019 terminées — valide monitoring_by_location_config) |
| **Fichiers cibles** | `experiments/exp_022–023/` |
| **Complété le** | 2026-04-14 |
| **Statut** | ✅ Terminé |

---

## Objectif

Lancer les 2 expériences non supervisées sur le scénario **monitoring_by_location** (Dataset 2) :

| Exp | Modèle | Script |
|-----|--------|--------|
| exp_022_kmeans_monitoring_by_location | KMeans | `scripts/train_unsupervised.py` |
| exp_023_dbscan_monitoring_by_location | DBSCAN | `scripts/train_unsupervised.py` |

**Critère de succès** : 2 répertoires créés avec `metrics.json` + `acc_matrix.npy` shape `(5, 5)`.

---

## Commandes d'exécution

```bash
# exp_022 — KMeans
python scripts/train_unsupervised.py \
  --config configs/unsupervised_config.yaml \
  --data_config configs/monitoring_by_location_config.yaml \
  --model kmeans \
  --exp_dir experiments/exp_022_kmeans_monitoring_by_location

# exp_023 — DBSCAN
python scripts/train_unsupervised.py \
  --config configs/unsupervised_config.yaml \
  --data_config configs/monitoring_by_location_config.yaml \
  --model dbscan \
  --exp_dir experiments/exp_023_dbscan_monitoring_by_location
```

---

## Critères d'acceptation

- [x] `experiments/exp_022_kmeans_monitoring_by_location/results/metrics_all.json` existe
- [x] `experiments/exp_023_dbscan_monitoring_by_location/results/metrics_all.json` existe
- [x] Chaque `acc_matrix.npy` de shape `(5, 5)` — `acc_matrix_kmeans_dataset2.npy` et `acc_matrix_dbscan_dataset2.npy`

> Note : le script génère `metrics_{model}_dataset2.json` + `metrics_all.json` (pas `metrics.json`),
> cohérent avec exp_016–023. Les noms de fichiers dans les critères ont été mis à jour en conséquence.

---

## Questions ouvertes

- `TODO(dorra)` : DBSCAN avec `eps=auto` (k-distance graph) est-il stable sur le Dataset 2
  multi-équipements ? Les densités peuvent varier fortement entre types d'équipements.
