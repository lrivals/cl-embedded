# S6-08 — Run exp_020–021 : pump_by_id pour KMeans et DBSCAN

| Champ | Valeur |
|-------|--------|
| **ID** | S6-08 |
| **Sprint** | Sprint 6 — Phase 1 Extension (≥ 15 avril 2026) |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 1h |
| **Dépendances** | S6-06 (exp_012–015 terminées — valide que pump_by_id_config fonctionne) |
| **Fichiers cibles** | `experiments/exp_020–021/` |
| **Complété le** | 2026-04-14 |
| **Statut** | ✅ Terminé |

---

## Objectif

Lancer les 2 expériences non supervisées sur le scénario **pump_by_id** :

| Exp | Modèle | Script |
|-----|--------|--------|
| exp_020_kmeans_pump_by_id | KMeans (k_method=silhouette) | `scripts/train_unsupervised.py` |
| exp_021_dbscan_pump_by_id | DBSCAN (eps auto) | `scripts/train_unsupervised.py` |

**Critère de succès** : 2 répertoires créés avec `metrics.json` + `acc_matrix.npy`.

---

## Commandes d'exécution

```bash
# exp_020 — KMeans
python scripts/train_unsupervised.py \
  --config configs/unsupervised_config.yaml \
  --data_config configs/pump_by_id_config.yaml \
  --model kmeans \
  --exp_dir experiments/exp_020_kmeans_pump_by_id

# exp_021 — DBSCAN
python scripts/train_unsupervised.py \
  --config configs/unsupervised_config.yaml \
  --data_config configs/pump_by_id_config.yaml \
  --model dbscan \
  --exp_dir experiments/exp_021_dbscan_pump_by_id
```

---

## Critères d'acceptation

- [x] `experiments/exp_020_kmeans_pump_by_id/results/metrics_all.json` existe
- [x] `experiments/exp_021_dbscan_pump_by_id/results/metrics_all.json` existe
- [x] Chaque `acc_matrix.npy` de shape `(5, 5)` — `acc_matrix_kmeans_dataset1.npy` et `acc_matrix_dbscan_dataset1.npy`

> Note : le script génère `metrics_{model}_dataset1.json` + `metrics_all.json` (pas `metrics.json`),
> cohérent avec exp_016–019. Les noms de fichiers dans les critères ont été mis à jour en conséquence.

---

## Questions ouvertes

- `TODO(arnaud)` : Pour les modèles non supervisés sur pump_by_id, le seuil d'anomalie est calculé
  sur Pump_ID=1 (Task 0). Si les Pump_IDs ont des distributions très différentes, ce seuil peut être
  inadapté pour les tâches suivantes. Recalibrer le seuil par tâche ?
