# S19-08 — exp_155b–160b : 6 modèles Pronostia accumulate

| Champ | Valeur |
|-------|--------|
| **ID** | S19-08 |
| **Sprint** | Sprint 19 |
| **Priorité** | 🟢 Nice-to-have (si temps restant) |
| **Durée estimée** | 2h |
| **Dépendances** | S19-04 (exp_155–160 refit terminées) |
| **Fichiers cibles** | `experiments/exp_155b/` → `experiments/exp_160b/` |

---

## Objectif

Exécuter les 6 modèles d'anomaly detection sur Pronostia en stratégie **accumulate**. Avec ~90% de normaux, l'accumulate est particulièrement intéressant sur Pronostia : le pool de normaux early_life + mid_life est très large, ce qui devrait stabiliser les modèles.

Cette tâche est conditionnelle : ne démarrer que si S19-04, S19-05, S19-06, S19-07 sont terminées et qu'il reste du budget temps.

---

## Expériences

| Exp | Modèle | Stratégie | Config | Statut |
|-----|--------|-----------|--------|--------|
| exp_155b | HDC | accumulate | `configs/hdc_config.yaml` | ⬜ |
| exp_156b | TinyOL AE | accumulate | `configs/tinyol_config.yaml` | ⬜ |
| exp_157b | KMeans | accumulate | `configs/unsupervised_config.yaml` | ⬜ |
| exp_158b | Mahalanobis | accumulate | `configs/unsupervised_config.yaml` | ⬜ |
| exp_159b | DBSCAN | accumulate | `configs/unsupervised_config.yaml` | ⬜ |
| exp_160b | EWC one-class | accumulate | `configs/ewc_oneclass_config.yaml` | ⬜ |

---

## Commande (batch)

```bash
declare -A MODELS=(
    [exp_155b]=hdc
    [exp_156b]=tinyol_ae
    [exp_157b]=kmeans
    [exp_158b]=mahalanobis
    [exp_159b]=dbscan
    [exp_160b]=ewc_oneclass
)

for exp_id in "${!MODELS[@]}"; do
    model=${MODELS[$exp_id]}
    python scripts/run_anomaly_detection.py \
        --model $model \
        --dataset pronostia \
        --scenario by_bearing_condition \
        --strategy accumulate \
        --exp_id $exp_id
done
```

---

## Points d'attention

- **Accumulate sur Pronostia** : avec ~90% de normaux, le pool croît rapidement. Le modèle en end_of_life a accès à tous les normaux early+mid+end → seuil plus stable.
- **Mémoire Mahalanobis (exp_158b)** : en accumulate, la cov 13×13 est estimée sur de plus en plus de données → plus stable mais vérifier `ram_peak_bytes` (676 B × facteur accumulate).
- **DBSCAN (exp_159b)** : avec 13D, les distances en accumulate peuvent concentrer encore plus → surveiller si min_samples=5 reste adapté avec un grand pool de normaux.

---

## Format `metrics_anomaly.json` (Pronostia accumulate)

```json
{
  "model": "hdc",
  "dataset": "pronostia",
  "scenario": "by_bearing_condition",
  "strategy": "accumulate",
  "normal_ratio": 0.90,
  "n_train_normal_per_task": [0, 0, 0],
  "auroc_per_task": [0.0, 0.0, 0.0],
  "auroc_mean": 0.0,
  "avg_forgetting": 0.0,
  "ram_peak_bytes": 0,
  "inference_latency_ms": 0.0
}
```

> `n_train_normal_per_task` est une liste croissante pour la stratégie accumulate.

---

## Critères d'acceptation

- [ ] exp_155b–160b : `metrics_anomaly.json` présents avec `auroc_per_task` de longueur 3
- [ ] `n_train_normal_per_task` est une liste de 3 valeurs croissantes dans chaque JSON
- [ ] `config_snapshot.yaml` présents dans les 6 répertoires
- [ ] `ram_peak_bytes` documenté pour Mahalanobis 13D accumulate

## Statut

⬜ En attente S19-04 (et budget temps disponible)
