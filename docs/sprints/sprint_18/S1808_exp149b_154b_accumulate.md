# S18-08 — exp_149b–154b : 6 modèles Equipment Monitoring accumulate

| Champ | Valeur |
|-------|--------|
| **ID** | S18-08 |
| **Sprint** | Sprint 18 |
| **Priorité** | 🟢 Nice-to-have (si temps restant) |
| **Durée estimée** | 2h |
| **Dépendances** | S18-04 (exp_149–154 refit terminées) |
| **Fichiers cibles** | `experiments/exp_149b/` → `experiments/exp_154b/` |

---

## Objectif

Exécuter les 6 modèles d'anomaly detection sur Equipment Monitoring en stratégie **accumulate**. Avec ~50% de données normales, l'accumulate est particulièrement intéressant : le pool de normaux croît à chaque tâche, ce qui peut stabiliser les seuils de détection.

Cette tâche est conditionnelle : ne démarrer que si S18-04, S18-05, S18-06, S18-07 sont terminées et qu'il reste du budget temps.

---

## Expériences

| Exp | Modèle | Stratégie | Config | Statut |
|-----|--------|-----------|--------|--------|
| exp_149b | HDC | accumulate | `configs/hdc_config.yaml` | ⬜ |
| exp_150b | TinyOL AE | accumulate | `configs/tinyol_config.yaml` | ⬜ |
| exp_151b | KMeans | accumulate | `configs/unsupervised_config.yaml` | ⬜ |
| exp_152b | Mahalanobis | accumulate | `configs/unsupervised_config.yaml` | ⬜ |
| exp_153b | DBSCAN | accumulate | `configs/unsupervised_config.yaml` | ⬜ |
| exp_154b | EWC one-class | accumulate | `configs/ewc_oneclass_config.yaml` | ⬜ |

---

## Commande (batch)

```bash
declare -A MODELS=(
    [exp_149b]=hdc
    [exp_150b]=tinyol_ae
    [exp_151b]=kmeans
    [exp_152b]=mahalanobis
    [exp_153b]=dbscan
    [exp_154b]=ewc_oneclass
)

for exp_id in "${!MODELS[@]}"; do
    model=${MODELS[$exp_id]}
    python scripts/run_anomaly_detection.py \
        --model $model \
        --dataset equipment_monitoring \
        --scenario by_equipment_type \
        --strategy accumulate \
        --exp_id $exp_id
done
```

---

## Points d'attention

- **Accumulate sur Equipment Monitoring** : le pool de normaux croît (t1: ~N_pump, t2: ~N_pump+N_turbine, t3: tout). Vérifier si l'accumulation améliore ou dégrade l'AUROC vs refit.
- **Mémoire** : avec INPUT_DIM=4, même la stratégie accumulate reste légère. Documenter `ram_peak_bytes` pour confirmer la compatibilité 64 Ko.
- **KMeans (exp_151b)** : avec accumulate, le centroïde converge vers la moyenne de tous les normaux de tous les équipements vus → peut être moins spécialisé par type mais plus robuste globalement.

---

## Format `metrics_anomaly.json` (Equipment Monitoring accumulate)

```json
{
  "model": "hdc",
  "dataset": "equipment_monitoring",
  "scenario": "by_equipment_type",
  "strategy": "accumulate",
  "normal_ratio": 0.50,
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

- [ ] exp_149b–154b : `metrics_anomaly.json` présents avec `auroc_per_task` de longueur 3
- [ ] `n_train_normal_per_task` est une liste de 3 valeurs croissantes dans chaque JSON
- [ ] `config_snapshot.yaml` présents dans les 6 répertoires
- [ ] `ram_peak_bytes` documenté — confirmer compatibilité 64 Ko

## Statut

⬜ En attente S18-04 (et budget temps disponible)
