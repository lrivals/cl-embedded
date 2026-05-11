# S18-04 — exp_149–154 : 6 modèles Equipment Monitoring refit

| Champ | Valeur |
|-------|--------|
| **ID** | S18-04 |
| **Sprint** | Sprint 18 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | S18-02 (loader), S18-03 (config), S14-01 (EWC one-class) |
| **Fichiers cibles** | `experiments/exp_149/` → `experiments/exp_154/` |

---

## Objectif

Exécuter les 6 modèles d'anomaly detection sur Equipment Monitoring (by_equipment_type) en stratégie refit.

---

## Expériences

| Exp | Modèle | Config | Statut |
|-----|--------|--------|--------|
| exp_149 | HDC | `configs/hdc_config.yaml` | ⬜ |
| exp_150 | TinyOL AE | `configs/tinyol_config.yaml` | ⬜ |
| exp_151 | KMeans | `configs/unsupervised_config.yaml` | ⬜ |
| exp_152 | Mahalanobis | `configs/unsupervised_config.yaml` | ⬜ |
| exp_153 | DBSCAN | `configs/unsupervised_config.yaml` | ⬜ |
| exp_154 | EWC one-class | `configs/ewc_oneclass_config.yaml` | ⬜ |

---

## Commande (batch)

```bash
declare -A MODELS=(
    [exp_149]=hdc
    [exp_150]=tinyol_ae
    [exp_151]=kmeans
    [exp_152]=mahalanobis
    [exp_153]=dbscan
    [exp_154]=ewc_oneclass
)

for exp_id in "${!MODELS[@]}"; do
    model=${MODELS[$exp_id]}
    python scripts/run_anomaly_detection.py \
        --model $model \
        --dataset equipment_monitoring \
        --scenario by_equipment_type \
        --strategy refit \
        --exp_id $exp_id
done
```

---

## Points d'attention spécifiques Equipment Monitoring

- **Ratio favorable (~50% normaux)** : conditions idéales pour one-class — les modèles devraient converger facilement. Documenter si un modèle échoue malgré ces conditions.
- **4 features seulement** : espace de très faible dimensionnalité — Mahalanobis (cov 4×4 = 64 B @ FP32) est le modèle le plus léger. AUROC attendu élevé car l'espace est séparable.
- **DBSCAN (exp_153)** : avec EPS=0.5 et min_samples=5, vérifier que les normaux forment bien un cluster unique (pas de fragmentation).
- **Comparaison avec CWRU** : noter l'impact du ratio normal sur l'AUROC. Equipment Monitoring devrait surpasser CWRU pour tous les modèles.

---

## Format `metrics_anomaly.json` (Equipment Monitoring)

```json
{
  "model": "hdc",
  "dataset": "equipment_monitoring",
  "scenario": "by_equipment_type",
  "strategy": "refit",
  "normal_ratio": 0.50,
  "n_train_normal_per_task": 0,
  "auroc_per_task": [0.0, 0.0, 0.0],
  "auroc_mean": 0.0,
  "avg_forgetting": 0.0,
  "ram_peak_bytes": 0,
  "inference_latency_ms": 0.0
}
```

---

## Critères d'acceptation

- [ ] exp_149–154 : `metrics_anomaly.json` présents avec `auroc_per_task` de longueur 3
- [ ] `n_train_normal_per_task` reporté dans chaque JSON
- [ ] `config_snapshot.yaml` présents dans les 6 répertoires
- [ ] `ram_peak_bytes` documenté pour comparaison cross-dataset

## Statut

⬜ En attente S18-01, S18-02
