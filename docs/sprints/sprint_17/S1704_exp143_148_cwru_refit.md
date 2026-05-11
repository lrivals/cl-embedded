# S17-04 — exp_143–148 : 6 modèles CWRU refit

| Champ | Valeur |
|-------|--------|
| **ID** | S17-04 |
| **Sprint** | Sprint 17 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | S17-02 (loader), S17-03 (config), S14-01 (EWC one-class) |
| **Fichiers cibles** | `experiments/exp_143/` → `experiments/exp_148/` |

---

## Objectif

Exécuter les 6 modèles d'anomaly detection sur CWRU (scénario retenu en S17-01) en stratégie refit. Ces 6 expériences clôturent la numérotation exp_123–148 de la Phase Anomaly Detection.

---

## Expériences

| Exp | Modèle | Config | Statut |
|-----|--------|--------|--------|
| exp_143 | HDC | `configs/hdc_config.yaml` | ⬜ |
| exp_144 | TinyOL AE | `configs/tinyol_config.yaml` | ⬜ |
| exp_145 | KMeans | `configs/unsupervised_config.yaml` | ⬜ |
| exp_146 | Mahalanobis | `configs/unsupervised_config.yaml` | ⬜ |
| exp_147 | DBSCAN | `configs/unsupervised_config.yaml` | ⬜ |
| exp_148 | EWC one-class | `configs/ewc_oneclass_config.yaml` | ⬜ |

---

## Commande (batch)

```bash
SCENARIO=$(grep SPLIT_STRATEGY configs/unsupervised_config.yaml | awk '{print $2}')

declare -A MODELS=(
    [exp_143]=hdc
    [exp_144]=tinyol_ae
    [exp_145]=kmeans
    [exp_146]=mahalanobis
    [exp_147]=dbscan
    [exp_148]=ewc_oneclass
)

for exp_id model in "${!MODELS[@]}" "${MODELS[@]}"; do
    python scripts/run_anomaly_detection.py \
        --model $model \
        --dataset cwru \
        --scenario $SCENARIO \
        --strategy refit \
        --exp_id $exp_id
done
```

---

## Points d'attention spécifiques CWRU

- **Peu de données normales (~77 / tâche)** : vérifier que l'entraînement converge (loss monitoring pendant fit_task)
- **EWC one-class (exp_148)** : threshold_percentile=80 (override config CWRU) — à surveiller
- **DBSCAN (exp_147)** : min_samples=3 et eps=0.8 (override) — risque de tout classer en bruit (-1). Si AUROC ≈ 0.5, documenter et augmenter eps
- **KMeans (exp_145)** : N_CLUSTERS=1 — détecteur very simple mais robuste sur peu de données

---

## Format `metrics_anomaly.json` (CWRU)

```json
{
  "model": "hdc",
  "dataset": "cwru",
  "scenario": "by_severity",
  "strategy": "refit",
  "normal_ratio": 0.10,
  "n_train_normal_per_task": 77,
  "auroc_per_task": [0.0, 0.0, 0.0],
  "auroc_mean": 0.0,
  "avg_forgetting": 0.0,
  "ram_peak_bytes": 0,
  "inference_latency_ms": 0.0
}
```

> Le champ `n_train_normal_per_task` doit être rempli pour la traçabilité (critique pour le manuscrit sur le faible ratio normal).

---

## Critères d'acceptation

- [ ] exp_143–148 : `metrics_anomaly.json` présents avec `auroc_per_task` de longueur 3
- [ ] `n_train_normal_per_task` reporté dans chaque JSON
- [ ] `config_snapshot.yaml` présents dans les 6 répertoires
- [ ] DBSCAN (exp_147) : si AUROC < 0.5, documenter la cause (tout classé en bruit) et ajuster eps dans un test additionnel

## Statut

⬜ En attente S17-01, S17-02
