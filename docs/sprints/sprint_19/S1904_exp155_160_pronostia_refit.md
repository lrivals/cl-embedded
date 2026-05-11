# S19-04 — exp_155–160 : 6 modèles Pronostia refit

| Champ | Valeur |
|-------|--------|
| **ID** | S19-04 |
| **Sprint** | Sprint 19 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | S19-02 (loader), S19-03 (config), S14-01 (EWC one-class) |
| **Fichiers cibles** | `experiments/exp_155/` → `experiments/exp_160/` |

---

## Objectif

Exécuter les 6 modèles d'anomaly detection sur Pronostia (by_bearing_condition) en stratégie refit. Ces 6 expériences clôturent la numérotation de la Phase Anomaly Detection (exp_086–160).

---

## Expériences

| Exp | Modèle | Config | Statut |
|-----|--------|--------|--------|
| exp_155 | HDC | `configs/hdc_config.yaml` | ⬜ |
| exp_156 | TinyOL AE | `configs/tinyol_config.yaml` | ⬜ |
| exp_157 | KMeans | `configs/unsupervised_config.yaml` | ⬜ |
| exp_158 | Mahalanobis | `configs/unsupervised_config.yaml` | ⬜ |
| exp_159 | DBSCAN | `configs/unsupervised_config.yaml` | ⬜ |
| exp_160 | EWC one-class | `configs/ewc_oneclass_config.yaml` | ⬜ |

---

## Commande (batch)

```bash
declare -A MODELS=(
    [exp_155]=hdc
    [exp_156]=tinyol_ae
    [exp_157]=kmeans
    [exp_158]=mahalanobis
    [exp_159]=dbscan
    [exp_160]=ewc_oneclass
)

for exp_id in "${!MODELS[@]}"; do
    model=${MODELS[$exp_id]}
    python scripts/run_anomaly_detection.py \
        --model $model \
        --dataset pronostia \
        --scenario by_bearing_condition \
        --strategy refit \
        --exp_id $exp_id
done
```

---

## Points d'attention spécifiques Pronostia

- **13 features** : espace de haute dimensionnalité — KMeans et Mahalanobis peuvent être sensibles au curse of dimensionality. Comparer avec Equipment Monitoring (4D).
- **Mahalanobis (exp_158)** : matrice de covariance 13×13 = 676 B @ FP32. Vérifier que `REG_COVAR=1e-5` est suffisant en end_of_life (peu de normaux).
- **Ratio ~90% normal** : beaucoup de normaux d'entraînement → seuils de détection stables. AUROC attendu élevé en early_life, plus variable en end_of_life.
- **DBSCAN (exp_159)** : vérifier qu'avec 13D, les distances euclidiennes sont encore discriminantes (concentration des distances en haute dim).

---

## Format `metrics_anomaly.json` (Pronostia)

```json
{
  "model": "hdc",
  "dataset": "pronostia",
  "scenario": "by_bearing_condition",
  "strategy": "refit",
  "normal_ratio": 0.90,
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

- [ ] exp_155–160 : `metrics_anomaly.json` présents avec `auroc_per_task` de longueur 3
- [ ] `n_train_normal_per_task` reporté dans chaque JSON
- [ ] `config_snapshot.yaml` présents dans les 6 répertoires
- [ ] `ram_peak_bytes` documenté, notamment pour Mahalanobis 13D (676 B @ FP32)

## Statut

⬜ En attente S19-01, S19-02
