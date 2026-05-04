# S15-03 à S15-08 — exp_137–142 : 6 modèles Pronostia by_condition refit

| Champ | Valeur |
|-------|--------|
| **ID** | S15-03 → S15-08 |
| **Sprint** | Sprint 15 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 3h (0.5h × 6) |
| **Dépendances** | S15-01 (loader), S15-02 (config), S14-01 (EWC one-class) |
| **Fichiers cibles** | `experiments/exp_137/` → `experiments/exp_142/` |

---

## Objectif

Exécuter les 6 modèles d'anomaly detection sur Pronostia by_condition en stratégie refit et enregistrer les résultats standardisés.

---

## Expériences

| Exp | Modèle | Config | Commande |
|-----|--------|--------|---------|
| exp_137 | HDC | `configs/hdc_config.yaml` | `--model hdc --dataset pronostia --scenario by_condition --strategy refit` |
| exp_138 | TinyOL AE | `configs/tinyol_config.yaml` | `--model tinyol_ae --dataset pronostia --scenario by_condition --strategy refit` |
| exp_139 | KMeans | `configs/unsupervised_config.yaml` | `--model kmeans --dataset pronostia --scenario by_condition --strategy refit` |
| exp_140 | Mahalanobis | `configs/unsupervised_config.yaml` | `--model mahalanobis --dataset pronostia --scenario by_condition --strategy refit` |
| exp_141 | DBSCAN | `configs/unsupervised_config.yaml` | `--model dbscan --dataset pronostia --scenario by_condition --strategy refit` |
| exp_142 | EWC one-class | `configs/ewc_oneclass_config.yaml` | `--model ewc_oneclass --dataset pronostia --scenario by_condition --strategy refit` |

---

## Commande type (batch)

```bash
for exp_id model in \
    exp_137 hdc \
    exp_138 tinyol_ae \
    exp_139 kmeans \
    exp_140 mahalanobis \
    exp_141 dbscan \
    exp_142 ewc_oneclass; do
    python scripts/run_anomaly_detection.py \
        --model $model \
        --dataset pronostia \
        --scenario by_condition \
        --strategy refit \
        --exp_id $exp_id
done
```

---

## Sorties attendues

```
experiments/exp_137/
├── config_snapshot.yaml
└── results/
    └── metrics_anomaly.json   # AUROC [t1, t2, t3] + ram_peak_bytes
```

### Format `metrics_anomaly.json` (Pronostia)

```json
{
  "model": "hdc",
  "dataset": "pronostia",
  "scenario": "by_condition",
  "strategy": "refit",
  "failure_ratio": 0.10,
  "auroc_per_task": [0.0, 0.0, 0.0],
  "auroc_mean": 0.0,
  "avg_forgetting": 0.0,
  "ram_peak_bytes": 0,
  "inference_latency_ms": 0.0,
  "n_params": 0
}
```

---

## Points d'attention spécifiques Pronostia

- **Seuil EWC one-class** : avec input_dim=13 (override dans `ewc_oneclass_config.yaml`), vérifier que hidden_dim=64 charge dans la config avant de lancer exp_142
- **DBSCAN sur 13D** : eps=1.5 (override dans `unsupervised_config.yaml`) — si DBSCAN classe tout en anomalie ou rien, ajuster eps post-hoc et documenter
- **Ratio normal élevé (90%)** : les modèles one-class devraient bénéficier de ce ratio favorable — noter dans le notebook si AUROC < 0.7 pour un modèle (signal de problème)

---

## Critères d'acceptation

- [ ] exp_137–142 : `metrics_anomaly.json` présents avec `auroc_per_task` de longueur 3
- [ ] `config_snapshot.yaml` présents dans les 6 répertoires
- [ ] `failure_ratio: 0.10` reporté dans chaque `metrics_anomaly.json` (traçabilité)
- [ ] Aucun crash — si DBSCAN (exp_141) ne converge pas, documenter et ajuster eps

## Statut

⬜ À faire
