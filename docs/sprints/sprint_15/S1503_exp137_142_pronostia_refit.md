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

## Résultats

| Exp | Modèle | AUROC C1 | AUROC C2 | AUROC C3 | avg_AUROC | AF |
| --- | ------ | :------: | :------: | :------: | :-------: | :-: |
| exp_137 | HDC | 0.6450 | 0.7839 | 0.7404 | **0.7231** | −0.018 |
| exp_138 | TinyOL AE | 0.7009 | 0.6055 | 0.8740 | **0.7268** | +0.183 |
| exp_139 | KMeans | 0.7019 | 0.6440 | 0.8747 | **0.7402** | +0.027 |
| exp_140 | Mahalanobis | 0.6026 | 0.5202 | 0.8791 | **0.6673** | +0.205 |
| exp_141 | DBSCAN | 0.6861 | 0.5864 | 0.8378 | **0.7034** | +0.106 |
| exp_142 | EWC one-class | 0.7041 | 0.6180 | 0.8273 | **0.7165** | +0.185 |

> AF = `auroc_forgetting` (positif = oubli, négatif = transfert positif). HDC est le seul modèle sans oubli notable sur Pronostia.

---

## Critères d'acceptation

- [x] exp_137–142 : `metrics_anomaly.json` présents avec `auroc_per_task_final` de longueur 3
- [x] `config_snapshot.yaml` présents dans les 6 répertoires
- [x] `failure_ratio: 0.10` reporté dans chaque `metrics_anomaly.json` (traçabilité)
- [x] Aucun crash — DBSCAN (exp_141) a convergé avec `EPSILON: 1.5`, pas d'ajustement nécessaire

## Statut

✅ Terminé

## Bilan

Les 6 expériences refit sur Pronostia by_condition sont complètes. KMeans obtient le meilleur avg_AUROC (0.7402), Mahalanobis le plus faible (0.6673). L'oubli est élevé pour TinyOL AE, EWC one-class et Mahalanobis (AF > 0.18), ce qui s'explique par le refit complet à chaque tâche. HDC montre un léger transfert positif (AF = −0.018). DBSCAN a convergé avec `EPSILON=1.5` sans ajustement post-hoc. La clé JSON réelle est `auroc_per_task_final` (non `auroc_per_task` comme dans la spécification initiale).
