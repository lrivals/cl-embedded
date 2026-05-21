# S15-15 — exp_143–148 : 6 modèles CWRU by_severity refit

| Champ | Valeur |
|-------|--------|
| **ID** | S15-15 |
| **Sprint** | Sprint 15 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 3h (0.5h × 6) |
| **Dépendances** | S15-13 (loader), S15-14 (config), S14-01 (EWC one-class) |
| **Fichiers cibles** | `experiments/exp_143/` → `experiments/exp_148/` |

---

## Objectif

Exécuter les 6 modèles d'anomaly detection sur CWRU by_severity en stratégie refit et enregistrer les résultats standardisés.

---

## Scénario by_severity

Modélise la **dégradation progressive** d'un défaut de roulement : le diamètre du défaut augmente de tâche en tâche (0.007" → 0.014" → 0.021"). C'est le scénario le plus naturel pour la maintenance prédictive (drift temporel de la sévérité).

| Tâche | Domaine | Défauts (train faulty = test uniquement) |
|-------|---------|------------------------------------------|
| 0 | `007` | Ball_007 + IR_007 + OR_007 |
| 1 | `014` | Ball_014 + IR_014 + OR_014 |
| 2 | `021` | Ball_021 + IR_021 + OR_021 |

---

## Expériences

| Exp | Modèle | Config |
|-----|--------|--------|
| exp_143 | HDC | `configs/hdc_anomaly_detection_config.yaml` |
| exp_144 | TinyOL AE | `configs/tinyol_anomaly_detection_config.yaml` |
| exp_145 | KMeans | `configs/unsupervised_anomaly_detection_config.yaml` |
| exp_146 | Mahalanobis | `configs/unsupervised_anomaly_detection_config.yaml` |
| exp_147 | DBSCAN | `configs/unsupervised_anomaly_detection_config.yaml` |
| exp_148 | EWC one-class | `configs/ewc_oneclass_config.yaml` |

---

## Commande type (batch)

```bash
for model in hdc tinyol_ae kmeans mahalanobis dbscan ewc_oneclass; do
    exp_id="exp_14X"  # 143–148 selon le modèle
    python scripts/run_anomaly_detection.py \
        --model $model \
        --dataset cwru \
        --scenario by_severity \
        --strategy refit \
        --exp_id $exp_id
done
```

---

## Sorties attendues

```
experiments/exp_143/
├── config_snapshot.yaml
└── results/
    └── metrics_anomaly.json   # AUROC [t0, t1, t2] + ram_peak_bytes
```

### Format `metrics_anomaly.json` (CWRU)

```json
{
  "model": "hdc",
  "dataset": "cwru",
  "scenario": "by_severity",
  "strategy": "refit",
  "failure_ratio": null,
  "n_train_normal_per_task": [62, 62, 61],
  "auroc_per_task_final": [0.0, 0.0, 0.0],
  "avg_auroc": 0.0,
  "auroc_forgetting": 0.0,
  "auroc_bwt": 0.0,
  "ram_peak_bytes": 0,
  "inference_latency_ms": 0.0,
  "n_params": 0
}
```

> Note : `failure_ratio` est `null` pour CWRU (label binaire issu du dataset, pas d'un seuil temporel).

---

## Points d'attention spécifiques CWRU

- **Peu de normaux (~62/tâche)** : les modèles one-class peuvent être instables — un warning est attendu à chaque tâche
- **DBSCAN avec EPS=0.8, MIN_SAMPLES=3** : overrides dans la config CWRU — si DBSCAN classe tout en anomalie, ajuster EPS et documenter
- **KMeans avec N_CLUSTERS=1** : un seul centroïde normal (override justifié par le faible nombre de normaux)
- **config_snapshot.yaml** : contient des exp_ids erronés (artefact de ré-exécution depuis des configs anciennes) — `metrics_anomaly.json` est la référence fiable

---

## Résultats

| Exp | Modèle | AUROC T0 (007) | AUROC T1 (014) | AUROC T2 (021) | avg_AUROC | AF | RAM | Latence |
|-----|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| exp_143 | HDC | 0.9983 | 0.9761 | 0.9975 | **0.9906** | −0.0033 | 8 104 B | 0.096 ms |
| exp_144 | TinyOL AE | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 0.0000 | 1 992 B | 0.131 ms |
| exp_145 | KMeans | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 0.0000 | 5 432 B | 0.327 ms |
| exp_146 | Mahalanobis | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 0.0000 | 1 644 B | 0.008 ms |
| exp_147 | DBSCAN | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 0.0000 | 10 674 B | 0.219 ms |
| exp_148 | EWC one-class | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 0.0000 | 1 480 B | 0.361 ms |

> AF = `auroc_forgetting` (négatif = transfert positif = pas d'oubli). HDC est le seul modèle avec AUROC < 1.0 — la tâche T1 (014) est légèrement moins discriminée (0.9761). Les 5 autres modèles atteignent AUROC = 1.000 sur les 3 tâches.

**Analyse RAM :**

| Modèle | RAM mesurée | ≤ 64 Ko ? |
|--------|:-----------:|:---------:|
| HDC | 8 104 B (7.9 Ko) | ✅ |
| TinyOL AE | 1 992 B (1.9 Ko) | ✅ |
| KMeans | 5 432 B (5.3 Ko) | ✅ |
| Mahalanobis | 1 644 B (1.6 Ko) | ✅ |
| DBSCAN | 10 674 B (10.4 Ko) | ✅ |
| EWC one-class | 1 480 B (1.4 Ko) | ✅ |

> Contrairement à Pronostia, **tous les modèles respectent la contrainte 64 Ko sur CWRU** — y compris DBSCAN (10.4 Ko), car le petit nombre de normaux d'entraînement (~62) limite le stockage des points en mémoire.

---

## Critères d'acceptation

- [x] exp_143–148 : `metrics_anomaly.json` présents avec `auroc_per_task_final` de longueur 3
- [x] `config_snapshot.yaml` présents dans les 6 répertoires
- [x] `n_train_normal_per_task` reporté dans chaque `metrics_anomaly.json` (traçabilité)
- [x] `failure_ratio: null` dans chaque `metrics_anomaly.json`
- [x] DBSCAN (exp_147) a convergé avec EPS=0.8, MIN_SAMPLES=3 sans ajustement post-hoc

## Statut

✅ Terminé

## Bilan

Les 6 expériences refit sur CWRU by_severity sont complètes. 5 modèles sur 6 atteignent AUROC = 1.000 — les features spectrales CWRU permettent une séparation quasi-parfaite malgré seulement ~10% de données normales. HDC est le seul modèle non-parfait (avg_AUROC = 0.9906, légère difficulté sur T1 = sévérité intermédiaire 0.014"). AF négatif pour HDC (−0.0033) indique un léger transfert positif. Tous les modèles respectent la contrainte RAM 64 Ko — DBSCAN inclus (10.4 Ko vs 197 Ko sur Pronostia).
