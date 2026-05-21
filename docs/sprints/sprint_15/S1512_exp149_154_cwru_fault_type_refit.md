# S15-17 — exp_149–154 : 6 modèles CWRU by_fault_type refit

| Champ | Valeur |
|-------|--------|
| **ID** | S15-17 |
| **Sprint** | Sprint 15 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 3h (0.5h × 6) |
| **Dépendances** | S15-13 (loader), S15-14 (config) |
| **Fichiers cibles** | `experiments/exp_149/` → `experiments/exp_154/` |

---

## Objectif

Exécuter les 6 modèles d'anomaly detection sur CWRU by_fault_type en stratégie refit. Ce scénario est complémentaire à by_severity : il modélise l'**apparition successive de types de défauts inconnus** (déploiement incrémental de classes de défauts) plutôt que la progression d'un défaut existant.

---

## Scénario by_fault_type

| Tâche | Domaine | Défauts inclus (toutes sévérités) |
|-------|---------|----------------------------------|
| 0 | `ball` | Ball_007 + Ball_014 + Ball_021 |
| 1 | `inner_race` | IR_007 + IR_014 + IR_021 |
| 2 | `outer_race` | OR_007 + OR_014 + OR_021 |

Chaque tâche introduit un **type de défaut entièrement nouveau** — c'est la formalisation du scénario "class-incremental" transposé au monde de la détection d'anomalies.

---

## Expériences

| Exp | Modèle | Config |
|-----|--------|--------|
| exp_149 | HDC | `configs/hdc_anomaly_detection_config.yaml` |
| exp_150 | TinyOL AE | `configs/tinyol_anomaly_detection_config.yaml` |
| exp_151 | KMeans | `configs/unsupervised_anomaly_detection_config.yaml` |
| exp_152 | Mahalanobis | `configs/unsupervised_anomaly_detection_config.yaml` |
| exp_153 | DBSCAN | `configs/unsupervised_anomaly_detection_config.yaml` |
| exp_154 | EWC one-class | `configs/ewc_oneclass_config.yaml` |

---

## Commande type (batch)

```bash
for model in hdc tinyol_ae kmeans mahalanobis dbscan ewc_oneclass; do
    exp_id="exp_14X"  # 149–154 selon le modèle
    python scripts/run_anomaly_detection.py \
        --model $model \
        --dataset cwru \
        --scenario by_fault_type \
        --strategy refit \
        --exp_id $exp_id
done
```

---

## Résultats

| Exp | Modèle | AUROC T0 (ball) | AUROC T1 (inner) | AUROC T2 (outer) | avg_AUROC | AF | RAM | Latence |
|-----|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| exp_149 | HDC | 0.9950 | 0.9881 | 0.9971 | **0.9934** | −0.0035 | 8 104 B | 0.088 ms |
| exp_150 | TinyOL AE | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 0.0000 | 1 992 B | 0.098 ms |
| exp_151 | KMeans | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 0.0000 | 5 340 B | 0.501 ms |
| exp_152 | Mahalanobis | 0.9999 | 1.0000 | 1.0000 | **1.0000** | 0.0000 | 1 644 B | 0.008 ms |
| exp_153 | DBSCAN | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 0.0000 | 10 674 B | 0.220 ms |
| exp_154 | EWC one-class | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 0.0000 | 1 480 B | 0.084 ms |

> AF négatif pour HDC (−0.0035) = transfert positif léger = pas d'oubli. Mahalanobis T0 = 0.9999 (non 1.0000) — artefact numérique, le score est pratiquement parfait.

---

## Comparaison by_severity vs by_fault_type

| Modèle | avg_AUROC severity | avg_AUROC fault_type | Δ |
|--------|:-----------------:|:-------------------:|:-:|
| HDC | 0.9906 | 0.9934 | +0.0028 |
| TinyOL AE | 1.0000 | 1.0000 | 0.0000 |
| KMeans | 1.0000 | 1.0000 | 0.0000 |
| Mahalanobis | 1.0000 | 1.0000 | 0.0000 |
| DBSCAN | 1.0000 | 1.0000 | 0.0000 |
| EWC one-class | 1.0000 | 1.0000 | 0.0000 |

Les deux scénarios donnent des résultats quasi-identiques. Cela confirme que **les features spectrales CWRU encodent suffisamment d'information pour discriminer les anomalies indépendamment du type ou de la sévérité du défaut**. Du point de vue embarqué, le choix du scénario n'influence pas la performance.

---

## Critères d'acceptation

- [x] exp_149–154 : `metrics_anomaly.json` présents avec `scenario: by_fault_type`
- [x] `auroc_per_task_final` de longueur 3 dans chaque fichier
- [x] `config_snapshot.yaml` présents dans les 6 répertoires
- [x] Tableau comparatif by_severity vs by_fault_type produit dans le notebook S15-18

## Statut

✅ Terminé

## Bilan

Les 6 expériences refit sur CWRU by_fault_type sont complètes. Les résultats sont quasi-identiques à by_severity : 5 modèles à AUROC = 1.000, HDC légèrement en dessous (0.9934 vs 0.9906 en severity — légèrement meilleur en fault_type). Ce résultat confirme que les features spectrales CWRU sont robustes aux deux dimensions de variation (type et sévérité du défaut). L'absence d'accumulate pour ce scénario est justifiée par les Δ = 0 observés en by_severity — le même comportement est attendu.
