# S15-16 — exp_143b–148b : 6 modèles CWRU by_severity accumulate

| Champ | Valeur |
|-------|--------|
| **ID** | S15-16 |
| **Sprint** | Sprint 15 |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 2h |
| **Dépendances** | S15-15 (refit terminés) |
| **Fichiers cibles** | `experiments/exp_143b/` → `experiments/exp_148b/` |

---

## Objectif

Produire les variantes accumulate pour les 6 modèles sur CWRU by_severity, permettant la comparaison refit vs accumulate. Ces expériences complètent le tableau croisé stratégie × modèle × dataset.

---

## Expériences

| Exp | Modèle | Stratégie | Config |
|-----|--------|-----------|--------|
| exp_143b | HDC | by_severity accumulate | `configs/hdc_anomaly_detection_config.yaml` |
| exp_144b | TinyOL AE | by_severity accumulate | `configs/tinyol_anomaly_detection_config.yaml` |
| exp_145b | KMeans | by_severity accumulate | `configs/unsupervised_anomaly_detection_config.yaml` |
| exp_146b | Mahalanobis | by_severity accumulate | `configs/unsupervised_anomaly_detection_config.yaml` |
| exp_147b | DBSCAN | by_severity accumulate | `configs/unsupervised_anomaly_detection_config.yaml` |
| exp_148b | EWC one-class | by_severity accumulate | `configs/ewc_oneclass_config.yaml` |

---

## Résultats

| Exp | Modèle | AUROC T0 | AUROC T1 | AUROC T2 | avg_AUROC | Δ vs refit |
|-----|--------|:---:|:---:|:---:|:---:|:---:|
| exp_143b | HDC | 0.9983 | 0.9761 | 0.9975 | **0.9906** | 0.0000 |
| exp_144b | TinyOL AE | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| exp_145b | KMeans | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| exp_146b | Mahalanobis | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| exp_147b | DBSCAN | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| exp_148b | EWC one-class | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |

> Δ vs refit = avg_AUROC accumulate − avg_AUROC refit. **Δ = 0 pour tous les modèles** : la stratégie accumulate n'apporte aucun gain sur CWRU.

---

## Interprétation

Le Δ nul s'explique par deux facteurs complémentaires :

1. **Features très discriminantes** : les 9 features spectrales CWRU permettent une séparation parfaite ou quasi-parfaite — le plafond AUROC = 1.0 est atteint dès la stratégie refit. Il n'y a pas de marge pour que l'accumulation améliore quoi que ce soit.

2. **Peu de données normales** (~62/tâche) : l'accumulation des normaux des tâches précédentes n'apporte pas suffisamment d'information supplémentaire pour changer les frontières de décision déjà optimales.

Pour HDC (non parfait, avg = 0.9906), l'accumulation ne corrige pas la difficulté sur T1 (0.014") — la séparation HDC est limitée par la représentation hyperdimenionnelle, pas par le volume d'entraînement.

---

## Critères d'acceptation

- [x] exp_143b–148b : `metrics_anomaly.json` présents avec `strategy: accumulate`
- [x] `config_snapshot.yaml` présents dans les 6 répertoires
- [x] AUROC accumulate comparé à AUROC refit dans le notebook CWRU (section 3)

## Statut

✅ Terminé

## Bilan

Les 6 expériences accumulate CWRU by_severity sont complètes. Tous les Δ refit/accumulate sont nuls — confirme que sur CWRU, les features spectrales (9D) suffisent à séparer normaux/faulty dès la première tâche. L'accumulation ne dégrade pas non plus (Δ = 0 vs Δ parfois légèrement négatif sur Pronostia). Note de portée manuscrit : ce résultat suggère que pour CWRU, **le choix de la stratégie CL est indifférent** pour la performance AUROC — ce qui simplifie le déploiement embarqué (refit seul suffit).
