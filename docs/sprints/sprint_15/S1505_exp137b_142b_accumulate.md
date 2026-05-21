# S15-05 — exp_137b–142b : 6 modèles Pronostia by_condition accumulate

| Champ | Valeur |
|-------|--------|
| **ID** | S15-05 |
| **Sprint** | Sprint 15 |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 2h |
| **Dépendances** | S15-03 → S15-08 (refit terminés) |
| **Fichiers cibles** | `experiments/exp_137b/` → `experiments/exp_142b/` |

---

## Objectif

Produire les variantes accumulate (v2) pour les 6 modèles sur Pronostia, permettant la comparaison refit vs accumulate dans le notebook. Avec seulement 3 tâches, l'impact de la stratégie accumulate sera plus lisible que sur Monitoring (3 points AUROC par courbe).

---

## Expériences

| Exp | Modèle | Stratégie | Config |
|-----|--------|-----------|--------|
| exp_137b | HDC | by_condition accumulate | `configs/hdc_config.yaml` |
| exp_138b | TinyOL AE | by_condition accumulate | `configs/tinyol_config.yaml` |
| exp_139b | KMeans | by_condition accumulate | `configs/unsupervised_config.yaml` |
| exp_140b | Mahalanobis | by_condition accumulate | `configs/unsupervised_config.yaml` |
| exp_141b | DBSCAN | by_condition accumulate | `configs/unsupervised_config.yaml` |
| exp_142b | EWC one-class | by_condition accumulate | `configs/ewc_oneclass_config.yaml` |

---

## Note sur la numérotation

Les expériences accumulate Pronostia utilisent le suffixe `b` (ex. `exp_137b`) car les numéros 143–148 sont réservés pour CWRU (Sprint 16). Ce suffixe est une convention temporaire — si la numérotation est réorganisée, mettre à jour le `experiments_tracker.md`.

---

## Résultats

| Exp | Modèle | AUROC C1 | AUROC C2 | AUROC C3 | avg_AUROC | Δ vs refit |
| --- | ------ | :------: | :------: | :------: | :-------: | :--------: |
| exp_137b | HDC | 0.6450 | 0.7839 | 0.7404 | **0.7231** | 0.0000 |
| exp_138b | TinyOL AE | 0.6963 | 0.6226 | 0.8539 | **0.7243** | −0.0025 |
| exp_139b | KMeans | 0.6738 | 0.6500 | 0.8492 | **0.7243** | −0.0159 |
| exp_140b | Mahalanobis | 0.6123 | 0.6857 | 0.8323 | **0.7101** | +0.0429 |
| exp_141b | DBSCAN | 0.6443 | 0.6745 | 0.8134 | **0.7107** | +0.0073 |
| exp_142b | EWC one-class | 0.7098 | 0.6293 | 0.8281 | **0.7224** | +0.0059 |

> Δ vs refit = avg_AUROC accumulate − avg_AUROC refit. Valeurs proches de 0 : la stratégie accumulate n'apporte pas de gain notable sur Pronostia (3 tâches seulement).

---

## Critères d'acceptation

- [x] exp_137b–142b : `metrics_anomaly.json` présents avec `strategy: accumulate`
- [x] `config_snapshot.yaml` présents dans les 6 répertoires
- [x] AUROC accumulate comparé à AUROC refit dans le notebook Pronostia (section 2)

## Statut

✅ Terminé

## Bilan

Les 6 expériences accumulate sont complètes. Les écarts refit vs accumulate sont très faibles (|Δ| < 0.05 pour tous les modèles), ce qui confirme que sur Pronostia (3 tâches, peu de données par tâche), accumuler les données n'apporte pas d'avantage significatif par rapport au refit simple. HDC donne des résultats identiques (modèle non-paramétrique, insensible à la stratégie d'entraînement).
