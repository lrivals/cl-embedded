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

## Critères d'acceptation

- [ ] exp_137b–142b : `metrics_anomaly.json` présents avec `strategy: accumulate`
- [ ] `config_snapshot.yaml` présents
- [ ] AUROC accumulate comparé à AUROC refit dans le notebook Pronostia

## Statut

⬜ À faire
