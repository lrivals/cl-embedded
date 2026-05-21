# S15-04 — RAM profiling modèles sur Pronostia (input_dim=13)

| Champ | Valeur |
|-------|--------|
| **ID** | S15-04 |
| **Sprint** | Sprint 15 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 1h |
| **Dépendances** | S15-03 → S15-08 (expériences terminées) |
| **Fichier cible** | `evaluation/memory_profiler.py` |

---

## Objectif

Mesurer et comparer l'empreinte RAM des 6 modèles sur Pronostia (input_dim=13) et vérifier la conformité à la contrainte 64 Ko. Les modèles dont la taille varie avec input_dim (HDC, EWC one-class, Mahalanobis) ont une empreinte différente de Monitoring.

---

## Tableau théorique vs mesuré

| Modèle | RAM théorique (input_dim=13) | RAM mesurée | ≤ 64 Ko ? |
| ------ | ---------------------------- | ----------- | --------- |
| HDC | D × input_dim × 4 B | 15 272 B (14.9 Ko) | ✅ |
| TinyOL AE | (13×32 + 32×8 + 8×32 + 32×13) × 4 B ≈ 9 Ko | 1 992 B (1.9 Ko) | ✅ |
| KMeans | k × 13 × 4 B = 156 B (k=3) | 5 698 B (5.6 Ko) | ✅ |
| Mahalanobis | 13² × 4 B = 676 B (cov) | 1 756 B (1.7 Ko) | ✅ |
| DBSCAN | N_train × 13 × 4 B (variable) | 201 746 B (197 Ko) | ❌ dépasse |
| EWC one-class | 2 × (13×64 + 64×16 + 16×64 + 64×13) × 4 B ≈ 13 Ko | 1 480 B (1.4 Ko) | ✅ |

> DBSCAN est le seul modèle dont la RAM dépend du nombre d'échantillons d'entraînement — 201 Ko pour l'ensemble d'entraînement Pronostia (strategy refit, ~3 000 points × 13 features × 4 B). Ce dépassement est attendu et documenté ; DBSCAN est exclu du déploiement STM32N6.

---

## Commande

```bash
python scripts/profile_memory.py \
    --model all \
    --dataset pronostia \
    --config configs/unsupervised_config.yaml \
    --ewc_config configs/ewc_oneclass_config.yaml
```

---

## Critères d'acceptation

- [x] `ram_peak_bytes` mesuré pour les 6 modèles sur input_dim=13
- [x] Tous les modèles respectent ≤ 64 Ko (sauf DBSCAN qui dépasse — documenté ci-dessus)
- [x] Tableau comparatif RAM Monitoring (4D) vs Pronostia (13D) produit (section 4 du notebook S15-06)

## Statut

✅ Terminé

## Bilan

Les `ram_peak_bytes` sont mesurés dans chaque `metrics_anomaly.json` (exp_137–142). Cinq modèles sur six respectent la contrainte 64 Ko sur Pronostia (input_dim=13) : HDC (14.9 Ko), TinyOL AE (1.9 Ko), KMeans (5.6 Ko), Mahalanobis (1.7 Ko), EWC one-class (1.4 Ko). DBSCAN dépasse largement (197 Ko) en raison du stockage de l'ensemble d'entraînement complet — ce comportement est structurel et non lié à input_dim. Le tableau comparatif 4D vs 13D est produit dans la section 4 du notebook Pronostia (cell 14, exécutée).
