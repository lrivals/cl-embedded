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
|--------|------------------------------|-------------|-----------|
| HDC | D × input_dim × 4 B | — | — |
| TinyOL AE | (13×32 + 32×8 + 8×32 + 32×13) × 4 B ≈ 9 Ko | — | — |
| KMeans | k × 13 × 4 B = 156 B (k=3) | — | — |
| Mahalanobis | 13² × 4 B = 676 B (cov) | — | — |
| DBSCAN | N_train × 13 × 4 B (variable) | — | — |
| EWC one-class | 2 × (13×64 + 64×16 + 16×64 + 64×13) × 4 B ≈ 13 Ko | — | — |

> DBSCAN est le seul modèle dont la RAM dépend du nombre d'échantillons d'entraînement — documenter clairement cette propriété.

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

- [ ] `ram_peak_bytes` mesuré pour les 6 modèles sur input_dim=13
- [ ] Tous les modèles respectent ≤ 64 Ko (sauf DBSCAN qui peut dépasser — à documenter)
- [ ] Tableau comparatif RAM Monitoring (4D) vs Pronostia (13D) produit

## Statut

⬜ À faire
