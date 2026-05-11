# S17-06 — Notebook récapitulatif cross-dataset Phase Anomaly Detection

| Champ | Valeur |
|-------|--------|
| **ID** | S17-06 |
| **Sprint** | Sprint 17 |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 2h |
| **Dépendances** | S14-10 (Monitoring), S15-06 (Pronostia), S17-05 (CWRU) |
| **Fichier cible** | `notebooks/cl_eval/summary_anomaly_detection.ipynb` |

---

## Objectif

Produire le notebook de synthèse finale de la Phase Anomaly Detection (Sprints 13–17) : tableau AUROC 6 modèles × 3 datasets, analyse cross-dataset, et recommandations pour le manuscrit et le portage STM32N6.

Ce notebook est le **livrable central** de la Phase Anomaly Detection — il sera référencé directement dans le manuscrit.

---

## Expériences couvertes

| Dataset | Scénario | Stratégie | Expériences |
|---------|----------|-----------|-------------|
| Monitoring | by_equipment | refit | exp_086–089 + 123 + 125 |
| Monitoring | by_equipment | accumulate | exp_127–130 + 124 + 126 |
| Monitoring | by_location | refit | exp_131–136 |
| Pronostia | by_condition | refit | exp_137–142 |
| Pronostia | by_condition | accumulate | exp_137b–142b |
| CWRU | by_[scenario] | refit | exp_143–148 |

---

## Structure du notebook

### Section 1 — Tableau AUROC cross-dataset (tableau principal)

```
                    Monitoring       Pronostia       CWRU
Modèle           refit | accum    refit | accum    refit
─────────────────────────────────────────────────────────
HDC               X.XX  | X.XX    X.XX  | X.XX    X.XX
TinyOL AE         X.XX  | X.XX    X.XX  | X.XX    X.XX
KMeans            X.XX  | X.XX    X.XX  | X.XX    X.XX
Mahalanobis       X.XX  | X.XX    X.XX  | X.XX    X.XX
DBSCAN            X.XX  | X.XX    X.XX  | X.XX    X.XX
EWC one-class     X.XX  | X.XX    X.XX  | X.XX    X.XX
```

### Section 2 — Classement modèles par dataset

Heatmap AUROC normalisé — identifie le meilleur modèle par dataset.

### Section 3 — RAM cross-dataset

Barplot `ram_peak_bytes` : 6 modèles × 3 datasets. Ligne de référence 64 Ko.
Focus : quels modèles restent sous 64 Ko sur tous les datasets ?

### Section 4 — Impact de la stratégie CL (refit vs accumulate)

Scatter plot : AUROC refit vs AUROC accumulate, par dataset.
- Points au-dessus de la diagonale y=x : accumulate > refit (meilleure rétention)
- Points en dessous : refit > accumulate (drift négatif de l'accumulation)

### Section 5 — Recommandations embarquées

Tableau de recommandations pour le portage STM32N6 :

| Critère | Meilleur modèle | Justification |
|---------|-----------------|---------------|
| AUROC global | — | — |
| RAM minimale | Mahalanobis | Cov 9×9 = 324 B (CWRU) |
| Robustesse faible ratio normal | — | — |
| Compatibilité one-class STM32N6 | — | NPU inference-only |
| Meilleur trade-off | — | — |

### Section 6 — Position par rapport au Triple Gap

- **Gap 1** (données industrielles réelles) : Pronostia + CWRU validés ✅
- **Gap 2** (< 100 Ko RAM) : RAM mesurée pour tous les modèles — quels modèles respectent 64 Ko sur les 3 datasets ?
- **Gap 3** (quantification INT8) : non abordé dans cette phase (Sprint 16+)

---

## Figures à sauvegarder

```
notebooks/figures/anomaly_detection/
├── summary_auroc_crossdataset.png      # tableau heatmap principal
├── summary_ram_crossdataset.png        # RAM comparaison
├── summary_refit_vs_accumulate.png     # scatter plot stratégies
└── summary_triple_gap_phase_ad.png     # positionnement Triple Gap
```

---

## Critères d'acceptation

- [ ] Notebook exécutable end-to-end sans erreur
- [ ] Tableau AUROC cross-dataset complet (6 × 5 colonnes minimum)
- [ ] Heatmap présente et sauvegardée
- [ ] Section "Recommandations embarquées" rédigée
- [ ] Section "Triple Gap" référence les numéros d'expériences précis

## Statut

⬜ À faire
