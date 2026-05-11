# S19-06 — Notebook summary final cross-dataset Phase Anomaly Detection (Sprints 17+18+19)

| Champ | Valeur |
|-------|--------|
| **ID** | S19-06 |
| **Sprint** | Sprint 19 |
| **Priorité** | 🔴 Critique — livrable central Phase Anomaly Detection |
| **Durée estimée** | 2h |
| **Dépendances** | S18-06 (summary Sprints 17+18), S19-05 (notebook Pronostia terminé) |
| **Fichier cible** | `notebooks/cl_eval/summary_anomaly_detection.ipynb` |

---

## Objectif

Finaliser `summary_anomaly_detection.ipynb` avec les résultats des 3 datasets (CWRU + Equipment Monitoring + Pronostia). Ce notebook est le **livrable central** de la Phase Anomaly Detection — il sera référencé directement dans le manuscrit pour justifier le Triple Gap.

---

## Expériences couvertes (Phase Anomaly Detection complète)

| Dataset | Scénario | Stratégie | Expériences |
|---------|----------|-----------|-------------|
| CWRU | by_severity (ou by_fault_type) | refit | exp_143–148 |
| CWRU | by_severity | accumulate | exp_143b–148b (si S17-08) |
| Equipment Monitoring | by_equipment_type | refit | exp_149–154 |
| Equipment Monitoring | by_equipment_type | accumulate | exp_149b–154b (si S18-08) |
| Pronostia | by_bearing_condition | refit | exp_155–160 |
| Pronostia | by_bearing_condition | accumulate | exp_155b–160b (si S19-08) |

---

## Structure finale du notebook

### Section 1 — Tableau AUROC cross-dataset (tableau principal)

```
                    CWRU           Equipment Monitoring    Pronostia
Modèle           refit | accum    refit | accum           refit | accum
───────────────────────────────────────────────────────────────────────
HDC               X.XX  | —       X.XX  | —               X.XX  | —
TinyOL AE         X.XX  | —       X.XX  | —               X.XX  | —
KMeans            X.XX  | —       X.XX  | —               X.XX  | —
Mahalanobis       X.XX  | —       X.XX  | —               X.XX  | —
DBSCAN            X.XX  | —       X.XX  | —               X.XX  | —
EWC one-class     X.XX  | —       X.XX  | —               X.XX  | —
```

### Section 2 — Classement modèles par dataset

Heatmap AUROC normalisé — identifie le meilleur modèle par dataset et le modèle le plus robuste cross-dataset.

### Section 3 — Impact du ratio normal sur l'AUROC

Scatter plot AUROC moyen vs ratio normal par dataset (0.10 / 0.50 / 0.90) × 6 modèles. Quantifie la sensibilité de chaque modèle au ratio normal.

### Section 4 — Impact de la dimensionnalité sur l'AUROC

Scatter plot AUROC moyen vs input_dim (4 / 9 / 13) × 6 modèles. Identifie les modèles robustes à la haute dimensionnalité.

### Section 5 — RAM cross-dataset

Barplot `ram_peak_bytes` : 6 modèles × 3 datasets. Ligne de référence 64 Ko.
Tableau : quels modèles restent sous 64 Ko sur tous les datasets ?

### Section 6 — Recommandations embarquées STM32N6

Tableau de recommandations pour le portage :

| Critère | Meilleur modèle | Justification |
|---------|-----------------|---------------|
| AUROC global moyen | — | — |
| RAM minimale | Mahalanobis (CWRU) | Cov 9×9 = 324 B @ FP32 |
| Robustesse faible ratio normal | — | — |
| Robustesse haute dimensionnalité | — | — |
| Meilleur trade-off AUROC/RAM | — | — |

### Section 7 — Position par rapport au Triple Gap

- **Gap 1** (données industrielles réelles) : CWRU + Pronostia + Equipment Monitoring validés sur 18 expériences ✅
- **Gap 2** (< 100 Ko RAM) : RAM mesurée pour 6 modèles × 3 datasets — tableau conformité 64 Ko
- **Gap 3** (INT8 pendant entraînement) : non adressé dans la Phase Anomaly Detection (Sprint 16+)

---

## Figures à sauvegarder

```
notebooks/figures/anomaly_detection/
├── summary_auroc_crossdataset.png          # heatmap principal
├── summary_auroc_vs_normal_ratio.png       # scatter ratio normal
├── summary_auroc_vs_input_dim.png          # scatter dimensionnalité
├── summary_ram_crossdataset.png            # RAM comparaison
└── summary_triple_gap_phase_ad.png         # positionnement Triple Gap
```

---

## Critères d'acceptation

- [ ] Notebook exécutable end-to-end sans erreur
- [ ] Tableau AUROC cross-dataset complet (6 modèles × 3 datasets minimum)
- [ ] Heatmap présente et sauvegardée
- [ ] Section "Recommandations embarquées" rédigée (tableau complet)
- [ ] Section "Triple Gap" référence les numéros d'expériences précis (exp_143–160)
- [ ] Figures sauvegardées dans `notebooks/figures/anomaly_detection/`

## Statut

⬜ À faire (après S19-05)
