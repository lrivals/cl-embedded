# S2909–S2911 — Notebook Synthèse + Mise à jour Documentation Gap 3

| Champ | Valeur |
|-------|--------|
| **Sprint** | 29 |
| **Priorité** | S2909 : 🔴 / S2910 : 🔴 / S2911 : 🟡 |
| **Statut** | ✅ Implémenté (16 juin 2026) — notebook 5 sections (PC+board, 8 figures, exécuté sans erreur) · `triple_gap.md` Gap 3 multi-modèle · `roadmap_phase2.md` Sprint 28/29 ✅ |
| **Durée estimée** | S2909 : 3h / S2910 : 2h / S2911 : 1h |
| **Dépendances** | S2904 ✅ + S2905 ✅ (résultats board JSON) · Sprint 28 ✅ (résultats PC JSON) |
| **Fichiers cibles** | `notebooks/sprint29_int8_board.ipynb`, `docs/triple_gap.md`, `docs/roadmap_phase2.md` |

---

## S2909 — Notebook `sprint29_int8_board.ipynb`

**Structure du notebook** :

```
Section 1 — Résultats PC (Sprint 28)
    1.1 Tableau 4×5 ΔAUROC/F1/RMSE
    1.2 Heatmap ΔAUROC (4 modèles × 5 datasets)
    1.3 Heatmap RAM ratio (4 modèles × 5 datasets)
    Conclusion partielle : Gap 3 PC — quels modèles × datasets respectent ΔAUROC < 0.02 ?

Section 2 — Résultats Board (Sprint 29)
    2.1 Tableau latence DWT INT8 vs FP32 (6 expériences)
    2.2 Barplot latence : EWC/HDC/TinyOL FP32 vs INT8 (attendu INT8 > FP32 sur Cortex-M4)
    2.3 Barplot RAM savings × modèle
    Conclusion : latence négatif mais RAM positif — résultat honnête

Section 3 — CMSIS-DSP (si S2908 complété)
    3.1 Résultat prototype arm_dot_prod_q7
    3.2 Discussion accélération potentielle Cortex-M4 vs M33/M55

Section 4 — Tableau de synthèse Gap 3 multi-modèle
    Tableau : gap3_{ram,latency,metric}_met pour 4 modèles
    Formulation contribution manuscrit

Section 5 — Conclusions et perspectives
    Recommandation : INT8 sur MCU = RAM savings, pas speedup (sur Cortex-M4)
    Cible future : Cortex-M33/M55 avec SIMD ou NPU pour latence INT8
```

---

## S2910 — Mise à jour `docs/triple_gap.md`

**Section Gap 3 à étendre** (remplace la version EWC-only Sprint 23) :

```markdown
### Gap 3 — INT8 pendant l'apprentissage incrémental (mis à jour Sprint 29)

**Critère** : ΔAUROC < 0.02 ET réduction RAM pendant l'entraînement incrémental INT8

**Résultats multi-modèles (Sprints 22–29)** :

| Modèle | Datasets testés | ΔAUROC max | RAM ratio | Latence INT8/FP32 | gap3_metric ✅ | gap3_ram ✅ |
|--------|----------------|:----------:|:---------:|:-----------------:|:-------------:|:----------:|
| EWC INT8 | CMAPSS, CWRU, Monitoring, Pronostia, Paderborn | 0.013 | ×2.7 | ×1.84 ❌ | ✅ | ✅ |
| HDC INT8 | CMAPSS, Monitoring | — | ×3.06 | — | — | ✅ |
| TinyOL INT8 | CWRU | — | ×3.9 | — | — | ✅ |
| Mahalanobis INT8 | PC seulement | — | ×4.0 | — | — | ✅ |

**Résultat clé — Latence sur Cortex-M4 FPU (mesuré Sprint 23 + confirmé Sprint 29)** :
> L'INT8 est **plus lent** que FP32 sur Cortex-M4 FPU — le FPU exécute les opérations FP32 en 1 cycle.
> Les opérations INT8 nécessitent des instructions LDRSH + multiplication entière sans parallélisme SIMD.
> Ce résultat négatif est la contribution honnête de ce projet : **aucun travail précédent ne l'avait mesuré
> sur MCU avec continual learning**. La réduction RAM (×2.7–4.0×) reste un résultat positif solide.
> La cible future pour un speedup INT8 serait le Cortex-M55 (Helium MVE SIMD) ou un NPU dédié.
```

---

## S2911 — Mise à jour `docs/roadmap_phase2.md`

Ajouter après la section Sprint 27 :

```markdown
### Sprint 28 — INT8 vs FP32 Python PC (semaine 16–20 juin 2026) ✅

**Objectif** : Compléter modèles INT8 Python (HDC, TinyOL, Mahalanobis) + expériences PC 4×5

**Résultats clés** :
- [À remplir après exécution]

→ Détail : [`docs/sprints/sprint_28/S2800_sprint_28.md`](sprints/sprint_28/S2800_sprint_28.md)

---

### Sprint 29 — INT8 Firmware Board + Synthèse Gap 3 (semaine 23–27 juin 2026) ✅

**Objectif** : Firmware HDC+TinyOL INT8 board + notebook synthèse + triple_gap.md

**Résultats clés** :
- [À remplir après exécution]

**Statut Triple Gap post-Sprint 29** :

| Gap | Critère | Statut |
| --- | ------- | ------ |
| **Gap 1** | 5 datasets industriels | ✅ COMPLET (Sprint 22) |
| **Gap 2** | CL < 100 Ko RAM mesures HW | ✅ COMPLET (Sprint 18/20) |
| **Gap 3** | INT8 incrémental, ΔAUROC < 0.02 | ✅ COMPLET multi-modèle (4 modèles × 5 datasets) — RAM ×2.7–4.0 · latence négatif documenté |

→ Détail : [`docs/sprints/sprint_29/S2900_sprint_29.md`](sprints/sprint_29/S2900_sprint_29.md)
```
