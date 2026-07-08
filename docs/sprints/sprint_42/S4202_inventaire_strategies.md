# S4202 — Inventaire de référence des stratégies de quantification

| Champ | Valeur |
|-------|--------|
| **Sprint** | 42 |
| **Priorité** | 🔴 Critique — source de vérité textuelle des figures, slides et manuscrit |
| **Durée estimée** | ~3h (pur doc, parallélisable avec S4201) |
| **Statut** | ✅ Implémenté (7 juillet 2026) |
| **Fichier cible** | `docs/context/quantization_strategies.md` |

## Objectif

Un **document unique** qui décrit chaque stratégie de quantification utilisée dans le projet : ce qu'elle
fait aux données, où elle vit dans le code, ce qu'elle a donné expérimentalement, et le message à en tirer.
C'est le texte de référence que les slides (S4203–S4206) et le manuscrit citeront — aujourd'hui cette
connaissance est éclatée dans les statuts de sprints 22/28/29/34/36/39/40.

## Structure imposée du document

Pour **chaque** stratégie (FP32 réf, INT8 QAT PC, INT8 PTQ legacy board, INT8 v2 per-tensor/per-channel,
Q15, HDC int16-AM), une section avec les rubriques fixes :

1. **Principe** — la transformation mathématique (formule : scale/zero-point affine, grille Q15
   `q = round(x·32767/max|x|)`, fake-quant STE, etc.) et **ce qui change concrètement pour les données**
   (poids, activations, accumulateur, moment de l'application).
2. **Où dans le code** — fichiers Python + firmware + flags d'export (`export_weights_c.py --…`), avec
   liens relatifs. Ex. INT8 legacy : `firmware/stm32f4_blink/src/ewc_head_int8.c` (`ewc_int8_from_fp32`,
   échelle fixe 1/128, accumulateur int16).
3. **Impact mesuré** — pointeurs vers les expériences (`experiments/exp_S…`) et les chiffres clés **avec
   leur source** (pas de chiffre sans référence de fichier) : métrique, RAM, latence, parité board.
4. **Quand l'utiliser / pièges** — la recommandation issue des sprints (ex. « Q15 pour tenseurs à grande
   dynamique type `sigma_inv_` » ; « PTQ à échelle figée à proscrire, calibrer le scale »).

Puis trois sections transversales :

- **Tableau comparatif synthétique** (stratégie × {format poids, moment, calibration, RAM, latence Cortex-M4,
  métrique préservée ?, validation board}) — le tableau que les slides reprendront tel quel.
- **Chronologie** — quelle stratégie a été introduite à quel sprint et pourquoi (le fil narratif :
  QAT PC ✅ → PTQ board ❌ → diagnostic émulateur → v2 calibré / Q15 ✅).
- **Limites & travaux futurs** — paradoxe latence FPU (pas de chemin entier SIMD/CMSIS-NN, `TODO(dorra)`),
  `mu_` Q15 (piste Sprint 34), QAT exporté vers board.

## Règles

- **Chaque chiffre cité porte sa source** (exp JSON ou doc de sprint). Les mesures non faites (board v2
  Sprint 40) sont indiquées `« à mesurer »`.
- Le doc ne duplique pas les analyses complètes des sprints : il **synthétise et pointe** (liens relatifs
  vers `docs/sprints/…` et notebooks).
- Cohérence de notation avec le manuscrit (chapitres S41) — `TODO(arnaud)` si les symboles divergent.

## Critères d'acceptation

1. Les 6 stratégies documentées avec les 4 rubriques fixes.
2. Tableau comparatif complet, 0 cellule inventée (`« à mesurer »` autorisé).
3. Relecture croisée avec les figures S4203–S4205 : mêmes noms de stratégies, mêmes couleurs référencées.

## Réalisation (7 juillet 2026)

- `docs/context/quantization_strategies.md` créé : 6 stratégies × 4 rubriques (Principe / Où dans le code / Impact mesuré sourcé / Quand l'utiliser–pièges) + tableau comparatif + chronologie S22→S40 + limites (`TODO(dorra)` SIMD, `mu_` Q15, export QAT).
- Chiffres vérifiés dans les JSON avant citation (`exp_S39_ablation`, `exp_S39_quant_sweep/summary.json`, `exp_S34_maha_q15/summary.json`, `exp_S28_PC_ewc_hdc`) ; board v2 S40 = `« à mesurer »`.
- Noms/couleurs de stratégies = clés `STRATEGY_COLORS` de `src/figures/style.py` (cohérence S4203–S4205) ; liens croisés vers les figures pedagogy.
