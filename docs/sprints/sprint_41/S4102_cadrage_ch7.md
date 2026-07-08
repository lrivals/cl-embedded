# Fiche de cadrage — Ch. 7 Gap 3 : quantification pendant l'entraînement incrémental (~3.5 p., cible md `07_gap3_quantification.md`)

⚠️ **Chapitre le plus dépendant des travaux en cours (Sprints 39–40)** — rédiger en dernier au
sein de S4107, placeholders systématiques, résolution S4110.

## Messages clés (narratif honnête en 4 temps)

1. **RAM : gain démontré** — INT8 réduit l'empreinte des modèles ×2.33–4.0 selon modèle/dataset,
   PC (S28, 20 cellules) et board (S29, 20 cellules dont 18 streamées).
2. **QAT PC : métrique préservée** — fake-quant pendant l'entraînement incrémental, Δ≤0.006 pour
   EWC, Δ=0 HDC (S28). Cas dégradés identifiés honnêtement : Mahalanobis INT8 (−0.236/−0.238,
   grande dynamique `sigma_inv_`) → renvoi perspective Q15 (S34).
3. **PTQ board historique : dégradation identifiée puis expliquée** — F1 INT8 0.07–0.15 vs
   FP32 ≈0.92 (S36) ; diagnostic S39 (émulateur bit-exact) : accumulateur int16, échelle fixe 1/128
   non calibrée, PTQ one-shot sans recalibration.
4. **Correction : kernel v2 per-channel (S39, `[à confirmer — S40 board]`)** — ablation chiffrée des
   3 facteurs (`exp_S39_ablation/`), sweep de configs intermédiaires (`exp_S39_quant_sweep/`),
   comparaison à conditions identiques (`exp_S39_matched/` : legacy_c vs per_channel_int8 vs q15
   vs mixed). Conclusion attendue : la perte venait de l'implémentation PTQ, pas du principe INT8.
   Validation board v2 = Sprint 40 (S4002) `[à confirmer]`.
5. **Latence : pas de gain INT8 sur Cortex-M4 FPU** (déquantification FP32 dans la boucle) —
   résultat contre-intuitif assumé, mesuré S29 ; le gain INT8 est un gain *RAM*, pas vitesse,
   sans chemin SIMD (S2908 bloqué → perspective CMSIS-NN).

## Sources de chiffres (chemins vérifiés)

| Donnée | Source |
|---|---|
| QAT PC 4×5 (20 JSON) | `experiments/exp_S28_PC_ewc_hdc/`, `exp_S28_PC_tinyol_maha/` |
| Board INT8 5→20 couples, ratios RAM, latence | `experiments/exp_S29_board_int8/` |
| PTQ board F1 0.07–0.15, accord INT8↔FP32 | `experiments/exp_S36_board_{frozen,online}_int8_*/` + `exp_S36_summary.json` (clés `board_*_int8`) |
| Ablation des causes | `experiments/exp_S39_ablation/{cmapss,cwru,monitoring,paderborn,pronostia}.json` `[à confirmer]` |
| Sweep quantifications intermédiaires | `experiments/exp_S39_quant_sweep/summary.json` `[à confirmer]` |
| Comparaison à conditions identiques | `experiments/exp_S39_matched/matched_ewc_{cmapss,pronostia}_*.json` `[à confirmer]` |
| Validation board v2 | Sprint 40 S4002 — `[à confirmer — exp non encore produite]` |
| Firmware v2 | `firmware/stm32f4_blink/src/ewc_head_int8_v2.c` + tests Unity `test_ewc_int8_v2.c` |

## Figures prévues (S4109)

- Base existante `docs/figures/sprint39_int8_intermediate/` : `ablation_factors.png`,
  `tradeoff_pareto.png`, `heatmap_perchannel.png` vs `heatmap_legacy.png` — en retenir 2.
- 1 figure ratios RAM INT8/FP32 (S28/S29).

## Refs bib

`Ravaglia2021` (quantifie le buffer, pas l'entraînement — positionnement clé), **à ajouter S4103** :
`Jacob2018` (quantification inference/entraînement affine), `Krishnamoorthi2018` (whitepaper QAT/PTQ).

## Glossaire touché

PTQ, QAT, fake-quant, Q15, per-channel/per-tensor, BOPs (si mentionné) — entrées à créer S4104.

## Points ouverts

- **Formulation du claim final Gap 3** (validée avec Léonard en S4107/S4110) selon l'issue S39/S40 :
  option A « comblé : QAT PC préservé + kernel v2 corrige le board » (si S40 valide sur board) ;
  option B « comblé côté PC, correction board démontrée en émulation bit-exacte, validation board
  en perspective » (si S40 non terminé à la deadline).
- Q15 Mahalanobis (S34) : décision utilisateur = perspectives ; mais 1 phrase de renvoi ici est
  nécessaire pour le cas Maha dégradé.
