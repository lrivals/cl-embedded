# Fiche de cadrage — Ch. 7 Gap 3 : quantification pendant l'entraînement incrémental (~3.5 p., cible md `07_gap3_quantification.md`)

~~⚠️ **Chapitre le plus dépendant des travaux en cours (Sprints 39–40)** — placeholders
systématiques, résolution S4110.~~
✅ **S4110 (30 juillet 2026) : tous les placeholders sont résolus.** S39 et S40 sont terminés, et
quatre sprints postérieurs au cadrage (S46 moments, S47 profondeur, S48 sub-INT8 carte, S50
breakdown latence) enrichissent le chapitre — voir « Messages ajoutés » ci-dessous.

## Messages clés (narratif honnête en 4 temps)

1. **RAM : gain démontré** — INT8 réduit l'empreinte des modèles ×2.33–4.0 selon modèle/dataset,
   PC (S28, 20 cellules) et board (S29, 20 cellules dont 18 streamées).
2. **QAT PC : métrique préservée** — fake-quant pendant l'entraînement incrémental, Δ≤0.006 pour
   EWC, Δ=0 HDC (S28). Cas dégradés identifiés honnêtement : Mahalanobis INT8 (−0.236/−0.238,
   grande dynamique `sigma_inv_`) → renvoi perspective Q15 (S34).
3. **PTQ board historique : dégradation identifiée puis expliquée** — F1 INT8 0.08–0.15 vs
   FP32 0.9164 (pronostia) / 0.9194 (monitoring) (S36) ; diagnostic S39 (émulateur bit-exact).
   **Correction S4110** : l'ablation mesurée **n'appuie pas les « 3 causes concomitantes »** du
   cadrage initial — **une seule domine, la calibration d'échelle** (+0.878 monitoring /
   +0.880 pronostia), l'accumulateur int32 seul n'apporte rien (pronostia +0.0004) et **dégrade**
   monitoring (−0.076) tant que l'échelle reste fausse. Le texte du chapitre a été reformulé en
   conséquence : une cause dominante, deux secondaires.
4. **Correction : kernel v2 calibré — ✅ VALIDÉ CARTE (S40)**, la réserve est levée.
   Échelle d'ablation (`exp_S39_ablation/`) : monitoring 0.1178 → 0.0419 → **0.9201** → 0.9187 →
   0.9194 (Q15) ; pronostia 0.0663 → 0.0667 → **0.9462** → 0.9426 → 0.9616 (Q15).
   Carte réelle (`exp_S40_board_v2/results_per_channel_*_frozen.json`) : **F1 0.9173 (monitoring) /
   0.8995 (pronostia)**, **parité carte↔émulateur 1.000** (0 désaccord), latence 65/68 µs,
   RAM poids 2 688→672 o et 2 816→704 o (**÷4**).
5. **Latence : pas de gain INT8 sur Cortex-M4 FPU — désormais CHIFFRÉ (S50)** : MAC entier
   6 785 cycles (aucun gain vs FPU), **requant FP32→INT8 3 777 cycles = surcoût dominant**
   (`lroundf`), déquant 200 cycles → **74 µs INT8 vs 48–50 µs FP32 = +24 à +26 µs (~+50 %)**.
   Le gain INT8 est un gain *RAM*, pas vitesse, sans chemin SIMD (S2908 bloqué → perspective CMSIS-NN).

## Messages ajoutés en S4110 (sprints postérieurs au cadrage)

- **Nuance RAM ÷4 (S49)** — le ÷4 porte sur la **RAM des poids**, pas sur la RAM système :
   `exp_S49_ram` donne 105 300 B (FP32) vs 105 324 B (INT8) sur EWC×monitoring, **ratio 1.0002**.
   Distinction indispensable pour ne pas surestimer le résultat.
- **Le *moment* de la quantification (S46)** — QAT / PTQ / les-deux à modèle+seed fixés ; carte :
   F1 `both` **0.9213 / 0.9072**, parité 1.000, A/B vs PTQ calibrée seule **+0.004 / +0.008**.
   Constat honnête : **la calibration récupère l'essentiel, le QAT n'ajoute pas de gain décisif**.
- **La *profondeur* sub-INT8 (S47 PC + S48 carte)** — monitoring tient jusqu'au binaire
   (Δ=−0.0117), pronostia casse au binaire (−0.0275) mais tient en ternaire (−0.0153) ; le
   per-channel repousse la falaise (pronostia int2 : −0.046 → −0.009). **Nœud d'honnêteté mesuré** :
   le gain RAM sub-INT8 est **conditionnel au bit-packing** (sans packing, `.bss` invariant — même
   conteneur `int8_t`) ; packé, il économise 336–604 B ; le dépacking coûte **≈ +55 µs**
   (67 → 123 µs), sans menacer Gap 2. Parité 1.000 sur les 12 cellules.

## Sources de chiffres (chemins vérifiés)

| Donnée | Source |
|---|---|
| QAT PC 4×5 (20 JSON) | `experiments/exp_S28_PC_ewc_hdc/`, `exp_S28_PC_tinyol_maha/` |
| Board INT8 5→20 couples, ratios RAM, latence | `experiments/exp_S29_board_int8/` |
| PTQ board F1 0.08–0.15, accord INT8↔FP32 | `experiments/exp_S36_board_{frozen,online}_int8_*/` + `exp_S36_summary.json` (clés `board_*_int8`) |
| Ablation des causes | `experiments/exp_S39_ablation/{cmapss,cwru,monitoring,paderborn,pronostia}.json` — **confirmé S4110** (champ `ladder`) |
| Sweep quantifications intermédiaires | `experiments/exp_S39_quant_sweep/summary.json` — **confirmé S4110** |
| Comparaison à conditions identiques | `experiments/exp_S39_matched/matched_ewc_{cmapss,pronostia}_*.json` — **confirmé S4110** |
| Validation board v2 | `experiments/exp_S40_board_v2/results_per_channel_{monitoring,pronostia}_{frozen,online}.json` — **✅ produite et mesurée carte** |
| Firmware v2 | `firmware/stm32f4_blink/src/ewc_head_int8_v2.c` + tests Unity `test_ewc_int8_v2.c` |
| **Moment de quantification (S46)** | `experiments/exp_S46_board/{monitoring,pronostia}_both.json` + `exp_S46_ewc/` (PC) |
| **Profondeur sub-INT8 (S47/S48)** | `experiments/exp_S47_depth/*.json` (PC, 28 cellules) + `exp_S48_summary.json` (carte, 12 cellules) |
| **RAM totale FP32 vs INT8 (S49)** | `experiments/exp_S49_ram/summary.json` (champ `ratio_int8_vs_fp32`) |
| **Breakdown latence INT8 (S50)** | `experiments/exp_S50_int8_latency/ewc.json` (champ `segments` : dequant/mac/requant) |
| **Énergie** | `experiments/exp_S50_energy/summary.json` — **`"à mesurer"`, sonde LPM01A non posée** (ne rien avancer) |

## Figures prévues (S4109)

- Base existante `docs/figures/sprint39_int8_intermediate/` : `ablation_factors.png`,
  `tradeoff_pareto.png`, `heatmap_perchannel.png` vs `heatmap_legacy.png` — en retenir 2.
- 1 figure ratios RAM INT8/FP32 (S28/S29).

## Refs bib

`Ravaglia2021` (quantifie le buffer, pas l'entraînement — positionnement clé), **à ajouter S4103** :
`Jacob2018` (quantification inference/entraînement affine), `Krishnamoorthi2018` (whitepaper QAT/PTQ).

## Glossaire touché

PTQ, QAT, fake-quant, Q15, per-channel/per-tensor, BOPs (si mentionné) — entrées à créer S4104.

## Points ouverts — **résolus en S4110 (30 juillet 2026)**

- ~~**Formulation du claim final Gap 3** : option A ou option B selon l'issue S39/S40~~ →
  **option A retenue**, car S40 a bien validé le noyau v2 sur carte réelle. Formulation écrite
  (§7.8 du md) : *Gap 3 comblé côté PC **et** transposition embarquée mesurée* (F1 0.90–0.92,
  parité 1.000, RAM poids ÷4), avec deux nuances revendiquées — le gain RAM est à l'échelle du
  modèle et non du système (S49), et le gain sub-INT8 n'existe qu'avec bit-packing (S48) —
  **l'arbitrage énergétique restant le seul point réellement ouvert**.
- ~~Q15 Mahalanobis (S34) : 1 phrase de renvoi~~ → **fait** (§7.2 du md, renvoi ch. 8).

## Structure finale du chapitre (S4110)

Le chapitre est passé de 5 à 8 sections pour absorber les axes postérieurs au cadrage :
7.1 RAM (+ nuance système) · 7.2 QAT PC · 7.3 PTQ historique (diagnostic corrigé) ·
7.4 noyau v2 validé carte · **7.5 moment (S46)** · **7.6 profondeur (S47/S48)** ·
7.7 paradoxe de latence chiffré (S50) · 7.8 formulation du claim.

⚠️ **Budget** : le cadrage prévoyait ~3.5 p. ; l'ajout de 7.5 et 7.6 pousse au-delà. À arbitrer au
comptage de pages (S4110 reliquat) — candidats à la compression ou au renvoi en annexe : 7.6
(profondeur) puis 7.5 (moment), qui sont les deux axes les moins centraux au claim Gap 3.
