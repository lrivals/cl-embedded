# Stratégies de quantification — inventaire de référence (S4202)

> **Résumé** : ce document est la source de vérité textuelle des six stratégies de
> quantification du projet : ce que chacune fait aux données, où elle vit dans le code,
> ce qu'elle a donné expérimentalement (chaque chiffre porte sa source), et quand
> l'utiliser. Les slides (S4203–S4206) et le manuscrit citent ce document. Il synthétise
> et pointe — les analyses complètes restent dans les docs de sprints 22/28/29/34/36/39/40.

**Nomenclature et couleurs** : les clés `fp32`, `int8_qat`, `int8_ptq_legacy`, `int8_v2`,
`q15`, `int16_am` et leurs couleurs sont définies dans
[`src/figures/style.py`](../../src/figures/style.py) (`STRATEGY_COLORS`) et utilisées à
l'identique dans toutes les figures des catalogues `quantization/*`
([`docs/figures/quantization/`](../figures/quantization/)).

---

## 1. `fp32` — FP32, la référence

**Principe.** Aucune transformation : poids, activations et accumulations en `float32`
IEEE-754 (~7 chiffres significatifs, dynamique ±3,4·10³⁸). C'est le format natif du
Cortex-M4F (FPU simple précision) et la référence contre laquelle toute stratégie
quantifiée est mesurée.

**Où dans le code.**
- Python : [`src/models/ewc/ewc_mlp_multiclass.py`](../../src/models/ewc/ewc_mlp_multiclass.py) (tête board 5→32→16→2) et les autres modèles de `src/models/`.
- Émulateur : [`src/utils/int8_c_emulation.py`](../../src/utils/int8_c_emulation.py) `forward_fp32` (référence des ablations S39).
- Firmware : [`firmware/stm32f4_blink/src/ewc_head.c`](../../firmware/stm32f4_blink/src/ewc_head.c) (forward + SGD FP32 sur FPU).
- Export : `scripts/export_weights_c.py --ewc-head` → `inc/model_weights_ewc.h`.

**Impact mesuré.** Baselines F1 (condition 5feat, tête EWC) : cmapss 0,448 · cwru 0,996 ·
monitoring 0,919 · pronostia 0,962 · paderborn 0,800
([`experiments/exp_S39_ablation/*.json`](../../experiments/exp_S39_ablation/), champ `f1_fp32`).
RAM poids tête EWC : 3 016 B ([`exp_S39_quant_sweep/summary.json`](../../experiments/exp_S39_quant_sweep/summary.json)).
Latence board : EWC inférence ~50 µs, +MAJ CL 239–340 µs ≪ 100 ms (Gap 2 ✅, Sprint 36,
[`exp_S36_summary.json`](../../experiments/exp_S36_summary.json)).

**Quand l'utiliser / pièges.** Toujours comme référence et comme mode par défaut sur la
NUCLEO-F439ZI (FP32 n'y est pas une contrainte, cf.
[`hardware_constraints.md`](hardware_constraints.md)). Piège : conclure d'un échec INT8
que « la quantification ne marche pas » — Sprint 39 a montré que c'est le *schéma* qui
comptait, pas le principe.

---

## 2. `int8_qat` — INT8 QAT (fake-quant PC)

**Principe.** *Quantization-Aware Training* : pendant l'entraînement, un nœud
`quant → déquant` est inséré dans le forward — les poids/activations restent en float,
mais « arrondis » sur la grille INT8 (`x̃ = s·round(x/s)`, scale `s` appris/observé,
per-channel pour les poids). Le gradient traverse l'arrondi par le *straight-through
estimator* (STE, gradient ≈ identité) : **le modèle apprend en voyant l'erreur de
quantification** et adapte ses poids en conséquence. Rien ne change pour l'accumulateur
(float) — c'est une *simulation* de l'INT8, pas un chemin entier.
Figure mécanisme : [`qat_vs_ptq.png`](../figures/quantization/pedagogy/qat_vs_ptq.png),
[`fakequant_forward.png`](../figures/quantization/pedagogy/fakequant_forward.png).

**Où dans le code.**
- [`src/models/ewc/ewc_mlp_int8.py`](../../src/models/ewc/ewc_mlp_int8.py) : `FakeQuantize` + `HistogramObserver` (activations), `PerChannelMinMaxObserver` symétrique (poids), STE.
- Bench PC : `scripts/` Sprint 28 → `experiments/exp_S28_PC_*`.

**Impact mesuré.** Métrique **préservée** : EWC Δ ≤ 0,006 sur les 5 datasets, HDC Δ = 0
([`experiments/exp_S28_PC_ewc_hdc/`](../../experiments/exp_S28_PC_ewc_hdc/), champ
`delta_metric` ; bilan [`S2808`](../sprints/sprint_28/)). Gap 3 RAM ✅ ratio 4,00× (EWC).
TinyOL : fake-quant légèrement *régularisant* (amélioration |Δ| > 0,02 sur CMAPSS/CWRU,
même source).

**Quand l'utiliser / pièges.** La voie de référence quand on contrôle l'entraînement et
qu'on veut préserver la métrique. Pièges : (1) les scales appris ne sont **pas exportés
vers le firmware** à ce jour — le board reçoit une PTQ distincte, d'où l'écart
QAT PC ✅ / PTQ board ❌ du Sprint 36 ; l'export QAT→board est un travail futur (§ Limites) ;
(2) le fake-quant ne mesure ni latence ni RAM réelles (tout reste float en mémoire).

---

## 3. `int8_ptq_legacy` — INT8 PTQ legacy board (échelle figée 1/128)

**Principe.** *Post-Training Quantization* naïve, appliquée au flash : les poids FP32
entraînés sont convertis **après coup** avec une échelle **figée** `s = 1/128` :
`q = SAT8(trunc(w·128))` — tout poids `|w| > 127/128 ≈ 0,992` est **clampé** à ±127
(cf. [`mapping_affine_int8.png`](../figures/quantization/pedagogy/mapping_affine_int8.png)).
Activations Q7 même échelle (ReLU clampé à ~1), **accumulateur int16** avec wrap
(overflow latent), déquant `(acc >> 7)/128`. Aucune calibration : le modèle découvre
l'erreur au déploiement.

**Où dans le code.**
- Firmware : [`firmware/stm32f4_blink/src/ewc_head_int8.c`](../../firmware/stm32f4_blink/src/ewc_head_int8.c) (`ewc_int8_from_fp32`, chemin 0x40).
- Émulateur bit-exact : [`src/utils/int8_c_emulation.py`](../../src/utils/int8_c_emulation.py) `QuantConfig.legacy_c()` (S3902).

**Impact mesuré.** **Destructeur sur certains datasets** — F1 émulé (bit-exact C,
[`exp_S39_ablation/`](../../experiments/exp_S39_ablation/)) : pronostia 0,962 → **0,066**,
monitoring 0,919 → **0,118**, cmapss 0,448 → 0,227 ; cwru 0,996 → 0,929 et paderborn 0,800 → 0,800
s'en sortent (dynamique des poids favorable). Board réel : F1 INT8 0,07–0,15 vs ≈ 0,92
FP32 (Sprint 36, [`S3610_int8_fp32_board.md`](../sprints/sprint_36/S3610_int8_fp32_board.md),
`exp_S36_board_*_int8_*`) ; EWC AUROC 0,25 vs 0,63 (Sprint 29,
[`exp_S29_board_int8/`](../../experiments/exp_S29_board_int8/)). RAM ✅ ×4,0 mais latence
**non** améliorée (déquant FPU, pas de chemin SIMD — Sprint 29).
**Diagnostic S39** ([`S3904`](../sprints/sprint_39/S3904_ablation_perte_f1.md)) : la cause
dominante est **l'échelle figée non calibrée**, pas l'overflow int16 (`fix_acc32` ne
récupère presque rien : monitoring 0,118 → 0,042 ; `per_tensor_calib` récupère tout :
0,118 → 0,920).

**Quand l'utiliser / pièges.** **À proscrire** — conservé uniquement comme baseline
pédagogique et pour la reproductibilité des Sprints 22–36. Piège classique : imputer la
perte au « INT8 » en général alors qu'elle vient du scale figé (cf.
[`erreur_quantification_poids.png`](../figures/quantization/pedagogy/erreur_quantification_poids.png)).

---

## 4. `int8_v2` — INT8 v2 per-tensor / per-channel calibré

**Principe.** PTQ **calibrée** : échelle par tenseur (`s = max|W|/127`) ou par canal de
sortie (`s_j = max|W[j,:]|/127`), quantification arrondie (`round`, pas `trunc`),
activations 8-bit à scale calibré sur un lot représentatif, **accumulateur int32** (pas
de wrap), déquant exact `acc·s_w·s_a` sur FPU. C'est le miroir PTQ du schéma
per-channel du QAT PC.

**Où dans le code.**
- Firmware : [`firmware/stm32f4_blink/src/ewc_head_int8_v2.c`](../../firmware/stm32f4_blink/src/ewc_head_int8_v2.c) / [`inc/ewc_head_int8_v2.h`](../../firmware/stm32f4_blink/inc/ewc_head_int8_v2.h) (`ewc_int8_v2_from_fp32_calib` ; variantes de build `-DEWC_INT8_Q15` int16/acc int64, `-DEWC_INT8_MIXED` poids int8 + act int16).
- Export : `scripts/export_weights_c.py --int8-v2` → `inc/ewc_head_int8_v2_weights.h` (réutilise `_weight_scales`/`_quant_weight`/`calibrate_activations` de l'émulateur = parité par construction) ; `--int8-v2-test-vectors` → golden Unity ([`tests/test_ewc_int8_v2.c`](../../firmware/stm32f4_blink/tests/test_ewc_int8_v2.c)).
- Émulateur : `QuantConfig.per_tensor_calib()` / `QuantConfig.per_channel_int8()`.

**Impact mesuré.** **Récupère la métrique FP32** en 8 bits : F1 per-channel = 0,448/0,995/0,919/0,943/0,800
vs fp32 0,448/0,996/0,919/0,962/0,800 (cmapss/cwru/monitoring/pronostia/paderborn,
[`exp_S39_ablation/`](../../experiments/exp_S39_ablation/)). Sweep 4 modèles × 5 datasets :
ewc×cmapss legacy 0,350 → per-channel 0,4527 ≈ fp32 0,4532, RAM poids 3 016 → 754 B (×4)
([`exp_S39_quant_sweep/summary.json`](../../experiments/exp_S39_quant_sweep/summary.json)).
Board (frozen, 300 échantillons streamés) : [`exp_S39_board/`](../../experiments/exp_S39_board/)
`results_per_channel_int8_{cmapss,pronostia}.json` ; la **validation board complète**
(online, campagne S40) est **« à mesurer »** ([`S4002`](../sprints/sprint_40/S4002_board_validation_v2.md)).

**Quand l'utiliser / pièges.** Le chemin INT8 board recommandé pour les têtes
neuronales à dynamique modérée. Pièges : (1) la calibration d'activations exige un lot
représentatif (l'export par défaut utilise un lot synthétique seedé — calibrer sur de
vraies features quand disponible) ; (2) latence : le MAC reste déquantifié en FP32 sur
FPU, le gain BOPs ((32/8)² = 16, Sprint 33) ne se matérialise pas sans noyau entier
SIMD/CMSIS-NN (§ Limites).

---

## 5. `q15` — Q15 (grille 16 bits)

**Principe.** Quantification sur **65 536 niveaux** : `q = round(x·32767/max|x|)` stocké
`int16`, scale par-tenseur. Le pas de grille est 258× plus fin qu'en INT8 sur la même
dynamique — décisif pour les tenseurs à **grande dynamique** comme `sigma_inv_`
(Paderborn : coefficients de ~1,5 à 6,6·10⁵, pas INT8 ≈ 5 227 → 11/25 coefficients
écrasés à 0 ; pas Q15 ≈ 20,3 — cf.
[`grilles_int8_vs_q15.png`](../figures/quantization/pedagogy/grilles_int8_vs_q15.png) et
[`dynamique_sigma_inv.png`](../figures/quantization/pedagogy/dynamique_sigma_inv.png),
tenseur réel). RAM : ÷2 vs FP32 (contre ÷4 en INT8).

**Où dans le code.**
- Python : [`src/models/unsupervised/`](../../src/models/unsupervised/) `MahalanobisDetectorInt8` `quant: q15` (`sigma_inv_` int16 Q15, `mu_` reste INT8 affine — S3405).
- Firmware : [`firmware/stm32f4_blink/src/mahalanobis_q15.c`](../../firmware/stm32f4_blink/src/mahalanobis_q15.c) (déquant → distance FP32 sur FPU = parité bit-à-bit Python ; flag protocole `0xF0`) ; côté EWC, build `-DEWC_INT8_Q15` du kernel v2.
- Export : `scripts/export_weights_c.py --maha-q15` → `inc/mahalanobis_q15_weights.h`.
- Émulateur : `QuantConfig.q15()`.

**Impact mesuré.** **Récupération Mahalanobis** ([`exp_S34_maha_q15/summary.json`](../../experiments/exp_S34_maha_q15/summary.json)) :
Pronostia AUROC 0,860 (fp32) → 0,747 (int8, Δ = −0,113) → **0,873 (q15, Δ = +0,013 ✅)** ;
corrélation de rang au FP32 : Q15 > INT8 sur les 5 datasets (Pronostia 0,985 vs 0,649,
Paderborn 0,921 vs 0,827, CWRU 0,536 vs 0,409). Board réel (S3408) : **parité exacte
300/300**, latence DWT P50 = 5 µs, `.bss` +80 B. Tête EWC : Q15 = FP32 aux arrondis près
(F1 0,962 pronostia, [`exp_S39_ablation/`](../../experiments/exp_S39_ablation/)) mais
n'apporte ~rien de plus que l'INT8 calibré sur ces poids à faible dynamique.
Nuance honnête (S34) : sur très grande dynamique, l'erreur *absolue* de score Q15 peut
dépasser INT8 — non par infidélité de Σ⁻¹ (reconstruite ~200× mieux) mais parce que
`mu_` reste INT8 et que son erreur est amplifiée par les grands coefficients que Q15
préserve (INT8 les écrase → distances collapsées).

**Quand l'utiliser / pièges.** Recommandé pour les **tenseurs à grande dynamique**
(`sigma_inv_` Mahalanobis) — réponse au `TODO(arnaud)` S2805. Pièges : gain RAM ÷2
seulement ; envisager `mu_` en Q15 aussi (piste S34, § Limites).

---

## 6. `int16_am` — HDC, mémoire associative int16

**Principe.** Spécifique au HDC : les hypervecteurs restent ±1 (int8), mais la **mémoire
associative** (AM = sommes de bundling ∈ [−N, +N]) est stockée `int16` avec saturation
±32767 (`SAT16`), au lieu de int32. La requête est un produit scalaire int8×int16
accumulé int32 → argmax. Ce n'est pas une quantification par scale : c'est un
**rétrécissement du type de l'accumulateur de classe**, sans perte tant que les comptes
ne saturent pas.

**Où dans le code.**
- Python : [`src/models/hdc/hdc_int8.py`](../../src/models/hdc/hdc_int8.py).
- Firmware : [`firmware/stm32f4_blink/src/hdc_int8.c`](../../firmware/stm32f4_blink/src/hdc_int8.c) / [`inc/hdc_int8.h`](../../firmware/stm32f4_blink/inc/hdc_int8.h).

**Impact mesuré.** Métrique **strictement préservée** : Δ = 0 sur tous les datasets
mesurés (Sprint 28, [`exp_S28_PC_ewc_hdc/`](../../experiments/exp_S28_PC_ewc_hdc/)) ;
ratio RAM **2,33×** (int16-AM, borne basse des 2,33–4,00× de la grille Gap 3). Board :
validé Sprint 29 ([`exp_S29_board_int8/`](../../experiments/exp_S29_board_int8/)),
latence HDC ~2 095 µs dominée par l'encodage, pas par l'AM.

**Quand l'utiliser / pièges.** Toujours (aucun coût métrique mesuré). Pièges :
saturation possible de l'AM en online learning très long (borné par SAT16) ; le gain
est plafonné à ×2 sur l'AM seule — la projection reste le poste RAM/latence dominant.

---

## Tableau comparatif synthétique

| Stratégie | Format poids | Moment | Calibration | RAM (vs FP32) | Latence Cortex-M4 | Métrique préservée ? | Validation board |
|---|---|---|---|---|---|---|---|
| `fp32` | float32 | — | — | 1× (réf.) | réf. (FPU) | réf. | ✅ Sprints 20–38 |
| `int8_qat` | int8 simulé (float) | pendant l'entraînement | scales appris (STE) | ×4 (théorique) | N/A (PC) | ✅ Δ ≤ 0,006 (`exp_S28_PC_*`) | ❌ scales non exportés |
| `int8_ptq_legacy` | int8, s = 1/128 figé | après (au flash) | aucune | ×4 (`exp_S39_quant_sweep`) | ≈ FP32 (déquant FPU, S29) | ❌ F1 0,96 → 0,07 (pronostia, `exp_S39_ablation`) | ✅ mesuré… et invalidé (S29/S36) |
| `int8_v2` | int8 per-tensor/-channel | après, calibré | scale poids + activations | ×4 (754/3016 B) | ≈ FP32 (déquant FPU) | ✅ ≈ FP32 (`exp_S39_ablation`) | 🟡 frozen `exp_S39_board` ; campagne S40 **« à mesurer »** |
| `q15` | int16, s = max/32767 | après, calibré | scale par-tenseur | ×2 | ≈ FP32 (5 µs Maha, S3408) | ✅ AUROC récupérée (`exp_S34_maha_q15`) | ✅ parité exacte 300/300 (S3408) |
| `int16_am` | AM int16 (HV int8) | pendant (bundling saturé) | aucune | ×2,33 (`exp_S28_PC_ewc_hdc`) | ≈ FP32 (AM ≪ encodage) | ✅ Δ = 0 (S28) | ✅ Sprint 29 |

---

## Chronologie — le fil narratif

| Sprint | Événement |
|---|---|
| 22 | Gap 3 « comblé » une première fois : EWC INT8 + HDC INT8 Python + C. |
| 28 | Benchmark PC 4 × 5 : **QAT PC ✅** (EWC Δ ≤ 0,006, HDC Δ = 0) mais Mahalanobis INT8 dégradé (grande dynamique `sigma_inv_`) → fallback Q15 pressenti (`TODO(arnaud)` S2805). |
| 29 | Portage board INT8 : RAM ×2,70–4,00 ✅ mais latence non accélérée (FPU) et Maha INT8 reproduit la dégradation ; EWC PTQ board déjà suspect (AUROC 0,25). |
| 34 | **Q15 ✅** pour Mahalanobis : Pronostia ΔAUROC −0,113 → +0,013, parité board exacte. |
| 36 | Comparaison appariée PC↔board : **PTQ legacy board ❌** formellement établi (F1 0,07–0,15 vs 0,92) — distinct du QAT PC. |
| 39 | **Diagnostic par émulateur bit-exact** : la cause est l'échelle figée 1/128 (pas l'overflow int16) ; `per_tensor_calib` récupère +0,88 F1 (monitoring). Kernel **v2 calibré ✅** (émulé + frozen board). |
| 40 | Validation board v2 complète + article — mesures **« à mesurer »** (en cours). |

---

## Limites & travaux futurs

- **Paradoxe latence FPU** : l'INT8 réduit la RAM ×4 mais pas la latence — le MAC est
  déquantifié en FP32 sur FPU, sans chemin entier SIMD (`SMLAD`) ni CMSIS-NN. Spec :
  [`S3910_simd_cmsis_spec.md`](../sprints/sprint_39/S3910_simd_cmsis_spec.md), `TODO(dorra)`.
- **`mu_` en Q15** : dans le schéma Q15 Mahalanobis, `mu_` reste INT8 affine ; son erreur,
  amplifiée par les grands coefficients de Σ⁻¹, domine l'erreur absolue de score sur
  Paderborn (piste Sprint 34, [`S3405`](../sprints/sprint_34/)).
- **QAT exporté vers board** : les scales appris du QAT PC ne sont pas consommés par le
  firmware ; l'export QAT → `int8_v2` fermerait l'écart PC/board par construction
  (piste S36/S39).
- **Énergie** : le gain BOPs ((32/8)² = 16) est théorique tant que la sonde LPM01A n'a
  pas tourné — champs `« à mesurer »` ([`exp_S33_energy/`](../../experiments/exp_S33_energy/), Sprint 33).

> Cohérence de notation avec le manuscrit (chapitres S41) : `TODO(arnaud)` — valider que
> les symboles (s, z, q, Q15) du manuscrit suivent ce document.
