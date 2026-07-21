+# Quantification FP32 → INT8 — lecture pour la présentation

> **Résumé.** Ce document est la **version narrative** du dossier quantification, taillée
> pour la présentation. Il ne réénumère pas les six stratégies à plat : il suit un fil en
> **trois temps** — (1) `fp32`, la référence ; (2) pourquoi il ne faut **pas** comparer
> `int8_qat` et `int8_ptq_legacy` comme s'ils s'affrontaient ; (3) `int8_v2`, la meilleure
> approche rencontrée jusqu'ici et ce qui la distingue. L'inventaire exhaustif des six
> stratégies (avec Q15 et int16_am) reste la source de vérité dans
> [`quantization_strategies.md`](quantization_strategies.md). Chaque chiffre cité ici en
> provient et porte sa source expérimentale — **aucun n'est inventé ni recalculé**.

**Cohérence des couleurs de slides** : les clés `fp32`, `int8_qat`, `int8_ptq_legacy`,
`int8_v2` et leurs couleurs sont définies dans
[`src/figures/style.py`](../../src/figures/style.py) (`STRATEGY_COLORS`) et réutilisées à
l'identique dans les figures de [`docs/figures/quantization/`](../figures/quantization/).

---

## Axe 1 — `fp32`, la référence

**Principe.** Aucune transformation. Poids, activations et accumulations restent en
`float32` IEEE-754 : ~7 chiffres significatifs, dynamique ±3,4·10³⁸. C'est le **format
natif** du Cortex-M4F de la NUCLEO-F439ZI (FPU simple précision matérielle) : le firmware
n'a rien à émuler, chaque multiplication-accumulation tourne directement sur la FPU.

Point important pour cadrer tout le reste : **sur cette board, FP32 n'est pas une
contrainte** (cf. [`hardware_constraints.md`](hardware_constraints.md)). La cible originale
du stage à NPU/64 Ko n'étant pas disponible, on travaille sur une NUCLEO-F439ZI qui a
largement la RAM et la FPU pour tourner en FP32. FP32 est donc à la fois **le mode par
défaut** ET **l'étalon** : la quantification n'est pas une nécessité de survie ici, c'est
un objet d'étude (Gap 3) — on quantifie pour *mesurer* le compromis, pas parce qu'on y est
forcé.

**Où dans le code.**
- Python (modèle de référence) : [`src/models/ewc/ewc_mlp_multiclass.py`](../../src/models/ewc/ewc_mlp_multiclass.py) (tête board 5→32→16→2).
- Émulateur : [`src/utils/int8_c_emulation.py`](../../src/utils/int8_c_emulation.py) `forward_fp32` — c'est la **référence** des ablations du Sprint 39.
- Firmware : [`firmware/stm32f4_blink/src/ewc_head.c`](../../firmware/stm32f4_blink/src/ewc_head.c) (forward + SGD FP32 sur FPU).
- Export : `scripts/export_weights_c.py --ewc-head` → `inc/model_weights_ewc.h`.

**Impact mesuré.** Baselines F1 de la tête EWC (condition 5feat) :

| Dataset | cmapss | cwru | monitoring | pronostia | paderborn |
|---|---|---|---|---|---|
| **F1 fp32** | 0,448 | 0,996 | 0,919 | 0,962 | 0,800 |

Source : [`experiments/exp_S39_ablation/*.json`](../../experiments/exp_S39_ablation/), champ
`f1_fp32`. RAM des poids de la tête EWC : **3 016 B**
([`exp_S39_quant_sweep/summary.json`](../../experiments/exp_S39_quant_sweep/summary.json)).
Latence board : inférence ~50 µs, inférence + mise à jour CL 239–340 µs — **≪ 100 ms**
(Gap 2 ✅, Sprint 36, [`exp_S36_summary.json`](../../experiments/exp_S36_summary.json)).

**Message clé (à garder pour l'axe 2).** Toute stratégie INT8 se juge contre **cette
ligne** de chiffres FP32. Mais « se juger contre la référence » suppose encore qu'on
compare des choses comparables entre elles — c'est précisément ce qui manque quand on met
QAT et PTQ legacy côte à côte.

Figure : [`pipeline/pipeline_fp32.png`](../figures/quantization/pipeline/pipeline_fp32.png).

---

## Axe 2 — Pourquoi comparer `int8_qat` et `int8_ptq_legacy` n'est **pas** pertinent

C'est le cœur du rework. Dans les premières présentations, on mettait volontiers ces deux
INT8 face à face : « le QAT préserve la métrique (Δ ≤ 0,006) mais la PTQ legacy s'effondre
(F1 0,96 → 0,07) ». Formulé ainsi, on croit lire un match — **QAT gagne, PTQ perd**. C'est
une erreur de lecture : les deux ne répondent pas à la même question.

### Rappel express des deux

- **`int8_qat` — INT8 QAT (fake-quant PC).** *Quantization-Aware Training* : pendant
  l'entraînement, un nœud `quant → déquant` est inséré dans le forward. Les poids et
  activations restent en float mais « arrondis » sur la grille INT8 (`x̃ = s·round(x/s)`,
  scale `s` observé, per-channel pour les poids) ; le gradient traverse l'arrondi par le
  *straight-through estimator* (STE). **Le modèle apprend en voyant l'erreur de
  quantification** et s'y adapte. L'accumulateur, lui, reste float : c'est une
  **simulation PC de l'INT8**, pas un chemin entier. Résultat : métrique **préservée**
  (EWC Δ ≤ 0,006, HDC Δ = 0 ;
  [`exp_S28_PC_ewc_hdc/`](../../experiments/exp_S28_PC_ewc_hdc/), champ `delta_metric`).
  Code : [`src/models/ewc/ewc_mlp_int8.py`](../../src/models/ewc/ewc_mlp_int8.py)
  (`FakeQuantize`, `HistogramObserver`, `PerChannelMinMaxObserver`, STE).

- **`int8_ptq_legacy` — INT8 PTQ legacy board (échelle figée 1/128).** *Post-Training
  Quantization* naïve, appliquée **au flash**, sur des poids déjà entraînés : échelle
  **figée** `s = 1/128`, `q = SAT8(trunc(w·128))` — tout poids `|w| > 127/128 ≈ 0,992` est
  **clampé** à ±127. Activations Q7 même échelle, **accumulateur int16** avec *wrap*
  (overflow latent). Aucune calibration : le modèle découvre l'erreur au déploiement.
  Résultat **destructeur** sur certains datasets (F1 émulé bit-exact,
  [`exp_S39_ablation/`](../../experiments/exp_S39_ablation/)) : pronostia 0,962 → **0,066**,
  monitoring 0,919 → **0,118** ; board réel F1 0,07–0,15 vs ≈ 0,92 (Sprint 36,
  [`S3610_int8_fp32_board.md`](../sprints/sprint_36/S3610_int8_fp32_board.md)).
  Code : [`firmware/stm32f4_blink/src/ewc_head_int8.c`](../../firmware/stm32f4_blink/src/ewc_head_int8.c).

### Pourquoi c'est une erreur de catégorie

Les deux ne mesurent tout simplement pas la même chose. Ils diffèrent sur **quatre axes à
la fois** :

| Axe | `int8_qat` | `int8_ptq_legacy` |
|---|---|---|
| **Moment** | pendant l'entraînement | après coup (au flash) |
| **Lieu** | émulé sur PC (tout reste float) | board réelle (chemin entier) |
| **Calibration de l'échelle** | apprise / observée (STE, per-channel) | figée `1/128`, aucune |
| **Ce qui est mesuré** | métrique préservée — **ni** RAM **ni** latence réelles | RAM ×4 + latence + métrique **réelles** board |

Un « match » n'a de sens que si on fait varier **une** dimension à la fois. Ici on en fait
varier quatre. Le QAT tourne en float sur PC et ne mesure ni RAM ni latence réelles ; le
legacy tourne en entier sur la board et mesure tout. Dire « QAT > PTQ » à partir de ça,
c'est comparer un simulateur de vol à un atterrissage raté et conclure que le simulateur
pilote mieux.

### La conclusion honnête

QAT ✅ vs PTQ legacy ❌ **ne dit pas** « le QAT est supérieur à la PTQ ». La perte du legacy
ne vient **pas** du fait d'être *post-training* — elle vient de **l'échelle figée non
calibrée** (`1/128`), qui écrase et clampe les poids hors de la plage.

La preuve est apportée par ablation au Sprint 39, avec l'émulateur bit-exact du kernel C
([`S3904_ablation_perte_f1.md`](../sprints/sprint_39/S3904_ablation_perte_f1.md),
[`exp_S39_ablation/`](../../experiments/exp_S39_ablation/)). On désactive une cause à la fois :

- corriger l'**overflow int16** (`fix_acc32`, accumulateur en int32) ne récupère quasi rien :
  monitoring 0,118 → **0,042** ;
- **calibrer l'échelle** (`per_tensor_calib`) récupère **tout** : monitoring 0,118 → **0,920**.

Autrement dit : ce n'est pas « INT8 » qui casse, ni « PTQ » qui casse — c'est **le scale
figé**. Le vrai clivage n'est donc pas *QAT vs PTQ*, mais **PTQ bien calibrée vs PTQ
naïve** — ce qui mène directement à `int8_v2`.

Figures : [`pedagogy/qat_vs_ptq.png`](../figures/quantization/pedagogy/qat_vs_ptq.png),
[`pedagogy/mapping_affine_int8.png`](../figures/quantization/pedagogy/mapping_affine_int8.png),
[`impact/ablation_perte_f1.png`](../figures/quantization/impact/ablation_perte_f1.png),
[`pedagogy/erreur_quantification_poids.png`](../figures/quantization/pedagogy/erreur_quantification_poids.png).

---

## Axe 3 — `int8_v2`, la meilleure approche rencontrée + ses spécificités

`int8_v2` est la réponse à l'axe 2 : **une PTQ, mais calibrée**. Elle prend le legacy et en
corrige chaque défaut identifié par l'ablation, sans jamais avoir à réentraîner le modèle.

**Principe.** PTQ **calibrée** :
- échelle **par tenseur** (`s = max|W|/127`) ou **par canal de sortie**
  (`s_j = max|W[j,:]|/127`) — l'échelle s'adapte à la dynamique réelle des poids au lieu
  d'être figée à `1/128` ;
- quantification **arrondie** (`round`, pas `trunc`) — l'erreur d'arrondi est centrée, pas
  systématiquement biaisée vers zéro ;
- activations 8-bit à scale **calibré sur un lot représentatif** ;
- **accumulateur int32** — plus de *wrap*, plus d'overflow latent ;
- déquantification exacte `acc·s_w·s_a` sur FPU.

Idée-force : c'est le **miroir PTQ du schéma per-channel du QAT PC**. On récupère le
bénéfice du QAT (échelles adaptées par canal) **sans réentraîner** ni avoir à exporter des
scales appris vers le firmware — la calibration se fait *après coup*, directement sur les
poids FP32 déjà entraînés.

**Ce qui la distingue du legacy, point par point.**

| | `int8_ptq_legacy` | `int8_v2` |
|---|---|---|
| Échelle | figée `1/128` | **calibrée** `max\|W\|/127` (par tenseur ou par canal) |
| Arrondi | `trunc` (biaisé) | `round` (centré) |
| Accumulateur | int16 (wrap) | **int32** (pas de wrap) |
| Granularité | globale | **per-channel** possible |

**Où dans le code.**
- Firmware : [`firmware/stm32f4_blink/src/ewc_head_int8_v2.c`](../../firmware/stm32f4_blink/src/ewc_head_int8_v2.c)
  / [`inc/ewc_head_int8_v2.h`](../../firmware/stm32f4_blink/inc/ewc_head_int8_v2.h)
  (`ewc_int8_v2_from_fp32_calib`).
- Export : `scripts/export_weights_c.py --int8-v2` → `inc/ewc_head_int8_v2_weights.h`
  (réutilise `_weight_scales` / `_quant_weight` / `calibrate_activations` de l'émulateur =
  **parité C↔Python par construction**) ; `--int8-v2-test-vectors` → golden Unity
  ([`tests/test_ewc_int8_v2.c`](../../firmware/stm32f4_blink/tests/test_ewc_int8_v2.c)).
- Émulateur : `QuantConfig.per_tensor_calib()` / `QuantConfig.per_channel_int8()`.

**Impact mesuré — récupère la métrique FP32 en 8 bits.**

| Dataset | cmapss | cwru | monitoring | pronostia | paderborn |
|---|---|---|---|---|---|
| F1 **fp32** | 0,448 | 0,996 | 0,919 | 0,962 | 0,800 |
| F1 **int8_v2** (per-channel) | 0,448 | 0,995 | 0,919 | 0,943 | 0,800 |

Source : [`exp_S39_ablation/`](../../experiments/exp_S39_ablation/). Sur le sweep 4 modèles
× 5 datasets, ewc×cmapss passe de legacy **0,350** → per-channel **0,4527** ≈ fp32
**0,4532**, avec une RAM des poids **3 016 → 754 B** (×4)
([`exp_S39_quant_sweep/summary.json`](../../experiments/exp_S39_quant_sweep/summary.json)).
Board (frozen, 300 échantillons streamés) :
[`exp_S39_board/`](../../experiments/exp_S39_board/)
(`results_per_channel_int8_{cmapss,pronostia}.json`).

**Validation board (mesurée).** Le kernel v2 per-channel est validé sur NUCLEO-F439ZI en
**frozen** (parité bit-exacte 1,000 vs émulateur) **et en online** (mise à jour CL sur la
board). Cellules `monitoring` mesurées (campagne Sprint 40,
[`exp_S40_board_v2/`](../../experiments/exp_S40_board_v2/)) : frozen **F1 = 0,917**, parité
**1,000** ; online **F1 = 0,902**, parité **0,989** (float32 board vs float64 PC), latence
inf + MAJ **577 µs ≪ 100 ms** (Gap 2 ✅), RAM ×4, 0 CRC. `pronostia` mesuré de même
(frozen + online). Restent « à mesurer » les variantes hors périmètre de ce fil (Q15,
legacy A/B) de la grille complète [`S4002`](../sprints/sprint_40/S4002_board_validation_v2.md).

**Spécificité / piège assumé — le paradoxe latence.** `int8_v2` divise la RAM des poids par
4, **mais n'accélère pas la latence** : le MAC reste déquantifié en FP32 sur la FPU, il n'y
a pas de chemin entier SIMD (`SMLAD`) ni de CMSIS-NN. Le gain BOPs théorique ((32/8)² = 16,
Sprint 33) ne se matérialise donc pas encore côté temps de calcul. C'est un choix assumé,
pas un bug — voir § Limites.

Figures :
[`pipeline/pipeline_int8_v2_q15.png`](../figures/quantization/pipeline/pipeline_int8_v2_q15.png),
[`impact/metrique_par_strategie.png`](../figures/quantization/impact/metrique_par_strategie.png),
[`impact/ram_gap3.png`](../figures/quantization/impact/ram_gap3.png),
[`impact/paradoxe_latence.png`](../figures/quantization/impact/paradoxe_latence.png).

---

## Récapitulatif — les 4 stratégies du fil

| Stratégie | Format poids | Moment | Calibration | RAM (vs FP32) | Latence Cortex-M4 | Métrique préservée ? | Validation board |
|---|---|---|---|---|---|---|---|
| `fp32` | float32 | — | — | 1× (réf.) | réf. (FPU) | réf. | ✅ Sprints 20–38 |
| `int8_qat` | int8 simulé (float) | pendant l'entraînement | scales appris (STE) | ×4 (théorique) | N/A (PC) | ✅ Δ ≤ 0,006 (`exp_S28_PC_*`) | ❌ scales non exportés |
| `int8_ptq_legacy` | int8, `s = 1/128` figé | après (au flash) | aucune | ×4 (`exp_S39_quant_sweep`) | ≈ FP32 (déquant FPU, S29) | ❌ F1 0,96 → 0,07 (pronostia, `exp_S39_ablation`) | ✅ mesuré… et invalidé (S29/S36) |
| `int8_v2` | int8 per-tensor/-channel | après, calibré | scale poids + activations | ×4 (754/3 016 B) | ≈ FP32 (déquant FPU) | ✅ ≈ FP32 (`exp_S39_ablation`) | ✅ frozen + online (`exp_S40_board_v2`, monitoring F1 0,92/0,90 · parité 1,000/0,989) |

> **Hors de ce fil.** `q15` (grille 16 bits, décisive pour les tenseurs à **grande
> dynamique** comme `sigma_inv_` de Mahalanobis) et `int16_am` (mémoire associative HDC)
> répondent à d'autres besoins et sont détaillés dans
> [`quantization_strategies.md`](quantization_strategies.md) §5–6.

---

## Chronologie — le fil resserré

| Sprint | Événement |
|---|---|
| 22 | Gap 3 « comblé » une première fois : EWC INT8 + HDC INT8, Python + C. |
| 28 | Benchmark PC 4 × 5 : **QAT PC ✅** (EWC Δ ≤ 0,006, HDC Δ = 0). |
| 29 | Portage board INT8 : RAM ×2,70–4,00 ✅ mais **latence non accélérée** (FPU) ; EWC PTQ board déjà suspect (AUROC 0,25). |
| 36 | Comparaison appariée PC↔board : **PTQ legacy board ❌** formellement établi (F1 0,07–0,15 vs 0,92) — distinct du QAT PC. |
| 39 | **Diagnostic par émulateur bit-exact** : la cause est l'échelle figée `1/128` (pas l'overflow int16) ; `per_tensor_calib` récupère +0,88 F1 (monitoring). Kernel **v2 calibré ✅** (émulé + frozen board). |

---

## Limites & travaux futurs

- **Paradoxe latence FPU** : l'INT8 réduit la RAM ×4 mais pas la latence — le MAC est
  déquantifié en FP32 sur FPU, sans chemin entier SIMD (`SMLAD`) ni CMSIS-NN. Spec :
  [`S3910_simd_cmsis_spec.md`](../sprints/sprint_39/S3910_simd_cmsis_spec.md), `TODO(dorra)`.
- **QAT exporté vers board** : les scales appris du QAT PC ne sont pas consommés par le
  firmware ; exporter QAT → `int8_v2` fermerait l'écart PC/board par construction (piste
  S36/S39).
- **Grille board v2 complète** : le kernel v2 per-channel est mesuré board frozen + online
  (monitoring, pronostia) ; restent « à mesurer » les variantes Q15 et legacy A/B de la
  grille S4002 ([`S4002`](../sprints/sprint_40/S4002_board_validation_v2.md)), hors périmètre
  de ce fil FP32/INT8.
- **Énergie** : le gain BOPs ((32/8)² = 16) reste théorique tant que la sonde LPM01A n'a
  pas tourné — champs « à mesurer »
  ([`exp_S33_energy/`](../../experiments/exp_S33_energy/), Sprint 33).
