# Profondeur & schéma de quantification — taxonomie CL-Embedded (S4701)

> **Résumé** : ce document est la source de vérité textuelle du **troisième axe** de la
> quantification — la **profondeur en bits** (`weight_bits`), la **granularité** de la
> calibration (`per_tensor` / `per_channel`) et la **symétrie** du mapping (`symmetric` /
> `affine` à zero-point). Il fige le vocabulaire et le mapping par modèle (EWC-only), pour
> que le harnais [`run_s47_quant_depth.py`](../../scripts/run_s47_quant_depth.py) (S4702) et
> les figures (S4706) parlent le même langage. **Aucun chiffre de résultat ici** — ce
> document est une taxonomie ; les valeurs sortent de S4703–S4705.

**Trois axes orthogonaux**. La quantification a trois axes que le projet a traités
séparément et qui se citent mutuellement :

1. **Moment** — *quand* quantifier : `before` (QAT) / `after` (PTQ) / `both` (QAT→export
   PTQ). Traité par [`quantization_moments.md`](quantization_moments.md) (S4601, Sprint 46).
2. **Format** — *quel type* : `int8_ptq_legacy`, `int8_v2` per-tensor/per-canal, `q15`,
   `int16_am`. Traité par [`quantization_strategies.md`](quantization_strategies.md) (S4202).
3. **Profondeur & schéma** — *à quelle résolution et avec quelle calibration* : **c'est
   l'objet de ce document**.

Ces trois axes sont indépendants : un même **moment** (`after`) peut employer plusieurs
**formats** (INT8, Q15) à plusieurs **profondeurs** (8, 4, 2 bits) ; un même **format**
(INT8 calibré per-canal symétrique) est le point de la grille profondeur/schéma à
`weight_bits=8, granularity=per_channel, symmetry=symmetric`. La profondeur/schéma
**généralise** les formats connus **en-dessous de 8 bits**.

---

## 1. Les trois sous-axes (bits / granularité / symétrie)

| Sous-axe | Clé config | Valeurs | Définition |
|----------|-----------|---------|------------|
| **Profondeur** | `weight_bits` | `8, 6, 4, 3, 2, ternaire, binaire` | Nombre de bits de la grille de poids. `qmax = 2^(b−1)−1` (8→127, 6→31, 4→7, 3→3, 2→1). **ternaire** = {−1, 0, +1} (1,58 bit ; seuil TWN `Δ_j = 0,7·mean\|W[j,:]\|` par canal, scale `α_j = mean\|w[\|w\|>Δ]\|`). **binaire-poids** = {−1, +1} (BWN, scale par-canal `α_j = mean\|W[j,:]\|` ; les **activations restent 8-bit**). |
| **Granularité** | `granularity` | `per_tensor, per_channel` | `per_tensor` : un scale `max\|W\|/qmax` par couche. `per_channel` : un scale par neurone de sortie (ligne de `W`, `max\|W[j,:]\|/qmax`). Hypothèse (H1) : la per-channel isole les canaux à grande dynamique et **repousse le « cliff »** de plusieurs bits. |
| **Symétrie** | `symmetry` | `symmetric, affine` | `symmetric` : `q = round(w/s)`, plage `[−qmax, qmax]` (schéma actuel de l'émulateur). `affine` : `q = round(a/s) + z` avec zero-point `z` (réutilise [`compute_scale_zero_point`](../../src/utils/quantization.py)), déquant `(q − z)·s`. Pertinent pour les **activations post-ReLU (≥ 0)** dont la dynamique asymétrique gaspille la moitié négative de la grille signée. Les **poids restent symétriques signés**. |

> **Renvoi format**. À **8 bits, per-channel, symétrique**, ce sous-axe **coïncide avec
> `int8_v2`** (S4202) ; à **16 bits**, avec `q15` (Sprint 34, borne haute). La
> profondeur/schéma généralise ces formats connus vers le sub-INT8.

L'émulateur [`int8_c_emulation.py`](../../src/utils/int8_c_emulation.py) paramétrise **déjà**
`n_bits` (dans `_weight_scales` / `_quant_weight` / `_act_params`, `qmax = (1<<(n_bits−1))−1`)
et la granularité ; le Sprint 47 **expose** ces paramètres via `QuantConfig.subint8(...)` et
ajoute les modes ternaire/binaire et l'activation affine, **sans réécrire le moteur** (0
régression sur les presets S39).

---

## 2. RAM théorique (bit-packée) vs matérialisée — point d'honnêteté

| `weight_bits` | Ratio RAM **théorique** (bit-packé) | Condition de matérialisation |
|:---:|:---:|---|
| 8 | ×4 vs FP32 | conteneur `int8` natif |
| 6 | ×5,3 | **packing 6-bit** (non trivial) |
| 4 | ×8 | **packing 2 poids/octet** |
| 3 | ×10,7 | packing 3-bit |
| 2 | ×16 | **packing 4 poids/octet** |
| ternaire | ×~20 | encodage 2-bit ou RLE (info 1,58 bit) |
| binaire | ×32 | packing 1 bit/poids |

**Sur PC (émulateur, ce sprint), la RAM reportée est THÉORIQUE** — calculée depuis
`weight_bits`, elle **suppose** le packing. Un poids INT4 **stocké dans un `int8_t`**
n'économise **rien de plus** que l'INT8 : le gain ÷8 (INT4) / ÷16 (INT2) n'est réel qu'avec
un **kernel bit-packé** (dépacking + MAC FPU), dont le coût est **mesuré au Sprint 48**. Les
figures S47 étiquettent explicitement « RAM théorique (bit-packée) ».

**Ce que l'émulateur mesure / ne mesure pas** : il mesure la **métrique** (AUROC) et une
**RAM analytique** ; il **ne mesure ni latence ni RAM `.bss` réelles**. La latence sub-INT8
(dépacking + MAC FPU) est un `TODO(dorra)` tranché sur board (Sprint 48).

---

## 3. Mapping par modèle (pourquoi EWC-only)

| Modèle | Axe profondeur/schéma applicable ? | Justification |
|--------|:---:|---------------|
| **EWC** | ✅ (bits × granularité × symétrie) | Tête neuronale à poids continus quantifiés par scale → la profondeur en bits est un **vrai continuum** que l'émulateur bit-exact balaie. **Périmètre exclusif** du sprint. |
| **HDC** | ✖ **structurel** | **Nativement entier** : hypervecteurs ±1 (int8), mémoire associative int16. Il n'y a **pas de scale de poids** à réduire en bits — la « profondeur » est fixée par la structure (`int16_am`, S4202). Balayer des sub-bits n'a pas de sens. |
| **Mahalanobis** | ✖ **format-only** | Détecteur **sans poids appris par gradient** ; son axe pertinent est le **format de Σ⁻¹** (INT8 affine casse / Q15 récupère, Sprint 34), pas la profondeur d'une tête neuronale. |
| **TinyOL** | 🟡 **hors-périmètre** | A une tête entraînable (OtO), mais l'utilisateur a fixé le périmètre **EWC-only** ; documenté en **contexte** (S4705), pas balayé. |

**Conséquence de cadrage** : le sweep profondeur/schéma ne concerne que **EWC × {Monitoring
(D2), Pronostia (D4)}**. HDC / Maha / TinyOL sont documentés en **contexte N/A honnête**
(S4705) — jamais remplis par un chiffre artificiel.

---

## 4. Métrique & voie de référence

Reprise de la décision **S4601** : **voie QAT binaire → AUROC** (détection normal-vs-faute)
pour EWC, cohérente avec l'axe moment (Sprint 46) et disponible dans une seule classe de
modèle. La métrique reportée reste **native au modèle** (AUROC). `TODO(arnaud)` : confirmer
l'AUROC binaire comme référence de l'axe profondeur.

---

## 5. Clés de configuration

Configs `configs/quant_depth/*.yaml` (héritent de `ewc_int8_<dataset>.yaml` via `extends:`) :

```yaml
extends: ../ewc_int8_monitoring.yaml
weight_bits: 4            # 8 | 6 | 4 | 3 | 2 | ternaire | binaire
granularity: per_channel  # per_tensor | per_channel
symmetry: symmetric       # symmetric | affine
act_bits: 8               # activations (8 par défaut ; 16 = borne Q15)
seed: 42
metric: auroc
```

Ces clés mappent vers un `QuantConfig` étendu (S4702, preset `QuantConfig.subint8(bits,
granularity, symmetry, mode)`) ; **aucun hyperparamètre en dur** (conforme CLAUDE.md
§ Reproductibilité). Mapping du harnais : `weight_bits` entier → `mode="linear"` ;
`"ternaire"` → `mode="ternary"` (bits nominaux 2) ; `"binaire"` → `mode="binary"` (bits 1) ;
`granularity` → `weight_scale` ; `act_bits` 8/16 → `act_repr` `q7_calib`/`q15`.

---

## 6. Renvois

- **Format** (axe orthogonal) : [`quantization_strategies.md`](quantization_strategies.md) (S4202).
- **Moment** (axe orthogonal) : [`quantization_moments.md`](quantization_moments.md) (S4601, Sprint 46).
- **Q15 = borne haute 16-bit** : Sprint 34 (`mahalanobis_q15`, récupération de la dynamique).
- **Émulateur bit-exact** : [`int8_c_emulation.py`](../../src/utils/int8_c_emulation.py) (S3902).
- **Positionnement Gap 3** : CLAUDE.md § « Triple gap » — l'INT8 donne ÷4 ; ce sprint mesure
  jusqu'où descendre (÷8, ÷16) **avant** que la tête EWC perde sa métrique, et quel schéma
  repousse ce mur.
