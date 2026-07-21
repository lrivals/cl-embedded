# S4701 — Taxonomie profondeur / schéma de quantification + mapping EWC-only

| Champ | Valeur |
|-------|--------|
| **Sprint** | 47 |
| **Priorité** | 🔴 Critique — fonde le vocabulaire (bits / granularité / symétrie) et le périmètre ; harnais S4702 et figures S4706 en découlent. |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Durée estimée** | 4h |
| **Dépendances** | Sprint 39 ✅ (`int8_c_emulation.py`, `n_bits`+granularité déjà paramétrés) · Sprint 34 ✅ (Q15 = 16-bit borne haute) · `src/utils/quantization.py` ✅ (zero-point affine) · S4601 (voie/métrique EWC) |
| **Fichiers cibles** | `docs/context/quantization_depth.md` (nouveau), référencé par `docs/sprints/sprint_47/S4701_cadrage_taxonomie.md` |
| **Références** | `docs/context/quantization_strategies.md` (S4202, axe *format*) · `docs/context/quantization_moments.md` (S4601, axe *moment*) · CLAUDE.md § « Gap 3 » |

---

## Contexte

La quantification a **trois axes orthogonaux** que le projet a traités séparément :

1. **Moment** — *quand* : avant / après / les-deux (Sprint 46, `quantization_moments.md`).
2. **Format** — *quel type* : INT8 legacy / v2 / Q15 / int16-AM (S4202, `quantization_strategies.md`).
3. **Profondeur & schéma** — *à quelle résolution et avec quelle calibration* : **c'est l'objet de ce document**.

Cette tâche fige le **vocabulaire** du troisième axe et le **mapping EWC-only**, pour que le harnais S4702 et
les figures S4706 parlent le même langage. Elle produit `docs/context/quantization_depth.md`, à l'image de
`quantization_strategies.md` mais orienté *profondeur/schéma*.

## Spec

### 1. Les trois sous-axes de la profondeur/schéma

| Sous-axe | Clé config | Valeurs | Définition |
|----------|-----------|---------|------------|
| **Profondeur** | `weight_bits` | `8, 6, 4, 3, 2, ternaire, binaire` | Nombre de bits de la grille de poids. `qmax = 2^(b-1)−1` (8→127, 6→31, 4→7, 3→3, 2→1). **ternaire** = {−1,0,+1} (1,58 bit) ; **binaire-poids** = {−1,+1} (scale par-canal, activations restent 8-bit). |
| **Granularité** | `granularity` | `per_tensor, per_channel` | `per_tensor` : un scale `max|W|/qmax` par couche. `per_channel` : un scale par neurone de sortie (ligne de `W`). Hypothèse : la per-channel repousse le « cliff » de plusieurs bits. |
| **Symétrie** | `symmetry` | `symmetric, affine` | `symmetric` : `q = round(w/s)`, plage `[−qmax, qmax]` (schéma actuel émulateur). `affine` : `q = round(w/s)+z` avec zero-point `z` (réutilise `src/utils/quantization.py::compute_scale_zero_point`), pertinent pour les **activations post-ReLU ≥ 0** (dynamique asymétrique). |

> **Renvoi format** : à 8 bits per-channel symétrique, ce sous-axe **coïncide avec `int8_v2`** (S4202) ; à 16 bits,
> avec `q15`. La profondeur/schéma **généralise** ces formats connus en-dessous de 8 bits.

### 2. RAM : théorique vs matérialisée (point d'honnêteté)

| `weight_bits` | Ratio RAM **théorique** (bit-packé) | Condition de matérialisation |
|:---:|:---:|---|
| 8 | ×4 vs FP32 | conteneur `int8` natif |
| 6 | ×5,3 | **packing 6-bit** (non trivial) |
| 4 | ×8 | **packing 2 poids/octet** |
| 3 | ×10,7 | packing 3-bit |
| 2 | ×16 | **packing 4 poids/octet** |
| ternaire | ×~20 | encodage 2-bit ou RLE |
| binaire | ×32 | packing 1 bit/poids |

**Sur PC (émulateur, ce sprint)** : la RAM reportée est **théorique** — calculée depuis `weight_bits`, elle
suppose le packing. **Un INT4 stocké dans un `int8_t` n'économise rien de plus que l'INT8** : le gain n'est réel
qu'avec un kernel bit-packé, dont le coût (dépacking + MAC FPU) est mesuré au **Sprint 48**. Les figures S47
étiquettent explicitement « RAM théorique (bit-packée) ».

### 3. Mapping par modèle (pourquoi EWC-only)

| Modèle | Axe profondeur/schéma applicable ? | Justification |
|--------|:---:|---------------|
| **EWC** | ✅ (3-way bits × granularité × symétrie) | Tête neuronale à poids continus quantifiés par scale → la profondeur en bits est un **vrai continuum** ; l'émulateur bit-exact le balaie. Périmètre exclusif du sprint. |
| **HDC** | ✖ structurel | **Nativement entier** : hypervecteurs ±1 (int8), mémoire associative int16. Il n'y a **pas de scale de poids** à réduire en bits — la « profondeur » est fixée par la structure (S4202 `int16_am`). Balayer des sub-bits n'a pas de sens. |
| **Mahalanobis** | ✖ format-only | Détecteur **sans poids appris par gradient** ; son axe pertinent est le **format de Σ⁻¹** (INT8 casse / Q15 récupère, Sprint 34), pas la profondeur d'une tête neuronale. |
| **TinyOL** | 🟡 hors-périmètre | A une tête entraînable, mais l'utilisateur a fixé le périmètre **EWC-only** ; documenté en contexte (S4705), pas balayé. |

**Conséquence de cadrage** : le sweep profondeur/schéma ne concerne que **EWC × {Monitoring, Pronostia}**.
HDC/Maha/TinyOL sont documentés en **contexte** (S4705) avec cellules explicitement **N/A** justifiées, jamais
remplies par un chiffre artificiel.

### 4. Métrique & voie de référence

Reprise de la décision S4601 : **voie QAT binaire → AUROC** (normal-vs-faute) pour EWC, car cohérente avec
l'axe moment (S46) et disponible dans une seule classe de modèle. La métrique reportée reste **native au
modèle** (AUROC). `TODO(arnaud)` : confirmer AUROC binaire comme référence.

### 5. Clés de configuration

Configs `configs/quant_depth/*.yaml` :

```yaml
model: ewc                # ewc (seul modèle du périmètre)
dataset: monitoring       # monitoring | pronostia
weight_bits: 4            # 8 | 6 | 4 | 3 | 2 | ternaire | binaire
granularity: per_channel  # per_tensor | per_channel
symmetry: symmetric       # symmetric | affine
act_bits: 8               # activations (8 par défaut ; 16 = borne Q15)
seed: 42
metric: auroc
```

Ces clés mappent vers un `QuantConfig` étendu (S4702) ; aucun hyperparamètre en dur (conforme CLAUDE.md).

## Format de sortie

Document `docs/context/quantization_depth.md` structuré :

```
# Profondeur & schéma de quantification — taxonomie CL-Embedded
## 1. Les trois sous-axes (bits / granularité / symétrie)   -> table + définitions
## 2. RAM théorique vs matérialisée (bit-packing)            -> table ratios + honnêteté
## 3. Mapping par modèle (EWC balayé ; HDC/Maha/TinyOL N/A)  -> table justifiée
## 4. Métrique & voie de référence (AUROC binaire, S4601)
## 5. Clés weight_bits / granularity / symmetry + exemple YAML
## 6. Renvois : S4202 (format), S4601 (moment), S34 (Q15), CLAUDE.md Gap 3
```

## Contraintes

- Aucun chiffre de résultat dans ce document (taxonomie pure) — les valeurs viennent de S4703–S4705.
- Le mapping HDC/Maha/TinyOL = N/A doit être **justifié** (structurel / format-only / hors-périmètre), pas laissé vide.
- Vocabulaire cohérent avec les deux autres axes : ce doc traite la **profondeur/schéma**, orthogonal au
  **moment** (S4601) et au **format** (S4202) — les trois se citent mutuellement.

## Vérification

```bash
test -f docs/context/quantization_depth.md
grep -c "weight_bits\|granularity\|symmetry" docs/context/quantization_depth.md   # > 0
grep -ci "théorique\|bit-pack" docs/context/quantization_depth.md                 # > 0 (honnêteté RAM)
grep -i "structurel\|format-only\|EWC" docs/context/quantization_depth.md         # mapping justifié
```

---

## Résolution (implémentée)

_À compléter lors de l'implémentation._
