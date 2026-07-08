# S3902 — Émulateur Python bit-exact du chemin C INT8

| Champ | Valeur |
|-------|--------|
| **Sprint** | 39 |
| **Priorité** | 🔴 Critique — pivot « maison » du sprint (reproduit la dégradation board sans flasher) |
| **Statut** | ✅ Implémenté (30 juin 2026) |
| **Durée estimée** | 4h |
| **Dépendances** | S3901 ✅ (audit) · `firmware/.../ewc_head_int8.c` ✅ · `src/models/ewc/ewc_mlp_multiclass.py` ✅ |
| **Fichier cible** | `src/utils/int8_c_emulation.py` |
| **Références** | `ewc_head_int8.c` (`ewc_int8_forward`, `ewc_int8_from_fp32`, `float_to_q7`) · `src/utils/quantization.py` |

---

## Contexte

La carte étant indisponible, il faut reproduire **au PC** la dégradation F1 INT8 observée sur board
(Sprint 29/36) pour pouvoir la diagnostiquer et tester des correctifs **avant** tout flash. L'émulateur
réimplémente en NumPy le forward INT8 du firmware, **bit-à-bit**, puis offre des variantes paramétrables
(accumulateur, scale, précision) pour l'ablation (S3904) et le balayage de schémas (S3906).

La tête émulée est `EWCMlpMulticlass` (5→32→16→2), qui est la **référence board** (parité exacte avec
`ewc_forward`, cf. Sprint 32) — et non la tête binaire QAT `EWCMlpInt8Classifier` (sigmoïde 1-sortie).

---

## Chemin de référence bit-exact (`QuantConfig.legacy_c()`)

Reproduit exactement `ewc_int8_forward` + la quantification d'entrée hôte :

| Étape | Code C | Émulateur |
|-------|--------|-----------|
| Quant poids | `SAT8((int)(w*128))` | `_sat8(_trunc_to_int(w*128))` (troncature vers 0) |
| Quant entrée | `(int8_t)(x*128)` (`float_to_q7`) | `_wrap_int8(_trunc_to_int(x*128))` (wrap mod 256) |
| MAC | `int16_t acc += w*x` | `acc = _wrap_int16(acc + w*x)` (**overflow latent**) |
| Déquant | `(float)(acc>>7)/128 + b` | `float(acc>>7)/128 + b` (Q14→Q7, biais FP32) |
| ReLU | `relu_q7(float_to_q7(val))` | `max(_wrap_int8(_trunc_to_int(val*128)), 0)` |

Le wrap `int16` modélise l'overflow décrit en S3901 (F1) ; les variantes utilisent un accumulateur `int32`
sans wrap + déquantification exacte `acc · scale_w[j] · scale_act`.

---

## API publique

```python
from src.utils.int8_c_emulation import (
    EWCHeadWeights, QuantConfig, ABLATION_LADDER,
    forward_fp32, forward_quant, calibrate_activations,
    predict, softmax_prob1, agreement,
)

w = EWCHeadWeights.from_state_dict(torch.load("ewc_head.pt"))   # extrait fc1..fc3
fp = forward_fp32(w, X)                                          # logits FP32 [N, 2]
lg = forward_quant(w, X, QuantConfig.legacy_c())                # chemin firmware bit-exact
lg = forward_quant(w, X, QuantConfig.per_channel_int8())        # variante calibrée
agreement(lg, fp)                                               # taux d'accord prédictions
```

### Schémas fournis (`QuantConfig`)

| Constructeur | weight_scale | act_repr | acc_dtype | Rôle |
|--------------|:------------:|:--------:|:---------:|------|
| `legacy_c()` | fixed_128 | q7_fixed | int16 | firmware actuel (bit-exact) |
| `fix_acc32()` | fixed_128 | q7_fixed | int32 | isole l'overflow |
| `per_tensor_calib()` | per_tensor | q7_calib | int32 | isole le 1/128 figé |
| `per_channel_int8()` | per_channel | q7_calib | int32 | cause racine (mirroir QAT PC) |
| `q15()` | per_channel | q15 | int32 | 16-bit fidélité 256× |
| `mixed_int8w_q15act()` | per_channel (int8) | q15 | int32 | poids int8 + activations q15 |

`ABLATION_LADDER` = échelle canonique du firmware actuel au schéma idéal.

---

## Vérification (smoke-test 30 juin 2026)

Tête synthétique 5→32→16→2, entrées de grande dynamique (σ=2.5) pour stresser clamp/overflow :

| Schéma | accord vs FP32 |
|--------|:--------------:|
| legacy_c | **0.923** (dégradé) |
| fix_acc32 | 0.920 |
| per_tensor_calib | **1.000** |
| per_channel_int8 | **1.000** |
| q15 | **1.000** |
| mixed_int8w_q15act | **1.000** |

→ Le chemin firmware dégrade ; tous les schémas calibrés récupèrent l'accord exact. Mécanisme de
diagnostic validé. La validation **quantitative contre les logs board réels** (F1 0.138 etc.) est l'objet
de S3903.

```bash
python -c "from src.utils.int8_c_emulation import *; import numpy as np; \
  w=EWCHeadWeights(*[np.random.randn(*s).astype('f4') for s in \
  [(32,5),(32,),(16,32),(16,),(2,16),(2,)]]); X=np.random.randn(300,5).astype('f4'); \
  print({c.name: round(agreement(forward_quant(w,X,c),forward_fp32(w,X)),3) for c in ABLATION_LADDER})"
```

---

## Bilan d'implémentation (30 juin 2026)

**Livré** : `src/utils/int8_c_emulation.py` (émulateur bit-exact + 6 schémas + métriques d'accord).
**Périmètre réel** : tête EWC multi-classe (board). Généralisation TinyOL/Maha non couverte (les schémas
Maha INT8/Q15 existent déjà en `src/models/unsupervised/`, réutilisés tels quels par S3906).
**Vérification** : smoke-test OK (legacy dégrade, variantes récupèrent). Parité numérique fine vs C laissée
à S3909 (tests Unity host) et S3903 (parité vs board).
