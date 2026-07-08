# S3907 — Kernel C v2 optimisé (`ewc_head_int8_v2.c/.h`)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 39 |
| **Priorité** | 🔴 Critique — corrige les défauts identifiés (S3901), validable sans board (`make test` host) |
| **Statut** | ✅ Implémenté (1er juillet 2026) — `make test` 122 tests (0 régression, 2 TinyOL préexistants) |
| **Durée estimée** | 4h |
| **Dépendances** | S3904 ✅ (facteurs prioritaires) · `firmware/.../ewc_head_int8.c` (base) · `src/utils/int8_c_emulation.py` (parité) |
| **Fichiers cibles** | `firmware/stm32f4_blink/src/ewc_head_int8_v2.c`, `firmware/stm32f4_blink/inc/ewc_head_int8_v2.h` |
| **Références** | `ewc_head_int8.c` (à copier, **garder intact** pour A/B) · `mahalanobis_q15.c` (patron Q15) |

---

## Contexte

Décision utilisateur : **nouveau fichier v2 séparé**, l'ancien `ewc_head_int8.c` reste intact pour la
comparaison A/B (board S3916). Le v2 corrige les trois défauts de S3901 et ajoute les schémas intermédiaires.

## Différences vs `ewc_head_int8.c`

| Aspect | v1 (actuel) | v2 (ce fichier) |
|--------|-------------|-----------------|
| Accumulateur | `int16_t` (overflow) | **`int32_t`** |
| Scale poids | `1/128` figé par-tenseur | **par-canal calibré** (importé du header généré S3908) |
| Scale activation | `1/128` figé (clamp >1) | **calibré** (ou Q15 si `EWC_INT8_Q15`) |
| Déquant | `(acc>>7)/128` | `acc · scale_w[j] · scale_act` (exacte) |
| Précision interm. | Q7 | Q7 par défaut, **`#ifdef EWC_INT8_Q15`** → Q15 |
| Latence | déquant→FP32 | idem (SIMD différé S3910/S3917) |

## Spec header (`ewc_head_int8_v2.h`)

```c
#ifndef EWC_HEAD_INT8_V2_H
#define EWC_HEAD_INT8_V2_H
#include <stdint.h>
#include "ewc_head.h"

/* Scales par-canal (importés via ewc_head_int8_v2_weights.h, généré S3908) */
typedef struct {
    int8_t  w1[EWC_H1][EWC_IN];   float scale_w1[EWC_H1];   /* un scale par neurone de sortie */
    float   b1[EWC_H1];
    int8_t  w2[EWC_H2][EWC_H1];   float scale_w2[EWC_H2];
    float   b2[EWC_H2];
    int8_t  w3[EWC_OUT][EWC_H2];  float scale_w3[EWC_OUT];
    float   b3[EWC_OUT];
    float   scale_act_in, scale_act_h1, scale_act_h2;   /* activations calibrées */
} EWCHeadInt8V2;

void ewc_int8_v2_from_fp32_calib(EWCHeadInt8V2 *dst, const EWCHead *src,
                                 const float *act_max /* [in,h1,h2] */);
void ewc_int8_v2_forward(const EWCHeadInt8V2 *h, const float *x, float *logits);
#endif
```

## Cœur du forward (couche 1, accumulateur int32 + déquant par-canal)

```c
for (int j = 0; j < EWC_H1; j++) {
    int32_t acc = 0;                                   /* ← int32 : plus d'overflow */
    int8_t x_q = sat8(lroundf(x[i] / h->scale_act_in));
    for (int i = 0; i < EWC_IN; i++)
        acc += (int32_t)h->w1[j][i] * (int32_t)x_q;
    float val = (float)acc * h->scale_w1[j] * h->scale_act_in + h->b1[j];  /* déquant exacte */
    h1[j] = val > 0.0f ? val : 0.0f;                   /* ReLU en FP32 (pas de clamp Q7) */
}
```

> Variante `#ifdef EWC_INT8_Q15` : `x_q` en int16 (scale `/32767`), poids int16 Q15 — sinon int8.
> Variante mixte : poids int8, activations int16.

> **Correctif S3909 (acc Q15)** : l'accumulateur `int32_t` **déborde en Q15** (int16×int16 sommé peut
> dépasser 2³¹). Le type d'accumulateur est désormais `ewc_v2_acc_t` = `int32_t` (int8/mixed, inchangé) /
> **`int64_t` (Q15)** — parité bit-à-bit avec l'accumulation int64 de l'émulateur, **0 régression** sur les
> builds int8/mixed. Découvert et validé par `test-v2-q15` (S3909).

## Budget mémoire (NUCLEO-F439ZI)

| Composant | v1 INT8 | v2 INT8 per-channel | v2 Q15 |
|-----------|:-------:|:-------------------:|:------:|
| Poids | 704 B | 704 B + scales (≈200 B) | 1408 B + scales |
| RAM vs FP32 | ×4 | ≈×3.5 (scales) | ≈×1.8 |

> Le coût des scales par-canal (FP32) réduit légèrement le ratio mais **récupère l'accuracy** (cible S3904).

## Vérification (sans board — `make test` sur host x86)

```bash
cd firmware/stm32f4_blink && make test       # inclut test_ewc_int8_v2 (S3909)
```

> Aucun flash requis : la correction de l'overflow et la parité par-canal se valident entièrement sur host
> via Unity. La latence/`.bss`/parité **board** sont S3915 (différé).
