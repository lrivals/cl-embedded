/**
 * hdc_int8.c — HDC à stockage INT8/INT16 pour Cortex-M4 (Sprint 29, S2901)
 *
 * Variante INT8 de hdc.c :
 *   - bv  : hypervecteurs de base binarisés ±1 (int8_t), figés à l'init (LCG seed fixe)
 *   - am  : mémoire associative INT16, accumulation de bundles ±1 (saturation ±32767)
 *   - query : produit scalaire INT8×INT16 accumulé en int32, argmax
 *
 * Allocation 100 % statique (pas de malloc). Pattern aligné sur ewc_head_int8.c.
 *
 * Référence : Benatti2019HDC, src/models/hdc/hdc_int8.py, hdc.c
 */

#include "hdc_int8.h"
#include <string.h>

/* Saturation INT16 — équivalent __SSAT 16 bits, reproduit la borne du modèle Python
 * (_INT16_MAX / _INT16_MIN dans hdc_int8.py::update_int8). */
#define SAT16(x)  ((int16_t)((x) > 32767 ? 32767 : ((x) < -32768 ? -32768 : (x))))

/* Seed LCG — toute modification invalide la projection (doit rester fixe PC↔board). */
#define HDC_INT8_LCG_SEED  0x1234ABCDU

/* ── Initialisation ─────────────────────────────────────────────────────── */

void hdc_int8_init(HDCInt8 *h)
{
    /* AM remise à zéro */
    memset(h->am, 0, sizeof(h->am));

    /* Base vectors binarisés ±1 via LCG déterministe (Numerical Recipes : a=1664525,
     * c=1013904223). Le bit de poids fort donne un ±1 décorrélé entre dimensions. */
    uint32_t state = HDC_INT8_LCG_SEED;
    for (int n = 0; n < HDC_I_N; n++) {
        for (int i = 0; i < HDC_I_D; i++) {
            state = state * 1664525U + 1013904223U;
            h->bv[n][i] = (state & 0x80000000U) ? (int8_t)1 : (int8_t)-1;
        }
    }
}

/* ── Encodage : projection signée → hypervecteur ±1 ─────────────────────── */

/**
 * hdc_int8_encode — Encode un vecteur de features en hypervecteur ±1.
 *
 * hv_out[i] = sign(Σ_n x[n] · bv[n][i]) ∈ {-1, +1}
 * Accumulation en FP32 (x est float), binarisation finale en int8.
 * bv est stocké feature-major (bv[n][i]) : transposé de proj[i][n] de hdc.c, même sémantique.
 *
 * MEM : 0 B stack supplémentaire (écrit dans hv_out fourni par l'appelant).
 */
void hdc_int8_encode(const HDCInt8 *h, const float *x, int8_t *hv_out)
{
    for (int i = 0; i < HDC_I_D; i++) {
        float dot = 0.0f;
        for (int n = 0; n < HDC_I_N; n++) {
            dot += (float)h->bv[n][i] * x[n];
        }
        hv_out[i] = (dot >= 0.0f) ? (int8_t)1 : (int8_t)-1;
    }
}

/* ── Inférence : argmax du produit scalaire INT8×INT16 ──────────────────── */

int hdc_int8_predict(const HDCInt8 *h, const int8_t *hv)
{
    int     best_class = 0;
    int32_t best_score = 0;

    for (int c = 0; c < HDC_I_C; c++) {
        int32_t score = 0;
        for (int i = 0; i < HDC_I_D; i++) {
            score += (int32_t)h->am[c][i] * (int32_t)hv[i];
        }
        if (c == 0 || score > best_score) {
            best_score = score;
            best_class = c;
        }
    }
    return best_class;
}

/* ── Update en ligne : accumulation INT16 saturée ───────────────────────── */

void hdc_int8_update(HDCInt8 *h, const int8_t *hv, int label)
{
    if (label < 0 || label >= HDC_I_C) return;   /* garde-fou OOB */

    for (int i = 0; i < HDC_I_D; i++) {
        int32_t acc = (int32_t)h->am[label][i] + (int32_t)hv[i];
        h->am[label][i] = SAT16(acc);
    }
}
