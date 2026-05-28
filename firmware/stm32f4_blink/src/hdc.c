#include "hdc.h"
#include <math.h>
#include <string.h>

void hdc_init(HDCClassifier *h)
{
    memset(h, 0, sizeof(*h));
}

void hdc_encode(const HDCClassifier *h, const float *x, float *hv_out)
{
    /* hv_out[i] = sign(proj[i] · x) ∈ {-1.0f, +1.0f}
     * Propriété : sum(hv_out[i]²) == HDC_DIM car chaque élément est ±1. */
    for (int i = 0; i < HDC_DIM; i++) {
        float dot = 0.0f;
        for (int j = 0; j < HDC_N_FEATURES; j++) {
            dot += h->proj[i][j] * x[j];
        }
        hv_out[i] = (dot >= 0.0f) ? 1.0f : -1.0f;
    }
}

int hdc_predict(const HDCClassifier *h, const float *hv)
{
    /* argmax_c dot(am[c], hv) — proxy cosinus pour hv binarisé */
    int   best_class = 0;
    float best_score = 0.0f;

    for (int c = 0; c < HDC_N_CLASSES; c++) {
        float score = 0.0f;
        for (int i = 0; i < HDC_DIM; i++) {
            score += h->am[c][i] * hv[i];
        }
        if (c == 0 || score > best_score) {
            best_score = score;
            best_class = c;
        }
    }
    return best_class;
}

void hdc_update(HDCClassifier *h, const float *hv, int label)
{
    /* Accumulation pure — pas de recalcul depuis zéro, pas d'oubli catastrophique. */
    for (int i = 0; i < HDC_DIM; i++) {
        h->am[label][i] += hv[i];
    }
    h->n_trained++;
}
