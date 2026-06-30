#include "hdc.h"
#include <math.h>
#include <string.h>

void hdc_init(HDCClassifier *h)
{
    memset(h, 0, sizeof(*h));
    /* Le memset met rb.storage à NULL : ré-initialiser le ring buffer sur buf_storage. */
    ring_buffer_init(&h->rb, h->buf_storage, HDC_BUF_ELEM_SIZE, HDC_RETRAIN_BUF);
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

/**
 * hdc_update_with_sample — Update AM + stocke l'échantillon brut dans le buffer.
 *
 * Stockage en uint8 pour économiser la RAM :
 *   raw = (uint8_t)((x[j] + 1.0f) * 127.5f)  — plage [-1,1] → [0,255]
 * MEM buf_storage : HDC_RETRAIN_BUF * (HDC_N_FEATURES + 1) B = 50 * 6 = 300 B
 *   (features quantifiées + label entrelacés, géré par ring_buffer — S3402)
 */
void hdc_update_with_sample(HDCClassifier *h, const float *x,
                             const float *hv, int label)
{
    hdc_update(h, hv, label);

    /* Stocker dans le buffer circulaire (S3402 : via l'abstraction ring_buffer).
     * Élément entrelacé : [feat0..feat4 quantifiés uint8][label]. */
    uint8_t elem[HDC_BUF_ELEM_SIZE];
    for (int j = 0; j < HDC_N_FEATURES; j++) {
        float clamped = x[j] < -1.0f ? -1.0f : (x[j] > 1.0f ? 1.0f : x[j]);
        elem[j] = (uint8_t)((clamped + 1.0f) * 127.5f);
    }
    elem[HDC_N_FEATURES] = (uint8_t)label;
    ring_buffer_push(&h->rb, elem);

    /* Miroirs pour accès direct (tests/diagnostics) */
    h->buf_head  = h->rb.head;
    h->buf_count = h->rb.count;
}

/**
 * hdc_binarize — Re-binarise les prototypes AM en ±1 après accumulation.
 *
 * Après N mises à jour, am[c][i] ∈ ℝ (somme de ±1).
 * Cette fonction repasse chaque am[c][i] à +1.0f si ≥ 0, -1.0f sinon.
 * À appeler lors de la consolidation de tâche (PROTO_FLAG_CONSOLIDATE).
 *
 * Complexité : O(HDC_N_CLASSES × HDC_DIM)
 * MEM stack : 0 B supplémentaire (in-place)
 */
void hdc_binarize(HDCClassifier *h)
{
    for (int c = 0; c < HDC_N_CLASSES; c++) {
        for (int i = 0; i < HDC_DIM; i++) {
            h->am[c][i] = (h->am[c][i] >= 0.0f) ? 1.0f : -1.0f;
        }
    }
}

/**
 * hdc_retrain — Réentraîne l'AM depuis le buffer interne borné.
 *
 * Remet à zéro l'AM, puis ré-accumule tous les échantillons du buffer
 * (dans l'ordre d'insertion, FIFO circulaire) et binarise.
 *
 * MEM stack : float hv[HDC_DIM] = HDC_DIM * 4 B = 4 Ko @ FP32
 *   → Ne pas appeler depuis une ISR.
 */
void hdc_retrain(HDCClassifier *h)
{
    /* Remettre AM à zéro */
    for (int c = 0; c < HDC_N_CLASSES; c++) {
        for (int i = 0; i < HDC_DIM; i++) {
            h->am[c][i] = 0.0f;
        }
    }

    /* Ré-accumulation depuis le buffer circulaire (lecture FIFO via ring_buffer, S3402).
     * stride=1 → mêmes échantillons dans le même ordre qu'avant la migration. */
    float hv[HDC_DIM];   /* MEM: 4 Ko @ FP32 (stack) */
    uint8_t win[HDC_RETRAIN_BUF * HDC_BUF_ELEM_SIZE]; /* MEM: 300 B (stack) */
    int count = ring_buffer_window(&h->rb, win, HDC_RETRAIN_BUF, 1);
    for (int k = 0; k < count; k++) {
        const uint8_t *elem = win + (size_t)k * HDC_BUF_ELEM_SIZE;
        /* Dequantize uint8 → float [-1, 1] : x = (raw / 127.5f) - 1.0f */
        float x[HDC_N_FEATURES];
        for (int j = 0; j < HDC_N_FEATURES; j++) {
            x[j] = ((float)elem[j] / 127.5f) - 1.0f;
        }
        hdc_encode(h, x, hv);
        hdc_update(h, hv, (int)elem[HDC_N_FEATURES]);
    }
    hdc_binarize(h);
}
