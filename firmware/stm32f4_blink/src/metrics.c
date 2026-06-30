/**
 * metrics.c — Métriques CL on-board : accuracy, AUROC approché, forgetting
 *
 * Toutes les structures sont statiques (pas de malloc).
 * OnlineAUROC utilise une fenêtre glissante de AUROC_WINDOW=50 samples.
 *
 * Référence : src/evaluation/online_metrics.py (équivalent Python).
 * Compatible STM32F439ZI Cortex-M4 FPU + STM32N6 Cortex-M55.
 */

#include "metrics.h"
#include <math.h>
#include <string.h>

/* ── Accuracy ───────────────────────────────────────────────────────────── */

void acc_init(OnlineAccuracy *a)
{
    a->n_correct = 0U;
    a->n_total   = 0U;
}

void acc_update(OnlineAccuracy *a, int pred, int true_label)
{
    if (pred == true_label) a->n_correct++;
    a->n_total++;
}

float acc_compute(const OnlineAccuracy *a)
{
    if (a->n_total == 0U) return 0.0f;
    return (float)a->n_correct / (float)a->n_total;
}

/* ── AUROC fenêtre glissante ────────────────────────────────────────────── */

void auroc_init(OnlineAUROC *a)
{
    memset(a->scores, 0, sizeof(a->scores));
    memset(a->labels, 0, sizeof(a->labels));
    a->head  = 0U;
    a->count = 0U;
}

void auroc_update(OnlineAUROC *a, float score, int label)
{
    a->scores[a->head] = score;
    a->labels[a->head] = (uint8_t)(label & 1);
    a->head = (a->head + 1U) % AUROC_WINDOW;
    if (a->count < AUROC_WINDOW) a->count++;
}

/* Wilcoxon-Mann-Whitney O(W²) — acceptable pour W=50 */
float auroc_compute(const OnlineAUROC *a)
{
    if (a->count < 2U) return 0.5f;

    uint32_t n_pos = 0U, n_neg = 0U;
    for (uint32_t i = 0; i < a->count; i++) {
        if (a->labels[i]) n_pos++;
        else               n_neg++;
    }
    if (n_pos == 0U || n_neg == 0U) return 0.5f;

    uint32_t concordant = 0U;
    for (uint32_t i = 0; i < a->count; i++) {
        if (!a->labels[i]) continue;       /* only positives in outer loop */
        for (uint32_t j = 0; j < a->count; j++) {
            if (a->labels[j]) continue;    /* only negatives in inner loop */
            if (a->scores[i] > a->scores[j]) concordant++;
        }
    }
    return (float)concordant / (float)(n_pos * n_neg);
}

/* ── Forgetting tracker ─────────────────────────────────────────────────── */

void fgt_init(ForgettingTracker *f)
{
    memset(f->peak_acc,    0, sizeof(f->peak_acc));
    memset(f->current_acc, 0, sizeof(f->current_acc));
    memset(f->seen,        0, sizeof(f->seen));
}

void fgt_update(ForgettingTracker *f, uint8_t task_id, float acc)
{
    if (task_id >= MAX_TASKS) return;
    f->current_acc[task_id] = acc;
    f->seen[task_id] = 1U;
    if (acc > f->peak_acc[task_id]) f->peak_acc[task_id] = acc;
}

float fgt_avg_forgetting(const ForgettingTracker *f)
{
    float total = 0.0f;
    uint8_t count = 0U;
    for (uint8_t t = 0; t < MAX_TASKS; t++) {
        if (!f->seen[t]) continue;
        float drop = f->peak_acc[t] - f->current_acc[t];
        if (drop > 0.0f) total += drop;
        count++;
    }
    return count > 0U ? total / (float)count : 0.0f;
}

float fgt_backward_transfer(const ForgettingTracker *f)
{
    /* BWT = mean(current_acc[t] - acc_at_end_of_task_t)
     * Simplifié ici : BWT ≈ -AF (corrélation directe pour détection anomalie) */
    return -fgt_avg_forgetting(f);
}

/* ── OnlineRMSE (Welford) ─────────────────────────────────────────────────── */

void online_rmse_init(OnlineRMSE *r)
{
    r->n    = 0U;
    r->mean = 0.0f;
    r->M2   = 0.0f;
    r->rmse = 0.0f;
}

/* Welford online : mise à jour en O(1), numériquement stable.
 * TODO(arnaud) : RMSE = sqrt(E[(ŷ-y)²]) si mean≈0 → aligner convention manuscrit */
void online_rmse_update(OnlineRMSE *r, float y_pred, float y_true)
{
    float err   = y_pred - y_true;
    float sq    = err * err;
    r->n++;
    float delta  = sq - r->mean;
    r->mean     += delta / (float)r->n;
    float delta2 = sq - r->mean;
    r->M2       += delta * delta2;
    r->rmse      = (r->n > 1U) ? sqrtf(r->M2 / (float)(r->n - 1U)) : 0.0f;
}

float online_rmse_get(const OnlineRMSE *r) { return r->rmse; }

/* ── OnlineF1Macro ────────────────────────────────────────────────────────── */

void online_f1_init(OnlineF1Macro *f)
{
    for (int i = 0; i < MAX_MC_CLASSES; i++)
        for (int j = 0; j < MAX_MC_CLASSES; j++)
            f->cm[i][j] = 0;
    f->n_classes = EWC_MC_N_CLASSES;
}

void online_f1_update(OnlineF1Macro *f, int pred, int true_label)
{
    if (true_label < 0 || true_label >= f->n_classes) return;
    if (pred       < 0 || pred       >= f->n_classes) return;
    if (f->cm[true_label][pred] < 32767) f->cm[true_label][pred]++;
}

/* F1 par classe = 2×TP / (2×TP + FP + FN), macro-average sur classes vues */
float online_f1_get(const OnlineF1Macro *f)
{
    float sum_f1 = 0.0f;
    int   n_seen = 0;

    for (int c = 0; c < f->n_classes; c++) {
        int tp = (int)f->cm[c][c];
        int fp = 0, fn = 0;
        for (int j = 0; j < f->n_classes; j++) {
            if (j != c) fp += (int)f->cm[j][c];
            if (j != c) fn += (int)f->cm[c][j];
        }
        int denom = 2 * tp + fp + fn;
        if (denom > 0) {
            sum_f1 += (2.0f * (float)tp) / (float)denom;
            n_seen++;
        }
    }
    return (n_seen > 0) ? sum_f1 / (float)n_seen : 0.0f;
}

/* ── Snapshot UART ──────────────────────────────────────────────────────── */

void metrics_encode_snapshot(const MetricsSnapshot *s, uint8_t *buf)
{
    /* Encode [acc:f32][auroc:f32][forgetting:f32] = 12 B little-endian */
    const uint8_t *p;

    p = (const uint8_t *)&s->accuracy;
    buf[0] = p[0]; buf[1] = p[1]; buf[2] = p[2]; buf[3] = p[3];

    p = (const uint8_t *)&s->auroc;
    buf[4] = p[0]; buf[5] = p[1]; buf[6] = p[2]; buf[7] = p[3];

    p = (const uint8_t *)&s->forgetting;
    buf[8] = p[0]; buf[9] = p[1]; buf[10] = p[2]; buf[11] = p[3];
}
