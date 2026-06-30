/**
 * meta_head.c — Méta-modèle de stacking sur MCU (Sprint 31 / S3105)
 *
 * Forward : logreg  → sigmoid(w·x + b)
 *           MLP      → sigmoid(w2·relu(W1·x + b1) + b2)   (META_HIDDEN > 0)
 *
 * Tout en stack local, pas de malloc. Compatible STM32F439ZI Cortex-M4 FPU.
 * Référence : src/ensemble/meta_learner.py (MetaLearner.export_weights / _meta_forward_np).
 */

#include "meta_head.h"
#include <math.h>    /* expf */
#include <string.h>  /* memcpy */

/* ── Utilitaires locaux ─────────────────────────────────────────────────── */

static float meta_sigmoid(float z)
{
    return 1.0f / (1.0f + expf(-z));
}

#if META_HIDDEN > 0
static float meta_relu(float v)
{
    return v > 0.0f ? v : 0.0f;
}
#endif

/* ── Initialisation depuis meta_weights.h (généré) ──────────────────────── */

void meta_init(MetaHead *m)
{
#if META_HIDDEN > 0
    memcpy(m->w1, META_W1, sizeof(m->w1));
    memcpy(m->b1, META_B1, sizeof(m->b1));
    memcpy(m->w2, META_W2, sizeof(m->w2));
    m->b2 = META_B2;
    /* logreg inutilisé en mode MLP — mis à zéro par cohérence */
    for (int i = 0; i < META_N_FEATURES; i++) {
        m->w[i] = 0.0f;
    }
    m->b = 0.0f;
#else
    memcpy(m->w, META_W, sizeof(m->w));
    m->b = META_B;
#endif
}

/* ── Forward pass ───────────────────────────────────────────────────────── */

float meta_forward(const MetaHead *m, const float *feats)
{
#if META_HIDDEN > 0
    /* MLP : relu(W1·x + b1) → sigmoid(w2·h + b2) */
    float hidden[META_HIDDEN];   /* MEM: META_HIDDEN × 4 B @ FP32 (stack) */
    for (int j = 0; j < META_HIDDEN; j++) {
        float acc = m->b1[j];
        for (int i = 0; i < META_N_FEATURES; i++) {
            acc += m->w1[j][i] * feats[i];
        }
        hidden[j] = meta_relu(acc);
    }
    float z = m->b2;
    for (int j = 0; j < META_HIDDEN; j++) {
        z += m->w2[j] * hidden[j];
    }
    return meta_sigmoid(z);
#else
    /* logreg : sigmoid(w·x + b) */
    float z = m->b;
    for (int i = 0; i < META_N_FEATURES; i++) {
        z += m->w[i] * feats[i];
    }
    return meta_sigmoid(z);
#endif
}

int meta_predict(const MetaHead *m, const float *feats)
{
    return meta_forward(m, feats) > 0.5f ? 1 : 0;
}
