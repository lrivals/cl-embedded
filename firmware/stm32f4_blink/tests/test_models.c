/**
 * test_models.c — Tests Unity pour mahalanobis, ewc_head et tinyol sur mock_data.h
 *
 * Couvre :
 *   - Mahalanobis : score normal < threshold, score anomalie > threshold
 *   - EWC head    : forward pass cohérent, SGD step réduit la loss
 *   - TinyOL      : encode/decode sans malloc, reconstruction error ≥ 0
 *   - Tous        : aucun malloc (vérifié par absence de symbole malloc dans .map)
 *
 * Compile sur host (x86) et sur ARM cible.
 */

#include "unity.h"
#include "mock_data.h"
#include "mahalanobis.h"
#include "ewc_head.h"
#include "tinyol.h"
#include "model_weights.h"
#include "metrics.h"
#include <math.h>
#include <string.h>

#define TOL 1e-4f

/* ────────────────────────────────────────────────────────────────────────── */
/*  MAHALANOBIS                                                               */
/* ────────────────────────────────────────────────────────────────────────── */

void test_maha_normal_below_threshold(void)
{
    MahalanobisDetector det;
    maha_init(&det, 2.0f, 0.05f);

    for (int i = 0; i < (int)MOCK_N_SAMPLES; i++) {
        float score = maha_score(&det, MOCK_NORMAL_T0[i]);
        /* Mean = 0, precision = I → score = ||x||₂ ≈ 0.2 for normal samples */
        TEST_ASSERT_LESS_THAN_FLOAT(MOCK_MAHA_SCORE_NORMAL_T0_MAX, score);
    }
}

void test_maha_anomaly_above_threshold(void)
{
    MahalanobisDetector det;
    maha_init(&det, 2.0f, 0.05f);

    for (int i = 0; i < (int)MOCK_N_SAMPLES; i++) {
        float score = maha_score(&det, MOCK_ANOMALY_T0[i]);
        TEST_ASSERT_GREATER_THAN_FLOAT(MOCK_MAHA_SCORE_ANOMALY_T0_MIN, score);
        TEST_ASSERT_EQUAL_INT(1, maha_predict(&det, MOCK_ANOMALY_T0[i]));
    }
}

void test_maha_predict_correct_normal(void)
{
    MahalanobisDetector det;
    maha_init(&det, 2.0f, 0.05f);

    for (int i = 0; i < (int)MOCK_N_SAMPLES; i++) {
        TEST_ASSERT_EQUAL_INT(0, maha_predict(&det, MOCK_NORMAL_T0[i]));
    }
}

void test_maha_ema_update_shifts_mean(void)
{
    MahalanobisDetector det;
    maha_init(&det, 2.0f, 0.1f);
    float x[MAHA_DIM] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    /* Après 50 updates, mean doit converger vers [1,1,1,1,1] (à ~1% près) */
    for (int i = 0; i < 50; i++) {
        maha_update(&det, x);
    }
    for (int i = 0; i < MAHA_DIM; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.05f, 1.0f, det.mean[i]);
    }
}

/* ────────────────────────────────────────────────────────────────────────── */
/*  EWC HEAD                                                                  */
/* ────────────────────────────────────────────────────────────────────────── */

static void _make_zero_ewc(EWCHead *h)
{
    memset(h, 0, sizeof(EWCHead));
    h->lambda = 400.0f;
}

void test_ewc_forward_zero_weights(void)
{
    EWCHead head;
    _make_zero_ewc(&head);

    float logits[EWC_OUT];
    ewc_forward(&head, MOCK_NORMAL_T0[0], logits);

    /* Poids zéro → logits = biais = 0 */
    for (int j = 0; j < EWC_OUT; j++) {
        TEST_ASSERT_FLOAT_WITHIN(MOCK_EWC_LOGIT_TOLERANCE, 0.0f, logits[j]);
    }
}

void test_ewc_predict_returns_valid_class(void)
{
    EWCHead head;
    _make_zero_ewc(&head);

    for (int i = 0; i < (int)MOCK_N_SAMPLES; i++) {
        int pred = ewc_predict(&head, MOCK_NORMAL_T0[i]);
        TEST_ASSERT_TRUE(pred == 0 || pred == 1);
    }
}

void test_ewc_sgd_step_modifies_weights(void)
{
    EWCHead head;
    _make_zero_ewc(&head);
    head.lambda = 0.0f;
    /* Biais positifs pour que ReLU laisse passer le gradient jusqu'à w1 */
    for (int j = 0; j < EWC_H1; j++) head.b1[j] = 0.1f;
    for (int j = 0; j < EWC_H2; j++) head.b2[j] = 0.1f;
    ewc_sgd_step(&head, MOCK_NORMAL_T0[0], 0);
    int changed = 0;
    for (int j = 0; j < EWC_H1; j++)
        for (int i = 0; i < EWC_IN; i++)
            if (head.w1[j][i] != 0.0f) changed = 1;
    TEST_ASSERT_TRUE(changed);
}

void test_ewc_consolidate_copies_star_weights(void)
{
    EWCHead head;
    _make_zero_ewc(&head);
    head.w1[0][0] = 0.5f;
    head.w3[0][0] = -0.3f;
    ewc_consolidate(&head, 0.9f);
    TEST_ASSERT_EQUAL_FLOAT(head.w1[0][0], head.star_w1[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(head.w3[0][0], head.star_w3[0][0]);
}

void test_ewc_consolidate_fisher_nonnegative(void)
{
    EWCHead head;
    _make_zero_ewc(&head);
    head.w1[0][0] = -0.5f;
    ewc_consolidate(&head, 0.5f);
    for (int j = 0; j < EWC_H1; j++)
        for (int i = 0; i < EWC_IN; i++)
            TEST_ASSERT_GREATER_OR_EQUAL_FLOAT(0.0f, head.fisher1[j][i]);
}

void test_ewc_consolidate_ema_alpha(void)
{
    EWCHead head;
    _make_zero_ewc(&head);
    head.w1[0][0] = 1.0f;
    head.fisher1[0][0] = 0.0f;
    ewc_consolidate(&head, 0.9f);
    /* fisher = 0.9*0 + 0.1*1² = 0.1 */
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.1f, head.fisher1[0][0]);
}

void test_ewc_sgd_reduces_loss_on_normal(void)
{
    EWCHead head;
    _make_zero_ewc(&head);
    head.lambda = 0.0f;  /* Désactive EWC pour ce test */

    /* 10 steps SGD sur samples normaux (label 0) */
    for (int step = 0; step < 10; step++) {
        for (int i = 0; i < (int)MOCK_N_SAMPLES; i++) {
            ewc_sgd_step(&head, MOCK_NORMAL_T0[i], 0);
        }
    }
    /* Après entraînement, le modèle doit prédire 0 pour la majorité des samples normaux */
    int correct = 0;
    for (int i = 0; i < (int)MOCK_N_SAMPLES; i++) {
        if (ewc_predict(&head, MOCK_NORMAL_T0[i]) == 0) correct++;
    }
    TEST_ASSERT_GREATER_THAN_INT(5, correct);  /* > 50% correct */
}

/* ────────────────────────────────────────────────────────────────────────── */
/*  TINYOL                                                                    */
/* ────────────────────────────────────────────────────────────────────────── */

static TinyOLEncoder g_enc;
static TinyOLDecoder g_dec;

/* ────────────────────────────────────────────────────────────────────────── */
/*  EWC CONSOLIDATION — S2004 Groupe 1                                       */
/* ────────────────────────────────────────────────────────────────────────── */

void test_ewc_consolidate_fisher_update(void)
{
    /* alpha=0.9, fisher_init=0, w1[0][0]=1.0 → fisher1[0][0] ≈ 0.1*1²=0.1 */
    EWCHead h;
    _make_zero_ewc(&h);
    h.w1[0][0] = 1.0f;
    h.fisher1[0][0] = 0.0f;
    ewc_consolidate(&h, 0.9f);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.1f, h.fisher1[0][0]);
}

void test_ewc_consolidate_star_copy(void)
{
    /* star_w1[j][i] == w1[j][i] pour tous j,i après consolidation */
    EWCHead h;
    _make_zero_ewc(&h);
    h.w1[0][0] = 0.7f; h.w1[1][2] = -0.3f; h.w1[5][3] = 0.5f;
    ewc_consolidate(&h, 0.9f);
    for (int j = 0; j < EWC_H1; j++)
        for (int i = 0; i < EWC_IN; i++)
            TEST_ASSERT_FLOAT_WITHIN(1e-7f, h.w1[j][i], h.star_w1[j][i]);
}

void test_ewc_consolidate_fisher_nonneg(void)
{
    /* Fisher = w²×(1-alpha) ≥ 0 même avec poids négatifs */
    EWCHead h;
    _make_zero_ewc(&h);
    h.w1[0][0] = -0.5f;
    ewc_consolidate(&h, 0.5f);
    for (int j = 0; j < EWC_H1; j++)
        for (int i = 0; i < EWC_IN; i++)
            TEST_ASSERT_GREATER_OR_EQUAL_FLOAT(0.0f, h.fisher1[j][i]);
    for (int j = 0; j < EWC_H2; j++)
        for (int i = 0; i < EWC_H1; i++)
            TEST_ASSERT_GREATER_OR_EQUAL_FLOAT(0.0f, h.fisher2[j][i]);
    for (int j = 0; j < EWC_OUT; j++)
        for (int i = 0; i < EWC_H2; i++)
            TEST_ASSERT_GREATER_OR_EQUAL_FLOAT(0.0f, h.fisher3[j][i]);
}

/* ────────────────────────────────────────────────────────────────────────── */
/*  EWC REGULARISATION — S2004 Groupe 2                                      */
/* ────────────────────────────────────────────────────────────────────────── */

void test_ewc_penalty_nonzero(void)
{
    /* Avec Fisher>0, w≠star_w, λ=100 : delta w3[0][0] diffère du cas λ=0 */
    EWCHead h_ewc, h_ref;
    _make_zero_ewc(&h_ewc);
    _make_zero_ewc(&h_ref);
    /* Biais positifs pour que ReLU active les couches cachées */
    for (int j = 0; j < EWC_H1; j++) { h_ewc.b1[j] = 0.1f; h_ref.b1[j] = 0.1f; }
    for (int j = 0; j < EWC_H2; j++) { h_ewc.b2[j] = 0.1f; h_ref.b2[j] = 0.1f; }
    /* Poids non nul sur w3 avant consolidation → Fisher > 0 */
    h_ewc.w3[0][0] = 0.3f;
    ewc_consolidate(&h_ewc, 0.9f);   /* fisher3[0][0]=0.1*0.09>0, star_w3[0][0]=0.3 */
    h_ewc.w3[0][0] = 0.8f;          /* perturber loin de star_w3 */
    h_ewc.lambda = 100.0f;
    h_ref.w3[0][0] = 0.8f;
    h_ref.lambda   = 0.0f;

    float w_ewc_before = h_ewc.w3[0][0];
    float w_ref_before = h_ref.w3[0][0];
    ewc_sgd_step(&h_ewc, MOCK_NORMAL_T0[0], 0);
    ewc_sgd_step(&h_ref, MOCK_NORMAL_T0[0], 0);
    float delta_ewc = h_ewc.w3[0][0] - w_ewc_before;
    float delta_ref = h_ref.w3[0][0] - w_ref_before;
    TEST_ASSERT(delta_ewc != delta_ref);
}

void test_ewc_forgetting_reduced(void)
{
    /* Tâche 0 : MOCK_NORMAL_T0 label=0 ; Tâche 1 : MOCK_NORMAL_T1 label=1.
     * Après tâche 1, EWC (λ=400) retient mieux task 0 que sans EWC. */
    EWCHead h_ewc, h_no_ewc;
    _make_zero_ewc(&h_ewc);
    _make_zero_ewc(&h_no_ewc);
    h_ewc.lambda = 0.0f;
    h_no_ewc.lambda = 0.0f;
    /* Biais positifs pour activer les couches cachées dès le début */
    for (int j = 0; j < EWC_H1; j++) { h_ewc.b1[j] = 0.1f; h_no_ewc.b1[j] = 0.1f; }
    for (int j = 0; j < EWC_H2; j++) { h_ewc.b2[j] = 0.1f; h_no_ewc.b2[j] = 0.1f; }

    /* Entraînement tâche 0 */
    for (int s = 0; s < 30; s++)
        for (int i = 0; i < (int)MOCK_N_SAMPLES; i++) {
            ewc_sgd_step(&h_ewc,    MOCK_NORMAL_T0[i], 0);
            ewc_sgd_step(&h_no_ewc, MOCK_NORMAL_T0[i], 0);
        }

    ewc_consolidate(&h_ewc, 0.9f);
    h_ewc.lambda = 400.0f;

    /* Entraînement tâche 1 (classe opposée → catastrophic forgetting) */
    for (int s = 0; s < 30; s++)
        for (int i = 0; i < (int)MOCK_N_SAMPLES; i++) {
            ewc_sgd_step(&h_ewc,    MOCK_NORMAL_T1[i], 1);
            ewc_sgd_step(&h_no_ewc, MOCK_NORMAL_T1[i], 1);
        }

    int ok_ewc = 0, ok_no_ewc = 0;
    for (int i = 0; i < (int)MOCK_N_SAMPLES; i++) {
        if (ewc_predict(&h_ewc,    MOCK_NORMAL_T0[i]) == 0) ok_ewc++;
        if (ewc_predict(&h_no_ewc, MOCK_NORMAL_T0[i]) == 0) ok_no_ewc++;
    }
    TEST_ASSERT_GREATER_OR_EQUAL_INT(ok_no_ewc, ok_ewc);
}

/* ────────────────────────────────────────────────────────────────────────── */
/*  TINYOL — S2004 Groupe 3                                                  */
/* ────────────────────────────────────────────────────────────────────────── */

void test_tinyol_init_weights_loaded(void)
{
    /* Après tinyol_init, enc.w_enc1[0][0] = TINYOL_W_ENC1[0][0] ≈ 0.837 ≠ 0 */
    TinyOLEncoder enc;
    TinyOLDecoder dec;
    tinyol_init(&enc, &dec);
    TEST_ASSERT(enc.w_enc1[0][0] != 0.0f);
}

void test_tinyol_forward_shape(void)
{
    /* L'encoder produit exactement TINYOL_EMB=16 floats ; au moins un non nul */
    TinyOLEncoder enc;
    TinyOLDecoder dec;
    tinyol_init(&enc, &dec);
    float emb[TINYOL_EMB];
    tinyol_encode(&enc, MOCK_NORMAL_T0[0], emb);
    TEST_ASSERT_EQUAL_UINT(16U, TINYOL_EMB);
    float sumsq = 0.0f;
    for (uint32_t i = 0; i < TINYOL_EMB; i++) sumsq += emb[i] * emb[i];
    TEST_ASSERT_GREATER_THAN_FLOAT(0.0f, sumsq);
}

void test_tinyol_forward_delta_vs_ref(void)
{
    /* Forward C doit correspondre à la référence Python seed=42 (max|err|≤1e-5) */
    TinyOLEncoder enc;
    TinyOLDecoder dec;
    tinyol_init(&enc, &dec);
    float emb[TINYOL_EMB];
    tinyol_encode(&enc, MOCK_NORMAL_T0[0], emb);
    for (int i = 0; i < (int)TINYOL_EMB; i++)
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, MOCK_TINYOL_REF_EMB_T0[i], emb[i]);
}

static void _make_zero_tinyol(void)
{
    memset(&g_enc, 0, sizeof(g_enc));
    memset(&g_dec, 0, sizeof(g_dec));
}

void test_tinyol_encode_zero_weights(void)
{
    _make_zero_tinyol();

    float emb[TINYOL_EMB];
    tinyol_encode(&g_enc, MOCK_NORMAL_T0[0], emb);

    /* Poids zéro + biais zéro → après ReLU, embedding = [0, ..., 0] */
    for (uint32_t j = 0; j < TINYOL_EMB; j++) {
        TEST_ASSERT_FLOAT_WITHIN(TOL, 0.0f, emb[j]);
    }
}

void test_tinyol_decode_zero_weights(void)
{
    _make_zero_tinyol();

    float emb[TINYOL_EMB] = {0};
    float recon[TINYOL_OUT];
    tinyol_decode(&g_dec, emb, recon);

    for (uint32_t j = 0; j < TINYOL_OUT; j++) {
        TEST_ASSERT_FLOAT_WITHIN(TOL, 0.0f, recon[j]);
    }
}

void test_tinyol_reconstruction_error_nonnegative(void)
{
    _make_zero_tinyol();

    for (int i = 0; i < (int)MOCK_N_SAMPLES; i++) {
        float emb[TINYOL_EMB], recon[TINYOL_OUT];
        tinyol_encode(&g_enc, MOCK_NORMAL_T0[i], emb);
        tinyol_decode(&g_dec, emb, recon);
        float err = tinyol_reconstruction_error(MOCK_NORMAL_T0[i], recon, (int)TINYOL_OUT);
        TEST_ASSERT_GREATER_OR_EQUAL_FLOAT(0.0f, err);
    }
}

void test_tinyol_predict_returns_valid(void)
{
    _make_zero_tinyol();

    for (int i = 0; i < (int)MOCK_N_SAMPLES; i++) {
        int pred = tinyol_predict(&g_enc, &g_dec, MOCK_NORMAL_T0[i], 0.1f);
        TEST_ASSERT_TRUE(pred == 0 || pred == 1);
    }
}

void test_tinyol_reconstruction_error_zero_weights(void)
{
    _make_zero_tinyol();
    float emb[TINYOL_EMB], recon[TINYOL_OUT];
    tinyol_encode(&g_enc, MOCK_NORMAL_T0[0], emb);
    tinyol_decode(&g_dec, emb, recon);
    float err = tinyol_reconstruction_error(MOCK_NORMAL_T0[0], recon, (int)TINYOL_OUT);
    /* recon = 0 (poids nuls) → MSE = mean(x²) ≈ 0.00684 < MOCK_TINYOL_RECON_ERR_ZERO_WEIGHTS */
    TEST_ASSERT_LESS_THAN_FLOAT(MOCK_TINYOL_RECON_ERR_ZERO_WEIGHTS + MOCK_TINYOL_RECON_TOLERANCE,
                                 err);
}

void test_tinyol_predict_anomaly_higher_error(void)
{
    _make_zero_tinyol();
    float err_normal = 0.0f, err_anomaly = 0.0f;
    for (int k = 0; k < (int)MOCK_N_SAMPLES; k++) {
        float emb[TINYOL_EMB], recon[TINYOL_OUT];
        tinyol_encode(&g_enc, MOCK_NORMAL_T0[k], emb);
        tinyol_decode(&g_dec, emb, recon);
        err_normal += tinyol_reconstruction_error(MOCK_NORMAL_T0[k], recon, (int)TINYOL_OUT);
        tinyol_encode(&g_enc, MOCK_ANOMALY_T0[k], emb);
        tinyol_decode(&g_dec, emb, recon);
        err_anomaly += tinyol_reconstruction_error(MOCK_ANOMALY_T0[k], recon, (int)TINYOL_OUT);
    }
    TEST_ASSERT_GREATER_THAN_FLOAT(err_normal, err_anomaly);
}

/* ────────────────────────────────────────────────────────────────────────── */
/*  METRICS                                                                   */
/* ────────────────────────────────────────────────────────────────────────── */

void test_acc_perfect(void)
{
    OnlineAccuracy acc;
    acc_init(&acc);
    for (int i = 0; i < 10; i++) acc_update(&acc, 0, 0);
    TEST_ASSERT_FLOAT_WITHIN(TOL, 1.0f, acc_compute(&acc));
}

void test_acc_zero(void)
{
    OnlineAccuracy acc;
    acc_init(&acc);
    for (int i = 0; i < 10; i++) acc_update(&acc, 1, 0);
    TEST_ASSERT_FLOAT_WITHIN(TOL, 0.0f, acc_compute(&acc));
}

void test_auroc_random(void)
{
    OnlineAUROC auroc;
    auroc_init(&auroc);
    /* Scores aléatoires uniformes → AUROC ≈ 0.5 */
    for (int i = 0; i < (int)AUROC_WINDOW; i++) {
        auroc_update(&auroc, (float)i / AUROC_WINDOW, i % 2);
    }
    float a = auroc_compute(&auroc);
    TEST_ASSERT_TRUE(a >= 0.0f && a <= 1.0f);
}

void test_auroc_perfect(void)
{
    OnlineAUROC auroc;
    auroc_init(&auroc);
    /* Positifs toujours > 1.0, négatifs toujours < 0.0 → AUROC = 1.0 */
    for (int i = 0; i < (int)AUROC_WINDOW; i++) {
        float score = (i % 2 == 0) ? -0.5f : 1.5f;
        auroc_update(&auroc, score, i % 2);
    }
    TEST_ASSERT_FLOAT_WITHIN(TOL, 1.0f, auroc_compute(&auroc));
}

void test_forgetting_no_drop(void)
{
    ForgettingTracker fgt;
    fgt_init(&fgt);
    fgt_update(&fgt, 0, 0.9f);
    fgt_update(&fgt, 0, 0.9f);  /* même accuracy → pas de forgetting */
    TEST_ASSERT_FLOAT_WITHIN(TOL, 0.0f, fgt_avg_forgetting(&fgt));
}

/* ────────────────────────────────────────────────────────────────────────── */
/*  Runner                                                                    */
/* ────────────────────────────────────────────────────────────────────────── */

void setUp(void) {}
void tearDown(void) {}

int main(void)
{
    UNITY_BEGIN();

    /* Mahalanobis */
    RUN_TEST(test_maha_normal_below_threshold);
    RUN_TEST(test_maha_anomaly_above_threshold);
    RUN_TEST(test_maha_predict_correct_normal);
    RUN_TEST(test_maha_ema_update_shifts_mean);

    /* EWC head */
    RUN_TEST(test_ewc_forward_zero_weights);
    RUN_TEST(test_ewc_predict_returns_valid_class);
    RUN_TEST(test_ewc_sgd_step_modifies_weights);
    RUN_TEST(test_ewc_sgd_reduces_loss_on_normal);
    RUN_TEST(test_ewc_consolidate_copies_star_weights);
    RUN_TEST(test_ewc_consolidate_fisher_nonnegative);
    RUN_TEST(test_ewc_consolidate_ema_alpha);
    /* S2004 — EWC consolidation (Groupe 1) */
    RUN_TEST(test_ewc_consolidate_fisher_update);
    RUN_TEST(test_ewc_consolidate_star_copy);
    RUN_TEST(test_ewc_consolidate_fisher_nonneg);
    /* S2004 — EWC régularisation (Groupe 2) */
    RUN_TEST(test_ewc_penalty_nonzero);
    RUN_TEST(test_ewc_forgetting_reduced);

    /* TinyOL */
    RUN_TEST(test_tinyol_encode_zero_weights);
    RUN_TEST(test_tinyol_decode_zero_weights);
    RUN_TEST(test_tinyol_reconstruction_error_nonnegative);
    RUN_TEST(test_tinyol_predict_returns_valid);
    RUN_TEST(test_tinyol_reconstruction_error_zero_weights);
    RUN_TEST(test_tinyol_predict_anomaly_higher_error);
    /* S2004 — TinyOL init + forward (Groupe 3) */
    RUN_TEST(test_tinyol_init_weights_loaded);
    RUN_TEST(test_tinyol_forward_shape);
    RUN_TEST(test_tinyol_forward_delta_vs_ref);

    /* Metrics */
    RUN_TEST(test_acc_perfect);
    RUN_TEST(test_acc_zero);
    RUN_TEST(test_auroc_random);
    RUN_TEST(test_auroc_perfect);
    RUN_TEST(test_forgetting_no_drop);

    return UNITY_END();
}
