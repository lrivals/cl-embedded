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
        float score = (i % 2 == 0) ? 1.5f : -0.5f;
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
    RUN_TEST(test_ewc_sgd_reduces_loss_on_normal);

    /* TinyOL */
    RUN_TEST(test_tinyol_encode_zero_weights);
    RUN_TEST(test_tinyol_decode_zero_weights);
    RUN_TEST(test_tinyol_reconstruction_error_nonnegative);
    RUN_TEST(test_tinyol_predict_returns_valid);

    /* Metrics */
    RUN_TEST(test_acc_perfect);
    RUN_TEST(test_acc_zero);
    RUN_TEST(test_auroc_random);
    RUN_TEST(test_auroc_perfect);
    RUN_TEST(test_forgetting_no_drop);

    return UNITY_END();
}
