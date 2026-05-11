/**
 * test_ewc_head.c — Tests unitaires Unity pour ewc_head.c
 *
 * Architecture : Input(5) → ReLU(32) → ReLU(16) → Output(2)
 * Tests :
 *   - forward poids zéro → logits nuls
 *   - predict retourne une classe valide {0, 1}
 *   - une étape SGD réduit la CE loss sur l'échantillon courant
 */

#include "unity.h"
#include "ewc_head.h"
#include <math.h>
#include <string.h>

#define TOL 1e-5f

/* Retourne une EWCHead avec tous les paramètres à zéro. */
static EWCHead make_zero_head(void)
{
    EWCHead h;
    memset(&h, 0, sizeof(h));
    return h;
}

/*
 * EWCHead avec des petits poids non nuls pour que les gradients passent
 * les couches ReLU. Utilise une initialisation déterministe simple.
 */
static EWCHead make_small_weights_head(void)
{
    EWCHead h;
    memset(&h, 0, sizeof(h));
    h.lambda = 0.0f;  /* pas de régularisation EWC pour ces tests */

    /* w1 : matrice diagonale tronquée (min(H1,IN) = 5 éléments actifs) */
    for (int i = 0; i < EWC_IN; i++) {
        h.w1[i][i] = 0.1f;
    }
    /* Biais couche 1 : légèrement positif pour que ReLU laisse passer */
    for (int j = 0; j < EWC_H1; j++) {
        h.b1[j] = 0.05f;
    }
    /* w2 : diagonale tronquée */
    for (int i = 0; i < EWC_H2; i++) {
        h.w2[i][i] = 0.1f;
    }
    for (int j = 0; j < EWC_H2; j++) {
        h.b2[j] = 0.05f;
    }
    /* w3 : sortie asymétrique pour que logits[0] ≠ logits[1] */
    for (int i = 0; i < EWC_H2; i++) {
        h.w3[0][i] = 0.1f;
        h.w3[1][i] = -0.1f;
    }

    return h;
}

/* Calcule la CE loss softmax sur les logits pour la classe `label`. */
static float ce_loss(const float *logits, int n_out, int label)
{
    float max_l = logits[0];
    for (int j = 1; j < n_out; j++) {
        if (logits[j] > max_l) max_l = logits[j];
    }
    float sum_exp = 0.0f;
    for (int j = 0; j < n_out; j++) {
        sum_exp += expf(logits[j] - max_l);
    }
    return -(logits[label] - max_l) + logf(sum_exp);
}

/* ── Tests forward ──────────────────────────────────────────────────────── */

void test_ewc_forward_all_zeros_weights(void)
{
    /* Poids et biais = 0 → ReLU(0) = 0 partout → logits = [0, 0] */
    EWCHead h = make_zero_head();
    float x[EWC_IN] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float out[EWC_OUT];
    ewc_forward(&h, x, out);

    for (int j = 0; j < EWC_OUT; j++) {
        TEST_ASSERT_FLOAT_WITHIN(TOL, 0.0f, out[j]);
    }
}

void test_ewc_forward_output_count(void)
{
    /* Vérifie que la forward produit exactement EWC_OUT = 2 valeurs */
    EWCHead h = make_small_weights_head();
    float x[EWC_IN] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float out[EWC_OUT] = {-999.0f, -999.0f};
    ewc_forward(&h, x, out);

    /* Les deux logits doivent avoir été écrits (plus -999) */
    for (int j = 0; j < EWC_OUT; j++) {
        TEST_ASSERT_NOT_EQUAL(-999.0f, out[j]);
    }
}

void test_ewc_forward_deterministic(void)
{
    /* Deux appels identiques → même sortie */
    EWCHead h = make_small_weights_head();
    float x[EWC_IN] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    float out1[EWC_OUT], out2[EWC_OUT];
    ewc_forward(&h, x, out1);
    ewc_forward(&h, x, out2);

    for (int j = 0; j < EWC_OUT; j++) {
        TEST_ASSERT_FLOAT_WITHIN(TOL, out1[j], out2[j]);
    }
}

/* ── Tests predict ──────────────────────────────────────────────────────── */

void test_ewc_predict_valid_class(void)
{
    EWCHead h = make_small_weights_head();
    float x[EWC_IN] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    int pred = ewc_predict(&h, x);
    TEST_ASSERT(pred == 0 || pred == 1);
}

void test_ewc_predict_zero_weights_returns_zero(void)
{
    /* Poids nuls → logits = [0,0] → argmax = 0 (premier par convention) */
    EWCHead h = make_zero_head();
    float x[EWC_IN] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    int pred = ewc_predict(&h, x);
    TEST_ASSERT_EQUAL_INT(0, pred);
}

/* ── Tests SGD step ─────────────────────────────────────────────────────── */

void test_ewc_sgd_step_decreases_loss(void)
{
    /*
     * Une étape SGD doit réduire la CE loss sur l'échantillon courant.
     * Initialisation : petits poids non nuls, lambda = 0 (pas EWC).
     */
    EWCHead h = make_small_weights_head();
    float x[EWC_IN] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    int label = 0;

    float logits_before[EWC_OUT];
    ewc_forward(&h, x, logits_before);
    float loss_before = ce_loss(logits_before, EWC_OUT, label);

    ewc_sgd_step(&h, x, label);

    float logits_after[EWC_OUT];
    ewc_forward(&h, x, logits_after);
    float loss_after = ce_loss(logits_after, EWC_OUT, label);

    TEST_ASSERT_LESS_THAN_FLOAT(loss_before, loss_after);
}

void test_ewc_sgd_step_modifies_weights(void)
{
    /* Une step SGD doit changer au moins un poids */
    EWCHead h = make_small_weights_head();
    float w1_00_before = h.w1[0][0];
    float x[EWC_IN] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ewc_sgd_step(&h, x, 0);
    /* w1[0][0] doit avoir changé (gradient non nul sur cet élément) */
    TEST_ASSERT_NOT_EQUAL(w1_00_before, h.w1[0][0]);
}
