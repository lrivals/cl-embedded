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

/* ── Tests ewc_consolidate ──────────────────────────────────────────────── */

void test_ewc_consolidate_fisher_ema(void)
{
    /* alpha=0.9, fisher initial=0 → après consolidate : fisher ≈ 0.1 * w² */
    EWCHead h = make_small_weights_head();
    ewc_consolidate(&h, 0.9f);

    for (int j = 0; j < EWC_H1; j++) {
        for (int i = 0; i < EWC_IN; i++) {
            float expected = 0.1f * h.star_w1[j][i] * h.star_w1[j][i];
            TEST_ASSERT_FLOAT_WITHIN(TOL, expected, h.fisher1[j][i]);
        }
    }
}

void test_ewc_consolidate_star_w_copied(void)
{
    /* Après consolidation, star_w doit être une copie exacte de w */
    EWCHead h = make_small_weights_head();
    ewc_consolidate(&h, 0.9f);

    for (int j = 0; j < EWC_H1; j++) {
        for (int i = 0; i < EWC_IN; i++) {
            TEST_ASSERT_FLOAT_WITHIN(TOL, h.w1[j][i], h.star_w1[j][i]);
        }
    }
    for (int j = 0; j < EWC_H2; j++) {
        for (int i = 0; i < EWC_H1; i++) {
            TEST_ASSERT_FLOAT_WITHIN(TOL, h.w2[j][i], h.star_w2[j][i]);
        }
    }
}

void test_ewc_consolidate_fisher_nonneg(void)
{
    /* La diagonale Fisher (grad²) est toujours ≥ 0 */
    EWCHead h = make_small_weights_head();
    ewc_consolidate(&h, 0.5f);

    for (int j = 0; j < EWC_H1; j++) {
        for (int i = 0; i < EWC_IN; i++) {
            TEST_ASSERT_GREATER_OR_EQUAL_FLOAT(0.0f, h.fisher1[j][i]);
        }
    }
    for (int j = 0; j < EWC_H2; j++) {
        for (int i = 0; i < EWC_H1; i++) {
            TEST_ASSERT_GREATER_OR_EQUAL_FLOAT(0.0f, h.fisher2[j][i]);
        }
    }
    for (int j = 0; j < EWC_OUT; j++) {
        for (int i = 0; i < EWC_H2; i++) {
            TEST_ASSERT_GREATER_OR_EQUAL_FLOAT(0.0f, h.fisher3[j][i]);
        }
    }
}

void test_ewc_penalty_active(void)
{
    /* Scénario 2 tâches : consolider tâche 0 → lambda > 0 → step tâche 1.
     * Le terme EWC doit produire un delta de poids différent de sans régularisation. */
    EWCHead h_no_ewc = make_small_weights_head();
    EWCHead h_ewc    = make_small_weights_head();
    float x[EWC_IN]  = {1.0f, 0.5f, 0.0f, 0.0f, 0.0f};
    int label = 1;

    /* Perturber avant consolidation pour que Fisher[0][0] = 0.1*(0.15)² > 0 */
    h_ewc.w1[0][0] += 0.05f;
    ewc_consolidate(&h_ewc, 0.9f);   /* star_w1[0][0] = 0.15, Fisher > 0 */

    /* Perturber après consolidation pour créer l'écart (w - star_w) ≠ 0 */
    h_ewc.w1[0][0] += 0.05f;
    h_ewc.lambda = 400.0f;

    float w_ewc_before    = h_ewc.w1[0][0];
    float w_no_ewc_before = h_no_ewc.w1[0][0];

    ewc_sgd_step(&h_ewc,    x, label);
    ewc_sgd_step(&h_no_ewc, x, label);

    float delta_ewc    = h_ewc.w1[0][0]    - w_ewc_before;
    float delta_no_ewc = h_no_ewc.w1[0][0] - w_no_ewc_before;

    /* La pénalité EWC doit modifier le gradient → deltas différents */
    TEST_ASSERT(delta_ewc != delta_no_ewc);
}

/* ── Tests ewc_init ─────────────────────────────────────────────────────── */

void test_ewc_init_weights_nonzero(void)
{
    /* Xavier LCG seed=42 → poids non nuls (évite le problème de symétrie) */
    EWCHead h;
    memset(&h, 0, sizeof(h));
    h.lambda = 100.0f;   /* ne doit pas être modifié par ewc_init */
    ewc_init(&h);

    /* Au moins un poids w1 doit être non nul */
    int any_nonzero = 0;
    for (int j = 0; j < EWC_H1 && !any_nonzero; j++)
        for (int i = 0; i < EWC_IN; i++)
            if (h.w1[j][i] != 0.0f) { any_nonzero = 1; break; }
    TEST_ASSERT_TRUE(any_nonzero);

    /* lambda ne doit pas avoir été modifié */
    TEST_ASSERT_FLOAT_WITHIN(TOL, 100.0f, h.lambda);
}

void test_ewc_init_fisher_zero(void)
{
    /* ewc_init doit remettre Fisher et star_w à zéro */
    EWCHead h = make_small_weights_head();
    ewc_consolidate(&h, 0.9f);   /* fisher et star_w non nuls */
    ewc_init(&h);

    for (int j = 0; j < EWC_H1; j++)
        for (int i = 0; i < EWC_IN; i++) {
            TEST_ASSERT_FLOAT_WITHIN(TOL, 0.0f, h.fisher1[j][i]);
            TEST_ASSERT_FLOAT_WITHIN(TOL, 0.0f, h.star_w1[j][i]);
        }
}

void test_ewc_init_deterministic(void)
{
    /* Deux appels successifs → poids identiques (LCG déterministe seed=42) */
    EWCHead h1, h2;
    memset(&h1, 0, sizeof(h1));
    memset(&h2, 0, sizeof(h2));
    ewc_init(&h1);
    ewc_init(&h2);

    for (int j = 0; j < EWC_H1; j++)
        for (int i = 0; i < EWC_IN; i++)
            TEST_ASSERT_FLOAT_WITHIN(TOL, h1.w1[j][i], h2.w1[j][i]);
}
