/**
 * test_runner.c — Point d'entrée Unity pour les tests embarqués (host x86).
 *
 * Compilation native uniquement (gcc x86) — pas de HAL STM32.
 * Ajout de tests : déclarer la fonction ici et appeler RUN_TEST().
 */

#include "unity.h"

/* setUp/tearDown centralisés ici pour éviter les définitions multiples */
void setUp(void) {}
void tearDown(void) {}

/* ── Déclarations — test_mahalanobis.c ──────────────────────────────────── */
void test_mahal_zero_distance(void);
void test_mahal_identity_is_euclidean(void);
void test_mahal_symmetric_input(void);
void test_mahal_predict_below_threshold(void);
void test_mahal_predict_above_threshold(void);
void test_mahal_ema_update_alpha_one(void);
void test_mahal_ema_update_alpha_zero(void);
void test_mahal_ema_update_convergence(void);
void test_mahal_custom_precision(void);

/* ── Déclarations — test_ewc_head.c ────────────────────────────────────── */
void test_ewc_forward_all_zeros_weights(void);
void test_ewc_forward_output_count(void);
void test_ewc_forward_deterministic(void);
void test_ewc_predict_valid_class(void);
void test_ewc_predict_zero_weights_returns_zero(void);
void test_ewc_sgd_step_decreases_loss(void);
void test_ewc_sgd_step_modifies_weights(void);

/* ── Main ───────────────────────────────────────────────────────────────── */

int main(void)
{
    UNITY_BEGIN();

    /* Mahalanobis */
    RUN_TEST(test_mahal_zero_distance);
    RUN_TEST(test_mahal_identity_is_euclidean);
    RUN_TEST(test_mahal_symmetric_input);
    RUN_TEST(test_mahal_predict_below_threshold);
    RUN_TEST(test_mahal_predict_above_threshold);
    RUN_TEST(test_mahal_ema_update_alpha_one);
    RUN_TEST(test_mahal_ema_update_alpha_zero);
    RUN_TEST(test_mahal_ema_update_convergence);
    RUN_TEST(test_mahal_custom_precision);

    /* EWC Head */
    RUN_TEST(test_ewc_forward_all_zeros_weights);
    RUN_TEST(test_ewc_forward_output_count);
    RUN_TEST(test_ewc_forward_deterministic);
    RUN_TEST(test_ewc_predict_valid_class);
    RUN_TEST(test_ewc_predict_zero_weights_returns_zero);
    RUN_TEST(test_ewc_sgd_step_decreases_loss);
    RUN_TEST(test_ewc_sgd_step_modifies_weights);

    return UNITY_END();
}
