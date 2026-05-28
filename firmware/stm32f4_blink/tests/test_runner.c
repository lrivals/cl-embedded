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
void test_ewc_consolidate_fisher_ema(void);
void test_ewc_consolidate_star_w_copied(void);
void test_ewc_consolidate_fisher_nonneg(void);
void test_ewc_penalty_active(void);
void test_ewc_init_weights_nonzero(void);
void test_ewc_init_fisher_zero(void);
void test_ewc_init_deterministic(void);

/* ── Déclarations — test_profiling.c ───────────────────────────────────── */
void test_profiling_latency_positive(void);
void test_profiling_latency_zero_cycles(void);
void test_profiling_throughput_nonzero(void);
void test_profiling_encode_format(void);
void test_profiling_encode_size(void);
void test_bss_size_within_limit(void);

/* ── Déclarations — test_pipeline.c ────────────────────────────────────── */
void test_pipeline_response_v3_21bytes(void);
void test_protocol_v3_fields(void);
void test_pipeline_debug_printf_contains_score(void);
void test_pipeline_response_v2_14bytes(void);
void test_pipeline_consolidate_flag(void);

/* ── Déclarations — test_hdc.c ──────────────────────────────────────────── */
void test_hdc_encode_norm(void);
void test_hdc_predict_label(void);

/* ── Déclarations — test_tinyol.c ───────────────────────────────────────── */
void test_tinyol_encode_zero_weights(void);
void test_tinyol_decode_zero_emb(void);
void test_tinyol_recon_error_zero_weights(void);
void test_tinyol_predict_anomaly_zero_weights(void);
void test_tinyol_predict_normal_zero_weights(void);
void test_tinyol_forward_deterministic(void);
void test_tinyol_init_loads_constants(void);
void test_tinyol_forward_delta(void);

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
    RUN_TEST(test_ewc_consolidate_fisher_ema);
    RUN_TEST(test_ewc_consolidate_star_w_copied);
    RUN_TEST(test_ewc_consolidate_fisher_nonneg);
    RUN_TEST(test_ewc_penalty_active);
    RUN_TEST(test_ewc_init_weights_nonzero);
    RUN_TEST(test_ewc_init_fisher_zero);
    RUN_TEST(test_ewc_init_deterministic);

    /* Profiling */
    RUN_TEST(test_profiling_latency_positive);
    RUN_TEST(test_profiling_latency_zero_cycles);
    RUN_TEST(test_profiling_throughput_nonzero);
    RUN_TEST(test_profiling_encode_format);
    RUN_TEST(test_profiling_encode_size);
    RUN_TEST(test_bss_size_within_limit);

    /* Pipeline */
    RUN_TEST(test_pipeline_response_v3_21bytes);
    RUN_TEST(test_protocol_v3_fields);
    RUN_TEST(test_pipeline_debug_printf_contains_score);
    RUN_TEST(test_pipeline_response_v2_14bytes);
    RUN_TEST(test_pipeline_consolidate_flag);

    /* HDC */
    RUN_TEST(test_hdc_encode_norm);
    RUN_TEST(test_hdc_predict_label);

    /* TinyOL */
    RUN_TEST(test_tinyol_encode_zero_weights);
    RUN_TEST(test_tinyol_decode_zero_emb);
    RUN_TEST(test_tinyol_recon_error_zero_weights);
    RUN_TEST(test_tinyol_predict_anomaly_zero_weights);
    RUN_TEST(test_tinyol_predict_normal_zero_weights);
    RUN_TEST(test_tinyol_forward_deterministic);
    RUN_TEST(test_tinyol_init_loads_constants);
    RUN_TEST(test_tinyol_forward_delta);

    return UNITY_END();
}
