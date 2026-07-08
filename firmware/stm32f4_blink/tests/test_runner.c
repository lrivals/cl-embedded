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

/* ── Déclarations — test_mahalanobis_q15.c (Sprint 34, S3409) ────────────── */
void test_maha_q15_parity_with_python(void);
void test_maha_q15_init_is_euclidean(void);
void test_maha_q15_zero_distance(void);
void test_maha_q15_predict_threshold(void);

/* ── Déclarations — test_mahalanobis_int8.c (Sprint 29, S2912) ───────────── */
void test_maha_int8_parity_with_python(void);
void test_maha_int8_init_is_euclidean(void);
void test_maha_int8_zero_distance(void);
void test_maha_int8_predict_threshold(void);

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

/* ── Déclarations — test_ewc_int8_v2.c (Sprint 39, S3909) ───────────────── */
void test_v2_no_overflow(void);
void test_v2_parity_emulator(void);
void test_v2_q15_parity(void);
void test_v2_recovers_f1(void);
void test_v1_unchanged(void);

/* ── Déclarations — test_profiling.c ───────────────────────────────────── */
void test_profiling_latency_positive(void);
void test_profiling_latency_zero_cycles(void);
void test_profiling_throughput_nonzero(void);
void test_profiling_encode_format(void);
void test_profiling_encode_size(void);
void test_bss_size_within_limit(void);
void test_stack_peak_partial_usage(void);
void test_stack_peak_untouched_is_zero(void);
void test_stack_peak_fully_used(void);
void test_stack_peak_repeated_constant_not_masked(void);

/* ── Déclarations — test_pipeline.c ────────────────────────────────────── */
void test_pipeline_response_v3_23bytes(void);
void test_protocol_v3_fields(void);
void test_pipeline_debug_printf_contains_score(void);
void test_pipeline_response_v2_14bytes(void);
void test_pipeline_consolidate_flag(void);

/* ── Déclarations — test_ewc_int8.c ─────────────────────────────────────── */
void test_int8_forward_close_to_fp32(void);
void test_int8_update_does_not_crash(void);
void test_int8_consolidate_updates_star_w(void);
void test_int8_relu_q7_clamps_negative(void);
void test_int8_float_q7_roundtrip(void);

/* ── Déclarations — test_hdc.c ──────────────────────────────────────────── */
void test_hdc_encode_norm(void);
void test_hdc_predict_label(void);
void test_hdc_update_accumulates(void);
void test_hdc_binarize_preserves_sign(void);
void test_hdc_binarize_norm_is_hdcdim(void);
void test_hdc_n_trained_increments(void);
void test_hdc_update_with_sample_fills_buffer(void);
void test_hdc_buf_wraps_at_retrain_buf(void);
void test_hdc_retrain_predicts_correct_after_reset(void);
void test_hdc_init_zeros_all_fields(void);

/* ── Déclarations — test_ring_buffer.c (S3402) ──────────────────────────── */
void test_ring_buffer_push_increments(void);
void test_ring_buffer_count_caps_at_capacity(void);
void test_ring_buffer_wraps_at_capacity(void);
void test_ring_buffer_is_full(void);
void test_ring_buffer_window_fifo_order(void);
void test_ring_buffer_window_after_wrap(void);
void test_ring_buffer_window_stride(void);
void test_ring_buffer_window_size_limit(void);
void test_ring_buffer_multibyte_elem(void);

/* ── Déclarations — test_drift_detector.c (S3803) ───────────────────────── */
void test_drift_normal_below_thresholds(void);
void test_drift_fault_instantaneous(void);
void test_drift_fault_priority_over_drift(void);
void test_drift_triggers_on_ratio(void);
void test_drift_reset_clears_window(void);
void test_drift_sequence_parity_python(void);

/* ── Déclarations — test_tinyol.c ───────────────────────────────────────── */
void test_tinyol_encode_zero_weights(void);
void test_tinyol_decode_zero_emb(void);
void test_tinyol_recon_error_zero_weights(void);
void test_tinyol_predict_anomaly_zero_weights(void);
void test_tinyol_predict_normal_zero_weights(void);
void test_tinyol_forward_deterministic(void);
void test_tinyol_init_loads_constants(void);
void test_tinyol_forward_delta(void);

/* ── Déclarations — test_metrics.c (S2606) ──────────────────────────────── */
void test_rmse_init_zero(void);
void test_rmse_three_samples(void);
void test_rmse_varying_errors(void);
void test_rmse_single_sample_returns_zero(void);
void test_f1_init_zero(void);
void test_f1_perfect_diagonal(void);
void test_f1_all_wrong(void);
void test_f1_out_of_range_ignored(void);

/* ── Déclarations — test_ewc_regression.c (S2614) ──────────────────────── */
void test_ewc_reg_forward_scalar(void);
void test_ewc_reg_sgd_reduces_error(void);
void test_ewc_reg_penalty_zero_before_consolidate(void);
void test_ewc_reg_penalty_nonzero_after_consolidate(void);
void test_ewc_reg_consolidate_copies_weights(void);

/* ── Déclarations — test_ewc_multiclass.c (S2615) ──────────────────────── */
void test_ewc_mc_forward_valid_logits(void);
void test_ewc_mc_softmax_sums_to_one(void);
void test_ewc_mc_predict_argmax(void);
void test_ewc_mc_penalty_zero_before_consolidate(void);
void test_ewc_mc_penalty_nonzero_after_consolidate(void);

/* ── Déclarations — test_pipeline.c Sprint 27 DUAL_MODE ─────────────────── */
void test_pipeline_response_dual_25bytes(void);
void test_pipeline_dual_response_fields(void);
void test_pipeline_dual_mode_dispatch(void);
void test_pipeline_dual_mode_update(void);

/* ── Déclarations — test_pipeline.c Sprint 30 PAIR_MODE ─────────────────── */
void test_pipeline_response_pair_22bytes(void);
void test_pipeline_pair_response_fields(void);
void test_pipeline_pair_mode_dispatch(void);

/* ── Déclarations — test_hdc_int8.c (S2906) ─────────────────────────────── */
void test_hdc_int8_init_zeros_am(void);
void test_hdc_int8_encode_bipolar(void);
void test_hdc_int8_predict_after_updates(void);
void test_hdc_int8_update_accumulates(void);
void test_hdc_int8_sizeof(void);

/* ── Déclarations — test_tinyol_int8.c (S2907) ──────────────────────────── */
void test_tinyol_int8_encode_range(void);
void test_tinyol_int8_encode_vs_fp32(void);
void test_oto_int8_predict_returns_binary(void);
void test_oto_int8_update_learns_class1(void);
void test_oto_int8_last_prob_range(void);

/* ── Déclarations — test_meta_head.c (S3112) ────────────────────────────── */
void test_meta_init_loads_weights(void);
void test_meta_forward_parity_python(void);
void test_meta_forward_in_unit_range(void);
void test_meta_predict_binary(void);

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

    /* test_mahalanobis_q15.c (Sprint 34, S3409) */
    RUN_TEST(test_maha_q15_parity_with_python);
    RUN_TEST(test_maha_q15_init_is_euclidean);
    RUN_TEST(test_maha_q15_zero_distance);
    RUN_TEST(test_maha_q15_predict_threshold);

    /* test_mahalanobis_int8.c (Sprint 29, S2912) */
    RUN_TEST(test_maha_int8_parity_with_python);
    RUN_TEST(test_maha_int8_init_is_euclidean);
    RUN_TEST(test_maha_int8_zero_distance);
    RUN_TEST(test_maha_int8_predict_threshold);

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
    RUN_TEST(test_stack_peak_partial_usage);
    RUN_TEST(test_stack_peak_untouched_is_zero);
    RUN_TEST(test_stack_peak_fully_used);
    RUN_TEST(test_stack_peak_repeated_constant_not_masked);

    /* Pipeline */
    RUN_TEST(test_pipeline_response_v3_23bytes);
    RUN_TEST(test_protocol_v3_fields);
    RUN_TEST(test_pipeline_debug_printf_contains_score);
    RUN_TEST(test_pipeline_response_v2_14bytes);
    RUN_TEST(test_pipeline_consolidate_flag);

    /* EWC INT8 */
    RUN_TEST(test_int8_forward_close_to_fp32);
    RUN_TEST(test_int8_update_does_not_crash);
    RUN_TEST(test_int8_consolidate_updates_star_w);
    RUN_TEST(test_int8_relu_q7_clamps_negative);
    RUN_TEST(test_int8_float_q7_roundtrip);

    /* EWC INT8 v2 — Sprint 39, S3909 */
    RUN_TEST(test_v2_no_overflow);
    RUN_TEST(test_v2_parity_emulator);
    RUN_TEST(test_v2_q15_parity);
    RUN_TEST(test_v2_recovers_f1);
    RUN_TEST(test_v1_unchanged);

    /* HDC */
    RUN_TEST(test_hdc_encode_norm);
    RUN_TEST(test_hdc_predict_label);
    RUN_TEST(test_hdc_update_accumulates);
    RUN_TEST(test_hdc_binarize_preserves_sign);
    RUN_TEST(test_hdc_binarize_norm_is_hdcdim);
    RUN_TEST(test_hdc_n_trained_increments);
    RUN_TEST(test_hdc_update_with_sample_fills_buffer);
    RUN_TEST(test_hdc_buf_wraps_at_retrain_buf);
    RUN_TEST(test_hdc_retrain_predicts_correct_after_reset);
    RUN_TEST(test_hdc_init_zeros_all_fields);

    /* Ring buffer générique — S3402 */
    RUN_TEST(test_ring_buffer_push_increments);
    RUN_TEST(test_ring_buffer_count_caps_at_capacity);
    RUN_TEST(test_ring_buffer_wraps_at_capacity);
    RUN_TEST(test_ring_buffer_is_full);
    RUN_TEST(test_ring_buffer_window_fifo_order);
    RUN_TEST(test_ring_buffer_window_after_wrap);
    RUN_TEST(test_ring_buffer_window_stride);
    RUN_TEST(test_ring_buffer_window_size_limit);
    RUN_TEST(test_ring_buffer_multibyte_elem);

    RUN_TEST(test_drift_normal_below_thresholds);
    RUN_TEST(test_drift_fault_instantaneous);
    RUN_TEST(test_drift_fault_priority_over_drift);
    RUN_TEST(test_drift_triggers_on_ratio);
    RUN_TEST(test_drift_reset_clears_window);
    RUN_TEST(test_drift_sequence_parity_python);

    /* TinyOL */
    RUN_TEST(test_tinyol_encode_zero_weights);
    RUN_TEST(test_tinyol_decode_zero_emb);
    RUN_TEST(test_tinyol_recon_error_zero_weights);
    RUN_TEST(test_tinyol_predict_anomaly_zero_weights);
    RUN_TEST(test_tinyol_predict_normal_zero_weights);
    RUN_TEST(test_tinyol_forward_deterministic);
    RUN_TEST(test_tinyol_init_loads_constants);
    RUN_TEST(test_tinyol_forward_delta);

    /* Métriques S2606 — OnlineRMSE + OnlineF1Macro */
    RUN_TEST(test_rmse_init_zero);
    RUN_TEST(test_rmse_three_samples);
    RUN_TEST(test_rmse_varying_errors);
    RUN_TEST(test_rmse_single_sample_returns_zero);
    RUN_TEST(test_f1_init_zero);
    RUN_TEST(test_f1_perfect_diagonal);
    RUN_TEST(test_f1_all_wrong);
    RUN_TEST(test_f1_out_of_range_ignored);

    /* EWC Régression S2614 */
    RUN_TEST(test_ewc_reg_forward_scalar);
    RUN_TEST(test_ewc_reg_sgd_reduces_error);
    RUN_TEST(test_ewc_reg_penalty_zero_before_consolidate);
    RUN_TEST(test_ewc_reg_penalty_nonzero_after_consolidate);
    RUN_TEST(test_ewc_reg_consolidate_copies_weights);

    /* EWC Multi-class S2615 */
    RUN_TEST(test_ewc_mc_forward_valid_logits);
    RUN_TEST(test_ewc_mc_softmax_sums_to_one);
    RUN_TEST(test_ewc_mc_predict_argmax);
    RUN_TEST(test_ewc_mc_penalty_zero_before_consolidate);
    RUN_TEST(test_ewc_mc_penalty_nonzero_after_consolidate);

    /* Sprint 27 — DUAL_MODE tests T76–T79 */
    RUN_TEST(test_pipeline_response_dual_25bytes);
    RUN_TEST(test_pipeline_dual_response_fields);
    RUN_TEST(test_pipeline_dual_mode_dispatch);
    RUN_TEST(test_pipeline_dual_mode_update);

    /* Sprint 30 — PAIR_MODE (Mahalanobis + supervisé) */
    RUN_TEST(test_pipeline_response_pair_22bytes);
    RUN_TEST(test_pipeline_pair_response_fields);
    RUN_TEST(test_pipeline_pair_mode_dispatch);

    /* HDC INT8 — S2906 */
    RUN_TEST(test_hdc_int8_init_zeros_am);
    RUN_TEST(test_hdc_int8_encode_bipolar);
    RUN_TEST(test_hdc_int8_predict_after_updates);
    RUN_TEST(test_hdc_int8_update_accumulates);
    RUN_TEST(test_hdc_int8_sizeof);

    /* TinyOL INT8 — S2907 */
    RUN_TEST(test_tinyol_int8_encode_range);
    RUN_TEST(test_tinyol_int8_encode_vs_fp32);
    RUN_TEST(test_oto_int8_predict_returns_binary);
    RUN_TEST(test_oto_int8_update_learns_class1);
    RUN_TEST(test_oto_int8_last_prob_range);

    /* Méta-modèle de stacking — S3112 */
    RUN_TEST(test_meta_init_loads_weights);
    RUN_TEST(test_meta_forward_parity_python);
    RUN_TEST(test_meta_forward_in_unit_range);
    RUN_TEST(test_meta_predict_binary);

    return UNITY_END();
}
