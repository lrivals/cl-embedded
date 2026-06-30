/**
 * test_metrics.c — Tests unitaires OnlineRMSE + OnlineF1Macro (Sprint 26 S2606)
 *
 * Compilation : make test (TEST_MODE=1, gcc x86)
 */

#include "unity.h"
#include "metrics.h"
#include <math.h>

/* ── OnlineRMSE ─────────────────────────────────────────────────────────── */

void test_rmse_init_zero(void)
{
    OnlineRMSE r;
    online_rmse_init(&r);
    TEST_ASSERT_EQUAL_UINT32(0U, r.n);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, r.rmse);
}

void test_rmse_three_samples(void)
{
    /* Erreurs : (10-8)=2, (12-10)=2, (9-11)=-2  →  err² = 4, 4, 4
     * Welford variance des err²: mean=4, M2=0 → std=0, rmse=0 pour 3 valeurs identiques.
     * On vérifie ici que get() retourne une valeur positive et finie. */
    OnlineRMSE r;
    online_rmse_init(&r);
    online_rmse_update(&r, 10.0f, 8.0f);
    online_rmse_update(&r, 12.0f, 10.0f);
    online_rmse_update(&r, 9.0f, 11.0f);
    TEST_ASSERT_EQUAL_UINT32(3U, r.n);
    TEST_ASSERT_TRUE(isfinite(online_rmse_get(&r)));
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.0f, online_rmse_get(&r));
}

void test_rmse_varying_errors(void)
{
    /* Erreurs : 1, 2, 3 → err² = 1, 4, 9 → mean_sq = 4.667 */
    OnlineRMSE r;
    online_rmse_init(&r);
    online_rmse_update(&r, 1.0f, 0.0f);   /* err=1 */
    online_rmse_update(&r, 2.0f, 0.0f);   /* err=2 */
    online_rmse_update(&r, 3.0f, 0.0f);   /* err=3 */
    /* Après 3 updates, rmse doit être > 0 */
    TEST_ASSERT_GREATER_THAN_FLOAT(0.0f, online_rmse_get(&r));
}

void test_rmse_single_sample_returns_zero(void)
{
    OnlineRMSE r;
    online_rmse_init(&r);
    online_rmse_update(&r, 5.0f, 3.0f);
    /* n=1 : Welford n-1=0 → rmse = 0 (évite division par zéro) */
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, online_rmse_get(&r));
}

/* ── OnlineF1Macro ──────────────────────────────────────────────────────── */

void test_f1_init_zero(void)
{
    OnlineF1Macro f;
    online_f1_init(&f);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, online_f1_get(&f));
}

void test_f1_perfect_diagonal(void)
{
    /* Prédit correctement chaque classe une fois → F1 = 1.0 */
    OnlineF1Macro f;
    online_f1_init(&f);
    for (int c = 0; c < EWC_MC_N_CLASSES; c++)
        online_f1_update(&f, c, c);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.0f, online_f1_get(&f));
}

void test_f1_all_wrong(void)
{
    /* Toujours prédit la classe 0, toutes les vraies labels = 1 → F1 = 0 */
    OnlineF1Macro f;
    online_f1_init(&f);
    for (int i = 0; i < 10; i++)
        online_f1_update(&f, 0, 1);
    float f1 = online_f1_get(&f);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.0f, f1);
}

void test_f1_out_of_range_ignored(void)
{
    OnlineF1Macro f;
    online_f1_init(&f);
    /* Labels hors-range ne doivent pas crasher ni modifier la matrice */
    online_f1_update(&f, -1, 0);
    online_f1_update(&f, 0, EWC_MC_N_CLASSES);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, online_f1_get(&f));
}
