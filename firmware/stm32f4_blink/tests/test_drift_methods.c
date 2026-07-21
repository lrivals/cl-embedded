/**
 * test_drift_methods.c — Tests unitaires Unity pour les détecteurs de drift portés
 *                        (Sprint 45 S4502 : page_hinkley.c, ddm.c, psi.c)
 *
 * Vérifie la parité bit-à-bit avec les références Python (src/models/drift/*.py) sur des
 * séquences connues dont les verdicts attendus sont produits par le Python lui-même
 * (scripts one-shot, cf. S4502) :
 *   - Page-Hinkley : DRIFT au bon indice sur un saut de moyenne
 *   - DDM          : franchit 2σ (WARNING) puis 3σ (DRIFT) sur un flux d'erreur
 *   - PSI          : DRIFT à la fin du bloc quand la distribution s'effondre dans un bac
 *   - *_reset      : remet l'état à zéro
 *
 * Footgun S3803 : les structs à état ne sont PAS copiables sans risque (comme g_drift du
 * firmware) → on les initialise TOUJOURS EN PLACE via un pointeur, jamais par valeur.
 */

#include "unity.h"
#include "drift/page_hinkley.h"
#include "drift/ddm.h"
#include "drift/psi.h"

/* ── Page-Hinkley ──────────────────────────────────────────────────────────── */

void test_ph_drift_on_mean_shift(void)
{
    /* delta=0.0, lambda=5.0, min_instances=5 ; saut 0→3 après 8 échantillons.
     * Verdicts identiques à PageHinkley.update (page_hinkley.py). */
    PageHinkley d; ph_init(&d, 0.0f, 5.0f, 5);
    const float seq[]        = {0,0,0,0,0,0,0,0, 3,3,3,3,3,3,3,3};
    const DriftMethodVerdict exp[] = {
        DM_NORMAL, DM_NORMAL, DM_NORMAL, DM_NORMAL, DM_NORMAL, DM_NORMAL, DM_NORMAL, DM_NORMAL,
        DM_NORMAL, DM_DRIFT,  DM_NORMAL, DM_NORMAL, DM_NORMAL, DM_NORMAL, DM_NORMAL, DM_NORMAL };
    int n = (int)(sizeof(seq)/sizeof(seq[0]));
    for (int i = 0; i < n; i++)
        TEST_ASSERT_EQUAL_INT(exp[i], ph_update(&d, seq[i]));
}

void test_ph_reset_clears_state(void)
{
    PageHinkley d; ph_init(&d, 0.0f, 5.0f, 5);
    for (int i = 0; i < 6; i++) ph_update(&d, 3.0f);
    ph_reset(&d);
    TEST_ASSERT_EQUAL_UINT32(0, d.n);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, d.cum_sum);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, d.min_sum);
    /* Après reset, un unique échantillon (< min_instances) ne peut pas déclencher. */
    TEST_ASSERT_EQUAL_INT(DM_NORMAL, ph_update(&d, 100.0f));
}

/* ── DDM ───────────────────────────────────────────────────────────────────── */

void test_ddm_warning_then_drift(void)
{
    /* warning=2.0, drift=3.0, min_instances=5 ; flux d'erreur graduel.
     * Séquence + verdicts issus de DDM.update (ddm.py) — couvre WARNING et DRIFT. */
    DDM d; ddm_init(&d, 2.0f, 3.0f, 5);
    const float seq[] = {0,0,0,0,0,1,0,0,1,0, 1,0,1,1,0,1,1,1,0,1, 1,1,1,1,1};
    const DriftMethodVerdict exp[] = {
        DM_NORMAL, DM_NORMAL, DM_NORMAL, DM_NORMAL, DM_NORMAL,
        DM_DRIFT,  DM_NORMAL, DM_NORMAL, DM_NORMAL, DM_NORMAL,
        DM_NORMAL, DM_NORMAL, DM_NORMAL, DM_NORMAL, DM_NORMAL,
        DM_NORMAL, DM_WARNING, DM_WARNING, DM_NORMAL, DM_WARNING,
        DM_WARNING, DM_WARNING, DM_WARNING, DM_WARNING, DM_WARNING };
    int n = (int)(sizeof(seq)/sizeof(seq[0]));
    for (int i = 0; i < n; i++)
        TEST_ASSERT_EQUAL_INT(exp[i], ddm_update(&d, seq[i]));
}

void test_ddm_reset_restores_init(void)
{
    DDM d; ddm_init(&d, 2.0f, 3.0f, 5);
    for (int i = 0; i < 6; i++) ddm_update(&d, 1.0f);
    ddm_reset(&d);
    TEST_ASSERT_EQUAL_UINT32(1, d.n);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, d.p);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, d.s);
}

/* ── PSI ───────────────────────────────────────────────────────────────────── */

/* Référence figée par set_params_from_reference (psi.py) : bins=4, edges + ref_probs. */
static const float k_edges[] = {0.05f, 0.275f, 0.5f, 0.725f, 0.95f};
static const float k_ref[]   = {0.25000001f, 0.20000001f, 0.35000001f, 0.20000001f};

void test_psi_drift_on_collapsed_block(void)
{
    /* block_size=5, threshold=0.2 ; bloc 1 étalé (NORMAL) puis bloc 2 concentré → DRIFT.
     * Verdicts identiques à PSI.update (psi.py, metric="psi"). */
    PSI d; psi_init(&d, 4, 5, 0.2f, k_edges, k_ref);
    const float seq[] = {0.1f,0.4f,0.6f,0.9f,0.5f, 0.95f,0.95f,0.95f,0.95f,0.95f};
    const DriftMethodVerdict exp[] = {
        DM_NORMAL, DM_NORMAL, DM_NORMAL, DM_NORMAL, DM_NORMAL,
        DM_NORMAL, DM_NORMAL, DM_NORMAL, DM_NORMAL, DM_DRIFT };
    int n = (int)(sizeof(seq)/sizeof(seq[0]));
    for (int i = 0; i < n; i++)
        TEST_ASSERT_EQUAL_INT(exp[i], psi_update(&d, seq[i]));
    /* Statistique du bloc déclenchant proche de la valeur Python (14.99 ± tol FP32). */
    TEST_ASSERT_FLOAT_WITHIN(0.05f, 14.9882f, d.last_stat);
}

void test_psi_reset_clears_block(void)
{
    PSI d; psi_init(&d, 4, 5, 0.2f, k_edges, k_ref);
    psi_update(&d, 0.1f); psi_update(&d, 0.4f);
    psi_reset(&d);
    TEST_ASSERT_EQUAL_INT(0, d.block_seen);
    for (int i = 0; i < d.bins; i++) TEST_ASSERT_EQUAL_UINT16(0, d.cur_counts[i]);
}
