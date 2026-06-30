/**
 * test_drift_detector.c — Tests unitaires Unity pour drift_detector.c (Sprint 38 S3803)
 *
 * Vérifie la parité bit-à-bit avec src/evaluation/drift_detector.py sur une séquence
 * de scores connue, plus les invariants du gate :
 *   - test_drift_normal_below_thresholds  : scores bas → NORMAL
 *   - test_drift_fault_instantaneous      : dépassement instantané → FAULT
 *   - test_drift_fault_priority_over_drift : FAULT > DRIFT (testé en premier)
 *   - test_drift_triggers_on_ratio        : DRIFT au franchissement de drift_ratio
 *   - test_drift_reset_clears_window      : drift_reset vide la fenêtre
 *   - test_drift_sequence_parity_python   : séquence complète == SlidingWindowDriftDetector
 */

#include "unity.h"
#include "drift_detector.h"

/* Seuils de test : fault=10, drift=5, fenêtre=4, ratio=0.5 (strict >). */
#define FAULT_THR 10.0f
#define DRIFT_THR 5.0f
#define WIN       4
#define RATIO     0.5f

/* NB : DriftDetector n'est PAS copiable (window.storage pointe vers d.storage[] interne).
 * On l'initialise toujours EN PLACE via un pointeur — comme le global statique g_drift
 * du firmware. Ne jamais retourner/copier une DriftDetector par valeur. */
static void make_det(DriftDetector *d)
{
    drift_init(d, WIN, FAULT_THR, DRIFT_THR, RATIO);
}

void test_drift_normal_below_thresholds(void)
{
    DriftDetector d; make_det(&d);
    TEST_ASSERT_EQUAL_INT(DRIFT_NORMAL, drift_update(&d, 1.0f));
    TEST_ASSERT_EQUAL_INT(DRIFT_NORMAL, drift_update(&d, 2.0f));
    TEST_ASSERT_EQUAL_INT(DRIFT_NORMAL, drift_update(&d, 0.5f));
}

void test_drift_fault_instantaneous(void)
{
    DriftDetector d; make_det(&d);
    /* Premier score > fault_threshold → FAULT immédiat (avant tout DRIFT). */
    TEST_ASSERT_EQUAL_INT(DRIFT_FAULT, drift_update(&d, 11.0f));
}

void test_drift_fault_priority_over_drift(void)
{
    DriftDetector d; make_det(&d);
    /* Remplir la fenêtre de scores en zone de dérive (≥ drift, < fault). */
    drift_update(&d, 6.0f);
    drift_update(&d, 6.0f);
    drift_update(&d, 6.0f);   /* la fenêtre est en DRIFT */
    /* Un score > fault doit renvoyer FAULT même si la fenêtre serait DRIFT. */
    TEST_ASSERT_EQUAL_INT(DRIFT_FAULT, drift_update(&d, 12.0f));
}

void test_drift_triggers_on_ratio(void)
{
    DriftDetector d; make_det(&d);
    /* fenêtre maxlen=4, ratio strict 0.5 */
    TEST_ASSERT_EQUAL_INT(DRIFT_NORMAL, drift_update(&d, 1.0f));   /* [1] : 0/1 */
    TEST_ASSERT_EQUAL_INT(DRIFT_NORMAL, drift_update(&d, 6.0f));   /* [1,6] : 1/2=0.5 ! >0.5 */
    TEST_ASSERT_EQUAL_INT(DRIFT_DRIFT,  drift_update(&d, 6.0f));   /* [1,6,6] : 2/3>0.5 */
}

void test_drift_reset_clears_window(void)
{
    DriftDetector d; make_det(&d);
    drift_update(&d, 6.0f);
    drift_update(&d, 6.0f);
    drift_update(&d, 6.0f);   /* DRIFT actif */
    drift_reset(&d);
    TEST_ASSERT_EQUAL_INT(0, d.window.count);
    /* Après reset, un score de dérive isolé ne déclenche pas DRIFT (1/1 > 0.5 → si !). */
    /* 6 > drift mais 1/1 = 1.0 > 0.5 → DRIFT dès le premier ; on teste plutôt un score bas. */
    TEST_ASSERT_EQUAL_INT(DRIFT_NORMAL, drift_update(&d, 1.0f));
}

void test_drift_sequence_parity_python(void)
{
    /* Séquence identique à la référence Python (SlidingWindowDriftDetector.update_batch).
     * Verdicts attendus calculés à la main + vérifiés contre le Python (test S3809). */
    DriftDetector d; make_det(&d);
    const float seq[]      = {1.0f, 6.0f, 6.0f, 11.0f, 6.0f, 2.0f};
    const DriftVerdict exp[] = {DRIFT_NORMAL, DRIFT_NORMAL, DRIFT_DRIFT,
                                DRIFT_FAULT, DRIFT_DRIFT, DRIFT_DRIFT};
    int n = (int)(sizeof(seq) / sizeof(seq[0]));
    for (int i = 0; i < n; i++) {
        TEST_ASSERT_EQUAL_INT(exp[i], drift_update(&d, seq[i]));
    }
}
