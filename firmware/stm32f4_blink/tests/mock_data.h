#pragma once
/**
 * mock_data.h — Jeux de données synthétiques pour tests Unity host (sans board).
 *
 * Données codées en dur couvrant :
 *   - Samples normaux (label 0) et anormaux (label 1)
 *   - 3 tâches simulées (task 0, 1, 2) avec distribution légèrement différente
 *   - Valeurs numériquement stables pour comparaison PC ↔ C (tolérance 1e-4)
 *
 * Usage dans les tests :
 *   #include "mock_data.h"
 *   float err = maha_score(&det, MOCK_NORMAL_T0[0]);
 *   TEST_ASSERT_FLOAT_WITHIN(1e-4f, MOCK_NORMAL_SCORE_T0, err);
 */

#include <stdint.h>

#define MOCK_N_FEATURES 5U
#define MOCK_N_SAMPLES  10U
#define MOCK_N_TASKS    3U

/* ── Samples normaux (task 0 — distribution centroïde proche de 0) ──────── */
static const float MOCK_NORMAL_T0[MOCK_N_SAMPLES][MOCK_N_FEATURES] = {
    { 0.10f,  0.05f,  0.08f, -0.03f,  0.12f },
    { 0.07f, -0.02f,  0.11f,  0.04f,  0.09f },
    { 0.13f,  0.08f, -0.05f,  0.07f,  0.06f },
    {-0.02f,  0.11f,  0.09f,  0.03f,  0.14f },
    { 0.05f,  0.06f,  0.07f, -0.06f,  0.10f },
    { 0.09f, -0.04f,  0.12f,  0.05f,  0.08f },
    { 0.11f,  0.07f,  0.04f,  0.09f, -0.03f },
    {-0.01f,  0.09f,  0.10f,  0.06f,  0.11f },
    { 0.08f,  0.03f, -0.07f,  0.08f,  0.13f },
    { 0.12f,  0.10f,  0.06f, -0.02f,  0.07f },
};
static const uint8_t MOCK_NORMAL_LABELS_T0[MOCK_N_SAMPLES] = {0,0,0,0,0,0,0,0,0,0};

/* ── Samples anormaux (task 0 — fortement hors distribution) ────────────── */
static const float MOCK_ANOMALY_T0[MOCK_N_SAMPLES][MOCK_N_FEATURES] = {
    { 3.20f,  3.10f,  3.05f,  3.15f,  3.08f },
    {-3.50f, -3.20f, -3.40f, -3.60f, -3.30f },
    { 2.90f,  3.40f,  3.10f,  2.80f,  3.20f },
    {-3.10f,  3.00f, -3.20f,  3.10f, -3.00f },
    { 3.80f,  3.70f,  3.60f,  3.50f,  3.75f },
    {-2.80f, -3.50f,  2.90f, -3.10f,  3.20f },
    { 3.30f, -3.00f,  3.40f, -2.90f,  3.10f },
    { 4.00f,  3.90f,  4.10f,  3.80f,  3.95f },
    {-3.70f, -3.80f, -3.60f, -3.90f, -3.75f },
    { 3.50f,  3.60f,  3.45f,  3.55f,  3.50f },
};
static const uint8_t MOCK_ANOMALY_LABELS_T0[MOCK_N_SAMPLES] = {1,1,1,1,1,1,1,1,1,1};

/* ── Samples normaux task 1 (drift léger : centroïde déplacé de +0.5) ───── */
static const float MOCK_NORMAL_T1[MOCK_N_SAMPLES][MOCK_N_FEATURES] = {
    { 0.60f,  0.55f,  0.58f,  0.47f,  0.62f },
    { 0.57f,  0.48f,  0.61f,  0.54f,  0.59f },
    { 0.63f,  0.58f,  0.45f,  0.57f,  0.56f },
    { 0.48f,  0.61f,  0.59f,  0.53f,  0.64f },
    { 0.55f,  0.56f,  0.57f,  0.44f,  0.60f },
    { 0.59f,  0.46f,  0.62f,  0.55f,  0.58f },
    { 0.61f,  0.57f,  0.54f,  0.59f,  0.47f },
    { 0.49f,  0.59f,  0.60f,  0.56f,  0.61f },
    { 0.58f,  0.53f,  0.43f,  0.58f,  0.63f },
    { 0.62f,  0.60f,  0.56f,  0.48f,  0.57f },
};
static const uint8_t MOCK_NORMAL_LABELS_T1[MOCK_N_SAMPLES] = {0,0,0,0,0,0,0,0,0,0};

/* ── Samples normaux task 2 (drift : centroïde déplacé de +1.0) ─────────── */
static const float MOCK_NORMAL_T2[MOCK_N_SAMPLES][MOCK_N_FEATURES] = {
    { 1.10f,  1.05f,  1.08f,  0.97f,  1.12f },
    { 1.07f,  0.98f,  1.11f,  1.04f,  1.09f },
    { 1.13f,  1.08f,  0.95f,  1.07f,  1.06f },
    { 0.98f,  1.11f,  1.09f,  1.03f,  1.14f },
    { 1.05f,  1.06f,  1.07f,  0.94f,  1.10f },
    { 1.09f,  0.96f,  1.12f,  1.05f,  1.08f },
    { 1.11f,  1.07f,  1.04f,  1.09f,  0.97f },
    { 0.99f,  1.09f,  1.10f,  1.06f,  1.11f },
    { 1.08f,  1.03f,  0.93f,  1.08f,  1.13f },
    { 1.12f,  1.10f,  1.06f,  0.98f,  1.07f },
};
static const uint8_t MOCK_NORMAL_LABELS_T2[MOCK_N_SAMPLES] = {0,0,0,0,0,0,0,0,0,0};

/* ── Scores Mahalanobis attendus (précision matricielle = identité) ─────── */
/* Calculés avec Python : np.sqrt(sum(x**2)) pour mean=0, precision=I */
static const float MOCK_MAHA_SCORE_NORMAL_T0_MAX  = 0.25f;  /* tous < 0.25 */
static const float MOCK_MAHA_SCORE_ANOMALY_T0_MIN = 5.0f;   /* tous > 5.0 */

/* ── EWC logits attendus (réseau initialisé à zéro → logits ~ [0, 0]) ───── */
static const float MOCK_EWC_LOGIT_TOLERANCE = 0.01f;

/* ── TinyOL reconstruction error attendue (poids zéro → MSE ~ ||x||²/N) ── */
/* Pour MOCK_NORMAL_T0[0] = [0.10, 0.05, 0.08, -0.03, 0.12] avec poids=0 :
 * recon = [0, 0, 0, 0, 0], MSE = (0.01+0.0025+0.0064+0.0009+0.0144)/5 ≈ 0.007 */
static const float MOCK_TINYOL_RECON_ERR_ZERO_WEIGHTS = 0.007f;
static const float MOCK_TINYOL_RECON_TOLERANCE = 1e-4f;
