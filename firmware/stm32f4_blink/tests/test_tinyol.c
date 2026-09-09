/**
 * test_tinyol.c — Tests unitaires Unity pour tinyol.c
 *
 * Architecture : Input(5) → ReLU(32) → ReLU(16) → Output(5) [autoencoder]
 * Tests :
 *   - poids zéro → embedding nul
 *   - poids zéro → reconstruction nulle
 *   - MSE reconstruction sur MOCK_NORMAL_T0[0] avec poids zéro ≈ 0.007
 *   - prédiction anomalie (MSE >> seuil)
 *   - déterminisme forward
 *   - tinyol_init() charge bien les constantes Flash
 *   - delta C vs Python (seed=42) ≤ 1e-5
 */

#include "unity.h"
#include "tinyol.h"
#include "model_weights.h"
#include "mock_data.h"
#include <math.h>
#include <string.h>

#ifndef TINYOL_NATIVE_DIM
#define TINYOL_NATIVE_DIM WEIGHTS_NATIVE_DIM
#endif

/* Référence Python du forward, GÉNÉRÉE avec les poids de model_weights.h par
 * scripts/export_weights_tinyol.py --emit-test-reference (S5201). Auparavant le
 * golden était recopié à la main et se désynchronisait du header au premier
 * ré-export — d'où un échec permanent de la parité C↔Python. */
#include "tinyol_reference.h"

/* ── Helpers ────────────────────────────────────────────────────────────── */

static TinyOLEncoder make_zero_enc(void)
{
    TinyOLEncoder enc;
    memset(&enc, 0, sizeof(enc));
    return enc;
}

static TinyOLDecoder make_zero_dec(void)
{
    TinyOLDecoder dec;
    memset(&dec, 0, sizeof(dec));
    return dec;
}

/* ── Tests encoder poids zéro ───────────────────────────────────────────── */

void test_tinyol_encode_zero_weights(void)
{
    /* Poids et biais nuls : ReLU(0) = 0 partout → embedding nul */
    TinyOLEncoder enc = make_zero_enc();
    float emb[TINYOL_EMB];
    tinyol_encode(&enc, MOCK_NORMAL_T0[0], emb);
    for (int i = 0; i < (int)TINYOL_EMB; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, emb[i]);
    }
}

void test_tinyol_decode_zero_emb(void)
{
    /* Embedding nul + poids nuls → reconstruction nulle */
    TinyOLDecoder dec = make_zero_dec();
    float emb[TINYOL_EMB] = {0};
    float recon[TINYOL_OUT];
    tinyol_decode(&dec, emb, recon);
    for (int i = 0; i < (int)TINYOL_OUT; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, recon[i]);
    }
}

void test_tinyol_recon_error_zero_weights(void)
{
    /* Poids nuls → recon=[0,...,0] → MSE = ||x||²/N ≈ 0.007
     * x = MOCK_NORMAL_T0[0] = {0.10, 0.05, 0.08, -0.03, 0.12} */
    TinyOLEncoder enc = make_zero_enc();
    TinyOLDecoder dec = make_zero_dec();
    float emb[TINYOL_EMB];
    float recon[TINYOL_OUT];
    tinyol_encode(&enc, MOCK_NORMAL_T0[0], emb);
    tinyol_decode(&dec, emb, recon);
    float err = tinyol_reconstruction_error(MOCK_NORMAL_T0[0], recon, (int)TINYOL_OUT);
    TEST_ASSERT_FLOAT_WITHIN(MOCK_TINYOL_RECON_TOLERANCE, MOCK_TINYOL_RECON_ERR_ZERO_WEIGHTS, err);
}

/* ── Tests prédiction ───────────────────────────────────────────────────── */

/* Seuil de test explicite. TINYOL_THRESHOLD est CALIBRÉ (P95×1.5 des MSE
 * d'entraînement) et régénéré à chaque export : sur des poids entraînés il vaut
 * ~1e-5, valeur devant laquelle même un échantillon normal reconstruit par des
 * poids NULS (MSE ≈ 0.0088) est une anomalie. L'utiliser ici comme oracle testait
 * donc la campagne d'export, pas la logique de tinyol_predict. */
#define TEST_TINYOL_THRESHOLD 0.05f

void test_tinyol_predict_anomaly_zero_weights(void)
{
    /* Poids nuls + entrée très hors distribution → MSE >> seuil → predict=1 */
    TinyOLEncoder enc = make_zero_enc();
    TinyOLDecoder dec = make_zero_dec();
    /* MOCK_ANOMALY_T0[0] = {3.20, 3.10, 3.05, 3.15, 3.08}
     * MSE ≈ (3.2²+3.1²+3.05²+3.15²+3.08²)/5 ≈ 9.71 >> 0.05 */
    int pred = tinyol_predict(&enc, &dec, MOCK_ANOMALY_T0[0], TEST_TINYOL_THRESHOLD);
    TEST_ASSERT_EQUAL_INT(1, pred);
}

void test_tinyol_predict_normal_zero_weights(void)
{
    /* Poids nuls + entrée normale → MSE ≈ 0.0088 < 0.05 → predict=0 */
    TinyOLEncoder enc = make_zero_enc();
    TinyOLDecoder dec = make_zero_dec();
    int pred = tinyol_predict(&enc, &dec, MOCK_NORMAL_T0[0], TEST_TINYOL_THRESHOLD);
    TEST_ASSERT_EQUAL_INT(0, pred);
}

/* ── Tests déterminisme ─────────────────────────────────────────────────── */

void test_tinyol_forward_deterministic(void)
{
    /* Même entrée × 2 → embedding identique */
    TinyOLEncoder enc = make_zero_enc();
    float emb1[TINYOL_EMB], emb2[TINYOL_EMB];
    tinyol_encode(&enc, MOCK_NORMAL_T0[0], emb1);
    tinyol_encode(&enc, MOCK_NORMAL_T0[0], emb2);
    for (int i = 0; i < (int)TINYOL_EMB; i++) {
        TEST_ASSERT_EQUAL_FLOAT(emb1[i], emb2[i]);
    }
}

/* ── Test tinyol_init ───────────────────────────────────────────────────── */

void test_tinyol_init_loads_constants(void)
{
    /* Vérifie que memcpy copie bien TINYOL_W_ENC1 → enc.w_enc1 */
    TinyOLEncoder enc;
    TinyOLDecoder dec;
    tinyol_init(&enc, &dec);
    TEST_ASSERT_FLOAT_WITHIN(1e-7f, TINYOL_W_ENC1[0][0], enc.w_enc1[0][0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-7f, TINYOL_W_ENC2[0][0], enc.w_enc2[0][0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-7f, TINYOL_W_DEC1[0][0], dec.w_dec1[0][0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-7f, TINYOL_W_DEC2[0][0], dec.w_dec2[0][0]);
}

/* ── Test delta C vs Python ─────────────────────────────────────────────── */

void test_tinyol_forward_delta(void)
{
    /* Parité numérique : forward C = forward Python à 1e-5 près, SUR LES POIDS
     * RÉELLEMENT EMBARQUÉS (référence régénérée avec eux). Tolérance 1e-5 :
     * FP32 sans quantification (conforme S2003 §3). */
#if (TINYOL_IN == TINYOL_REF_DIM) && (TINYOL_IN == TINYOL_NATIVE_DIM)
    TinyOLEncoder enc;
    TinyOLDecoder dec;
    tinyol_init(&enc, &dec);

    float emb[TINYOL_EMB];
    float recon[TINYOL_OUT];
    tinyol_encode(&enc, TINYOL_REF_INPUT, emb);
    tinyol_decode(&dec, emb, recon);

    /* Embedding */
    for (int i = 0; i < (int)TINYOL_EMB; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, TINYOL_REF_EMB[i], emb[i]);
    }

    /* Reconstruction */
    for (int i = 0; i < (int)TINYOL_OUT; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, TINYOL_REF_RECON[i], recon[i]);
    }

    /* MSE reconstruction */
    float err = tinyol_reconstruction_error(TINYOL_REF_INPUT, recon, (int)TINYOL_OUT);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, TINYOL_REF_MSE, err);
#else
    /* Build à une dim ≠ celle des poids/référence exportés : parité non
     * évaluable, et tinyol_init laisse volontairement l'auto-encodeur à zéro. */
    TEST_IGNORE_MESSAGE("TINYOL_IN != TINYOL_NATIVE_DIM/TINYOL_REF_DIM — parité N/A");
#endif
}
