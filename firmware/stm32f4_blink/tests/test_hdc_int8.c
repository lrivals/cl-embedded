/**
 * test_hdc_int8.c — Tests Unity pour hdc_int8.c (Sprint 29, S2906)
 *
 * Exécution sur x86 (pas de board requise). Teste les invariants arithmétiques
 * entiers de la variante INT8 : base vectors ±1, AM int16 saturée, predict int32.
 *
 * Référence : test_hdc.c (style FP32), inc/hdc_int8.h (interface).
 */

#include "unity.h"
#include "hdc_int8.h"
#include <string.h>

/* ── Test 1 — init met l'AM à zéro ──────────────────────────────────────── */
void test_hdc_int8_init_zeros_am(void)
{
    HDCInt8 h;
    memset(&h, 0xAB, sizeof(h));   /* pollution initiale */
    hdc_int8_init(&h);

    for (int c = 0; c < HDC_I_C; c++)
        for (int i = 0; i < HDC_I_D; i++)
            TEST_ASSERT_EQUAL_INT16(0, h.am[c][i]);
}

/* ── Test 2 — encode produit un hypervecteur strictement ±1 ─────────────── */
void test_hdc_int8_encode_bipolar(void)
{
    HDCInt8 h;
    hdc_int8_init(&h);
    float x[HDC_I_N];
    for (int i = 0; i < HDC_I_N; i++) x[i] = (float)(i + 1);

    int8_t hv[HDC_I_D];
    hdc_int8_encode(&h, x, hv);

    for (int i = 0; i < HDC_I_D; i++) {
        TEST_ASSERT_TRUE(hv[i] == 1 || hv[i] == -1);
    }
}

/* ── Test 3 — predict sépare deux classes après updates ─────────────────── */
void test_hdc_int8_predict_after_updates(void)
{
    HDCInt8 h;
    hdc_int8_init(&h);

    /* Deux inputs distincts → hypervecteurs distincts */
    float x0[HDC_I_N], x1[HDC_I_N];
    for (int i = 0; i < HDC_I_N; i++) { x0[i] =  1.0f; x1[i] = -1.0f; }

    int8_t hv0[HDC_I_D], hv1[HDC_I_D];
    hdc_int8_encode(&h, x0, hv0);
    hdc_int8_encode(&h, x1, hv1);

    for (int k = 0; k < 10; k++) {
        hdc_int8_update(&h, hv0, 0);
        hdc_int8_update(&h, hv1, 1);
    }

    TEST_ASSERT_EQUAL_INT(0, hdc_int8_predict(&h, hv0));
    TEST_ASSERT_EQUAL_INT(1, hdc_int8_predict(&h, hv1));
}

/* ── Test 4 — update accumule am[c][i] == N * hv[i] (sans saturation) ────── */
void test_hdc_int8_update_accumulates(void)
{
    HDCInt8 h;
    hdc_int8_init(&h);

    float x[HDC_I_N];
    for (int i = 0; i < HDC_I_N; i++) x[i] = 1.0f;
    int8_t hv[HDC_I_D];
    hdc_int8_encode(&h, x, hv);

    hdc_int8_update(&h, hv, 0);
    hdc_int8_update(&h, hv, 0);
    hdc_int8_update(&h, hv, 0);

    /* am[0][0] == 3 * hv[0] == ±3 (pas de saturation pour N=3) */
    int16_t expected = (int16_t)(3 * (int)hv[0]);
    TEST_ASSERT_EQUAL_INT16(expected, h.am[0][0]);
}

/* ── Test 5 — taille statique du struct conforme aux #defines ───────────── */
void test_hdc_int8_sizeof(void)
{
    /* bv : HDC_I_N × HDC_I_D × sizeof(int8_t)  = 9 × 2048 × 1 = 18 432 B
     * am : HDC_I_C × HDC_I_D × sizeof(int16_t) = 4 × 2048 × 2 = 16 384 B
     * Total minimum attendu : 34 816 B (padding possible mais borné) */
    TEST_ASSERT_TRUE(sizeof(HDCInt8) >= (size_t)(HDC_I_N * HDC_I_D + HDC_I_C * HDC_I_D * 2));
    /* Vérif que le struct n'excède pas 36 Ko (10% de marge de padding max) */
    TEST_ASSERT_TRUE(sizeof(HDCInt8) <= 36864u);
}
