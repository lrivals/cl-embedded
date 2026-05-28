/**
 * test_profiling.c — Tests unitaires Unity pour profiling.c
 *
 * Tests exécutés sur host (pas de DWT réel) : DWT_CYCCNT est mocké via
 * g_mock_cyccnt défini dans profiling.c (#ifdef TEST_HOST).
 */

#include "unity.h"
#include "profiling.h"
#include <string.h>
#include <stdint.h>

/* Déclaré dans profiling.c avec -DTEST_HOST */
extern volatile uint32_t g_mock_cyccnt;

/* ── Test 1 : latence positive après start/stop ─────────────────────────── */

void test_profiling_latency_positive(void)
{
    memset(&g_profiling, 0, sizeof(g_profiling));
    g_mock_cyccnt = 0U;

    /* Simule 180 cycles → 1 µs à 180 MHz */
    profiling_start();
    g_mock_cyccnt = 180U;
    profiling_stop();

    TEST_ASSERT_EQUAL_UINT32(1U, profiling_get_latency_us());
}

/* ── Test 2 : latence nulle si start/stop sans cycles ───────────────────── */

void test_profiling_latency_zero_cycles(void)
{
    memset(&g_profiling, 0, sizeof(g_profiling));
    g_mock_cyccnt = 100U;

    profiling_start();
    /* Pas de progression du compteur */
    profiling_stop();

    TEST_ASSERT_EQUAL_UINT32(0U, profiling_get_latency_us());
}

/* ── Test 3 : throughput non nul après première inférence ───────────────── */

void test_profiling_throughput_nonzero(void)
{
    memset(&g_profiling, 0, sizeof(g_profiling));
    g_mock_cyccnt = 0U;

    profiling_start();
    g_mock_cyccnt = 180000U;   /* 1000 µs = 1 ms → 1000 ips */
    profiling_stop();

    /* SYSCLK_HZ / avg_cycles = 180 000 000 / 180 000 = 1000 ips */
    TEST_ASSERT_GREATER_THAN_UINT16(0U, profiling_get_throughput_ips());
}

/* ── Test 4 : encodage little-endian 8 B ───────────────────────────────── */

void test_profiling_encode_format(void)
{
    memset(&g_profiling, 0, sizeof(g_profiling));

    g_profiling.last_latency_us = 0x01020304UL;
    g_profiling.bss_bytes       = 0x0506U;
    g_profiling.throughput_ips  = 0x0708U;

    uint8_t buf[PROFILING_ENCODED_SIZE];
    profiling_encode(buf);

    /* latency_us little-endian */
    TEST_ASSERT_EQUAL_UINT8(0x04U, buf[0]);
    TEST_ASSERT_EQUAL_UINT8(0x03U, buf[1]);
    TEST_ASSERT_EQUAL_UINT8(0x02U, buf[2]);
    TEST_ASSERT_EQUAL_UINT8(0x01U, buf[3]);

    /* bss_bytes little-endian */
    TEST_ASSERT_EQUAL_UINT8(0x06U, buf[4]);
    TEST_ASSERT_EQUAL_UINT8(0x05U, buf[5]);

    /* throughput little-endian */
    TEST_ASSERT_EQUAL_UINT8(0x08U, buf[6]);
    TEST_ASSERT_EQUAL_UINT8(0x07U, buf[7]);
}

/* ── Test 5 : taille encodée correcte ──────────────────────────────────── */

void test_profiling_encode_size(void)
{
    TEST_ASSERT_EQUAL_UINT32(8U, PROFILING_ENCODED_SIZE);
}

/* ── Test 6 : bss_bytes dans la limite acceptable ──────────────────────── */

void test_bss_size_within_limit(void)
{
    memset(&g_profiling, 0, sizeof(g_profiling));

    /* profiling_init() écrit dans DWT (mocké en TEST_HOST → no-op sur les
     * macros DWT_CTRL/CoreDebug_DEMCR) et calcule bss via symboles linker.
     * En host, _sbss et _ebss peuvent être égaux → bss = 0. On vérifie
     * uniquement l'absence de dépassement u16. */
    profiling_init();
    TEST_ASSERT_LESS_THAN_UINT32(65535U, (uint32_t)profiling_get_bss_bytes());
}
