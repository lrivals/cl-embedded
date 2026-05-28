/**
 * test_pipeline.c — Tests Unity pour pipeline.c (mock UART, host x86)
 *
 * Compilé avec -DTEST_MODE=1 -DDEBUG_PRINTF=1.
 * uart_send_byte / uart_getbyte sont des mocks capturant le trafic UART.
 */

#include "unity.h"
#include "pipeline.h"
#include "profiling.h"
#include "ewc_head.h"
#include <string.h>
#include <stdint.h>

/* ── Mock uart_send_byte ─────────────────────────────────────────────────── */

uint8_t uart_tx_buf[256];
int     uart_tx_count = 0;

void uart_send_byte(uint8_t b)
{
    if (uart_tx_count < (int)sizeof(uart_tx_buf))
        uart_tx_buf[uart_tx_count++] = b;
}

void uart_tx_reset(void) { uart_tx_count = 0; }
int  uart_tx_get(void)   { return uart_tx_count; }

/* ── Mock uart_getbyte ───────────────────────────────────────────────────── */

uint8_t uart_rx_buf[64];
int     uart_rx_pos  = 0;
int     uart_rx_len  = 0;

uint8_t uart_getbyte(void)
{
    if (uart_rx_pos < uart_rx_len)
        return uart_rx_buf[uart_rx_pos++];
    return 0x00U;
}

/* ── Helper : CRC8 (poly 0x07, identique à pipeline.c) ─────────────────── */

static uint8_t mock_crc8(const uint8_t *d, int len)
{
    uint8_t crc = 0;
    for (int i = 0; i < len; i++) {
        crc ^= d[i];
        for (int j = 0; j < 8; j++)
            crc = (crc & 0x80U) ? (uint8_t)((crc << 1) ^ 0x07U)
                                : (uint8_t)(crc << 1);
    }
    return crc;
}

/* ── Helper : construit une trame v2 valide dans uart_rx_buf ────────────── */
/*  Format : MAGIC(2) VERSION(1) TASK_ID(1) TS_MS(4) N(1) feats(N*4)       */
/*           label(1) flags(1) CRC8(1) — N=MAHA_DIM, feats=0.0f            */

static void build_valid_frame(void)
{
    uint8_t pay[64];
    int     pi = 0;

    pay[pi++] = 0xCDU;  /* MAGIC0 */
    pay[pi++] = 0xABU;  /* MAGIC1 */
    pay[pi++] = 0x02U;  /* VERSION v2 */
    pay[pi++] = 0x00U;  /* TASK_ID */

    /* TIMESTAMP_MS little-endian */
    pay[pi++] = 0x00U; pay[pi++] = 0x00U;
    pay[pi++] = 0x00U; pay[pi++] = 0x00U;

    pay[pi++] = (uint8_t)MAHA_DIM;  /* N features */

    /* features : MAHA_DIM × float 0.0f (little-endian) */
    for (int i = 0; i < MAHA_DIM; i++) {
        pay[pi++] = 0x00U; pay[pi++] = 0x00U;
        pay[pi++] = 0x00U; pay[pi++] = 0x00U;
    }

    pay[pi++] = 0x00U;  /* label */
    pay[pi++] = 0x00U;  /* flags (pas d'update) */

    uint8_t crc = mock_crc8(pay, pi);

    memcpy(uart_rx_buf, pay, (size_t)pi);
    uart_rx_buf[pi] = crc;
    uart_rx_len = pi + 1;
    uart_rx_pos = 0;
}

/* ── Déclarations des fonctions exposées sous TEST_MODE ─────────────────── */
void test_pipeline_send_response_v2(uint8_t pred, float conf,
                                     uint32_t lat_us, uint8_t status);
void test_pipeline_send_response_v3(uint8_t pred, float conf,
                                     uint32_t lat_us, uint8_t status,
                                     const MetricsSnapshot *snap);

/* ── Test 1 : réponse binaire v3 fait exactement 21 octets ─────────────── */

void test_pipeline_response_v3_21bytes(void)
{
    memset(&g_profiling, 0, sizeof(g_profiling));

    MetricsSnapshot snap = {.accuracy = 0.9f, .auroc = 0.75f, .forgetting = 0.01f};
    uart_tx_reset();
    test_pipeline_send_response_v3(0U, 0.9f, 0U, 0x00U, &snap);

    TEST_ASSERT_EQUAL_INT_MESSAGE(21, uart_tx_count,
        "uart_send_response_v3 doit emettre exactement 21 octets");
}

/* ── Test 2 : champs acc/auroc/forgetting décodés correctement (±1e-6) ───── */

void test_protocol_v3_fields(void)
{
    memset(&g_profiling, 0, sizeof(g_profiling));

    MetricsSnapshot snap = {.accuracy = 0.75f, .auroc = 0.88f, .forgetting = 0.05f};
    uart_tx_reset();
    test_pipeline_send_response_v3(1U, 0.6f, 0U, 0x00U, &snap);

    TEST_ASSERT_EQUAL_INT(21, uart_tx_count);

    union { float f; uint8_t b[4]; } uf;

    /* pred = byte 0 */
    TEST_ASSERT_EQUAL_UINT8(1U, uart_tx_buf[0]);

    /* conf = bytes 1..4 */
    memcpy(uf.b, &uart_tx_buf[1], 4);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.6f, uf.f);

    /* acc = bytes 9..12 (après pred+conf+lat_us) */
    memcpy(uf.b, &uart_tx_buf[9], 4);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.75f, uf.f);

    /* auroc = bytes 13..16 */
    memcpy(uf.b, &uart_tx_buf[13], 4);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.88f, uf.f);

    /* forgetting = bytes 17..20 */
    memcpy(uf.b, &uart_tx_buf[17], 4);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.05f, uf.f);
}

/* ── Test 3 : buffer UART contient "score=" quand DEBUG_PRINTF=1 ─────────── */

void test_pipeline_debug_printf_contains_score(void)
{
    memset(&g_profiling, 0, sizeof(g_profiling));
    pipeline_init();

    build_valid_frame();
    uart_tx_reset();

    pipeline_run();

    /* Chercher la chaîne "score=" dans les octets transmis */
    const char *needle = "score=";
    int needle_len     = (int)strlen(needle);
    int found          = 0;

    for (int i = 0; i <= uart_tx_count - needle_len; i++) {
        if (memcmp(&uart_tx_buf[i], needle, (size_t)needle_len) == 0) {
            found = 1;
            break;
        }
    }

    TEST_ASSERT_TRUE_MESSAGE(found, "\"score=\" absent du buffer UART TX");
}

/* ── Test 3 : réponse binaire v2 fait exactement 14 octets (non-régression) */

void test_pipeline_response_v2_14bytes(void)
{
    memset(&g_profiling, 0, sizeof(g_profiling));

    uart_tx_reset();
    test_pipeline_send_response_v2(0U, 0.9f, 0U, 0x00U);

    TEST_ASSERT_EQUAL_INT_MESSAGE(14, uart_tx_count,
        "uart_send_response_v2 doit emettre exactement 14 octets");
}

/* ── Test 5 : FLAGS=0x04 déclenche ewc_consolidate() — star_w copié ────────
 * On fixe g_ewc_head.w1[0][0] = 1.0f avant le run.
 * Après pipeline_run() avec PROTO_FLAG_CONSOLIDATE, ewc_consolidate()
 * doit avoir copié w1 dans star_w1, donc star_w1[0][0] == 1.0f.          */

static void build_frame_with_flags(uint8_t task_id, uint8_t flags)
{
    uint8_t pay[64];
    int     pi = 0;

    pay[pi++] = 0xCDU;
    pay[pi++] = 0xABU;
    pay[pi++] = 0x02U;            /* VERSION v2 */
    pay[pi++] = task_id;          /* TASK_ID */
    pay[pi++] = 0x00U; pay[pi++] = 0x00U;
    pay[pi++] = 0x00U; pay[pi++] = 0x00U;   /* TIMESTAMP_MS */
    pay[pi++] = (uint8_t)MAHA_DIM;

    for (int i = 0; i < MAHA_DIM; i++) {
        pay[pi++] = 0x00U; pay[pi++] = 0x00U;
        pay[pi++] = 0x00U; pay[pi++] = 0x00U;   /* features = 0.0f */
    }

    pay[pi++] = 0x00U;   /* label */
    pay[pi++] = flags;   /* FLAGS */

    uint8_t crc = mock_crc8(pay, pi);
    memcpy(uart_rx_buf, pay, (size_t)pi);
    uart_rx_buf[pi] = crc;
    uart_rx_len = pi + 1;
    uart_rx_pos = 0;
}

void test_pipeline_consolidate_flag(void)
{
    memset(&g_profiling, 0, sizeof(g_profiling));
    pipeline_init();

    /* Valeur sentinelle dans les poids courants */
    g_ewc_head.w1[0][0] = 1.0f;
    /* star_w doit être différent avant la consolidation */
    g_ewc_head.star_w1[0][0] = 0.0f;

    build_frame_with_flags(1U, PROTO_FLAG_CONSOLIDATE);
    uart_tx_reset();
    pipeline_run();

    /* ewc_consolidate() copie w1 → star_w1 */
    TEST_ASSERT_FLOAT_WITHIN_MESSAGE(1e-6f, 1.0f, g_ewc_head.star_w1[0][0],
        "PROTO_FLAG_CONSOLIDATE doit declencher ewc_consolidate() : star_w1[0][0] != w1[0][0]");
}
