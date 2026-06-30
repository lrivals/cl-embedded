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

/* ── Test 1 : réponse binaire v3 fait exactement 23 octets ─────────────── */

void test_pipeline_response_v3_23bytes(void)
{
    memset(&g_profiling, 0, sizeof(g_profiling));

    MetricsSnapshot snap = {.accuracy = 0.9f, .auroc = 0.75f, .forgetting = 0.01f};
    uart_tx_reset();
    test_pipeline_send_response_v3(0U, 0.9f, 0U, 0x00U, &snap);

    TEST_ASSERT_EQUAL_INT_MESSAGE(23, uart_tx_count,
        "uart_send_response_v3 doit emettre exactement 23 octets");
}

/* ── Test 2 : champs ram_b/acc/auroc/forgetting décodés correctement (±1e-6) */

void test_protocol_v3_fields(void)
{
    memset(&g_profiling, 0, sizeof(g_profiling));
    /* bss_bytes connu pour le test */
    g_profiling.bss_bytes = 1234U;

    MetricsSnapshot snap = {.accuracy = 0.75f, .auroc = 0.88f, .forgetting = 0.05f};
    uart_tx_reset();
    test_pipeline_send_response_v3(1U, 0.6f, 0U, 0x00U, &snap);

    TEST_ASSERT_EQUAL_INT(23, uart_tx_count);

    union { float f; uint8_t b[4]; } uf;
    uint16_t ram_b;

    /* pred = byte 0 */
    TEST_ASSERT_EQUAL_UINT8(1U, uart_tx_buf[0]);

    /* conf = bytes 1..4 */
    memcpy(uf.b, &uart_tx_buf[1], 4);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.6f, uf.f);

    /* ram_b = bytes 9..10 (u16 little-endian) */
    memcpy(&ram_b, &uart_tx_buf[9], 2);
    TEST_ASSERT_EQUAL_UINT16(1234U, ram_b);

    /* acc = bytes 11..14 (après pred+conf+lat_us+ram_b) */
    memcpy(uf.b, &uart_tx_buf[11], 4);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.75f, uf.f);

    /* auroc = bytes 15..18 */
    memcpy(uf.b, &uart_tx_buf[15], 4);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.88f, uf.f);

    /* forgetting = bytes 19..22 */
    memcpy(uf.b, &uart_tx_buf[19], 4);
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

/* ── Sprint 27 — DUAL_MODE helpers + tests T76–T79 ──────────────────────── */

/* Expose test_pipeline_send_response_dual (défini dans pipeline.c TEST_MODE) */
void test_pipeline_send_response_dual(uint8_t pred_fault, float conf_fault,
                                       float rul_pred, float f1_macro,
                                       float rmse_rul, float forgetting);

/* Construit une trame DUAL_MODE (N=9, TASK_ID=fault_label, label=rul_u8) */
static void build_dual_frame(uint8_t fault_label, uint8_t rul_u8, uint8_t flags)
{
    uint8_t pay[64];
    int     pi = 0;

    pay[pi++] = 0xCDU;
    pay[pi++] = 0xABU;
    pay[pi++] = 0x03U;          /* VERSION v3 */
    pay[pi++] = fault_label;    /* TASK_ID réutilisé en DUAL_MODE : fault_label */

    pay[pi++] = 0x00U; pay[pi++] = 0x00U;
    pay[pi++] = 0x00U; pay[pi++] = 0x00U;   /* TIMESTAMP_MS = 0 */

    pay[pi++] = 9U;             /* N = 9 features */

    for (int i = 0; i < 9; i++) {           /* features = 0.0f */
        pay[pi++] = 0x00U; pay[pi++] = 0x00U;
        pay[pi++] = 0x00U; pay[pi++] = 0x00U;
    }

    pay[pi++] = rul_u8;         /* label = RUL encodé uint8 */
    pay[pi++] = flags;          /* FLAGS */

    uint8_t crc = mock_crc8(pay, pi);
    memcpy(uart_rx_buf, pay, (size_t)pi);
    uart_rx_buf[pi] = crc;
    uart_rx_len     = pi + 1;
    uart_rx_pos     = 0;
}

/* T76 — uart_send_response_dual() produit exactement 25 B */
void test_pipeline_response_dual_25bytes(void)
{
    uart_tx_reset();
    test_pipeline_send_response_dual(2, 0.75f, 0.60f, 0.65f, 0.08f, 0.01f);
    TEST_ASSERT_EQUAL_INT(RESPONSE_DUAL_SIZE, uart_tx_count);
}

/* T77 — les 7 champs sont encodés aux bons offsets */
void test_pipeline_dual_response_fields(void)
{
    uart_tx_reset();
    uint8_t expected_pred  = 3U;
    float   expected_conf  = 0.82f;
    float   expected_rul   = 0.45f;
    float   expected_f1    = 0.70f;
    float   expected_rmse  = 0.07f;
    float   expected_fgt   = 0.02f;

    test_pipeline_send_response_dual(expected_pred, expected_conf, expected_rul,
                                      expected_f1, expected_rmse, expected_fgt);

    TEST_ASSERT_EQUAL_INT(25, uart_tx_count);

    TEST_ASSERT_EQUAL_UINT8(expected_pred, uart_tx_buf[0]);

    float decoded_conf;
    memcpy(&decoded_conf, uart_tx_buf + 1, 4);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, expected_conf, decoded_conf);

    float decoded_rul;
    memcpy(&decoded_rul, uart_tx_buf + 5, 4);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, expected_rul, decoded_rul);

    float decoded_f1;
    memcpy(&decoded_f1, uart_tx_buf + 13, 4);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, expected_f1, decoded_f1);

    float decoded_rmse;
    memcpy(&decoded_rmse, uart_tx_buf + 17, 4);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, expected_rmse, decoded_rmse);

    float decoded_fgt;
    memcpy(&decoded_fgt, uart_tx_buf + 21, 4);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, expected_fgt, decoded_fgt);
}

/* T78 — FLAGS=0x70 → 25 B (DUAL_MODE) ; FLAGS=0x30 → pas 25 B (non-régression).
 * Note : pipeline_run() inclut DEBUG_PRINTF qui ajoute ~30 B — DUAL_MODE bypass
 * via early return, donc seul DUAL_MODE produit exactement 25 B. */
void test_pipeline_dual_mode_dispatch(void)
{
    pipeline_init();
    uart_tx_reset();
    build_dual_frame(1, 128, PROTO_FLAG_DUAL_MODE);
    pipeline_run();
    TEST_ASSERT_EQUAL_INT_MESSAGE(25, uart_tx_count,
        "DUAL_MODE (0x70) doit produire 25 B (early return, pas de DEBUG_PRINTF)");

    pipeline_init();
    uart_tx_reset();
    build_frame_with_flags(2U, PROTO_FLAG_MULTICLASS_MODE);
    pipeline_run();
    /* DEBUG_PRINTF ajoute ~30 B en mode test → total > 23 != 25 */
    TEST_ASSERT_NOT_EQUAL_MESSAGE(25, uart_tx_count,
        "MULTICLASS_MODE (0x30) ne doit pas etre intercepte par DUAL_MODE");
    TEST_ASSERT_TRUE_MESSAGE(uart_tx_count >= 23,
        "MULTICLASS_MODE doit avoir emis au moins 23 B (reponse v3)");
}

/* T79 — FLAGS=0x71 (DUAL+UPDATE) met à jour les deux modèles.
 * Avec features=0 et biais initiaux=0, les gradients de w1 sont nuls.
 * On vérifie b3[0] (bias output) qui change même avec input nul : dout≠0. */
void test_pipeline_dual_mode_update(void)
{
    pipeline_init();

    float b3_reg_before = g_ewc_reg.b3[0];
    float b3_mc_before  = g_ewc_mc.b3[0];

    build_dual_frame(0, 100, (uint8_t)(PROTO_FLAG_DUAL_MODE | PROTO_FLAG_UPDATE));
    uart_tx_reset();
    pipeline_run();

    TEST_ASSERT_NOT_EQUAL_MESSAGE(b3_reg_before, g_ewc_reg.b3[0],
        "ewc_reg.b3[0] doit changer apres UPDATE en DUAL_MODE (dout=out-y!=0)");
    TEST_ASSERT_NOT_EQUAL_MESSAGE(b3_mc_before, g_ewc_mc.b3[0],
        "ewc_mc.b3[0] doit changer apres UPDATE en DUAL_MODE");

    TEST_ASSERT_EQUAL_INT(25, uart_tx_count);
}

/* ── Sprint 30 — PAIR_MODE helpers + tests T80–T82 ──────────────────────── */

/* Expose test_pipeline_send_response_pair (défini dans pipeline.c TEST_MODE) */
void test_pipeline_send_response_pair(uint8_t pred_maha, float score_maha,
                                      uint8_t pred_sup, float conf_sup,
                                      float auroc_maha, float f1_sup);

/* T80 — uart_send_response_pair() produit exactement 22 B */
void test_pipeline_response_pair_22bytes(void)
{
    uart_tx_reset();
    test_pipeline_send_response_pair(1, 4.2f, 2, 0.83f, 0.71f, 0.65f);
    TEST_ASSERT_EQUAL_INT(RESPONSE_PAIR_SIZE, uart_tx_count);
}

/* T81 — les champs sont encodés aux bons offsets (layout 22 B) */
void test_pipeline_pair_response_fields(void)
{
    uart_tx_reset();
    uint8_t exp_pred_maha = 1U;
    float   exp_score     = 4.20f;
    uint8_t exp_pred_sup  = 3U;
    float   exp_conf      = 0.82f;
    float   exp_auroc     = 0.71f;
    float   exp_f1        = 0.66f;

    test_pipeline_send_response_pair(exp_pred_maha, exp_score, exp_pred_sup,
                                      exp_conf, exp_auroc, exp_f1);

    TEST_ASSERT_EQUAL_INT(22, uart_tx_count);
    TEST_ASSERT_EQUAL_UINT8(exp_pred_maha, uart_tx_buf[0]);

    float decoded_score;
    memcpy(&decoded_score, uart_tx_buf + 1, 4);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, exp_score, decoded_score);

    TEST_ASSERT_EQUAL_UINT8(exp_pred_sup, uart_tx_buf[5]);

    float decoded_conf;
    memcpy(&decoded_conf, uart_tx_buf + 6, 4);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, exp_conf, decoded_conf);

    float decoded_auroc;
    memcpy(&decoded_auroc, uart_tx_buf + 14, 4);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, exp_auroc, decoded_auroc);

    float decoded_f1;
    memcpy(&decoded_f1, uart_tx_buf + 18, 4);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, exp_f1, decoded_f1);
}

/* T82 — FLAGS=0x90 (PAIR_MAHA_EWC) → 22 B (early return) ; non intercepté par
 * un autre mode. EWC=0x10 (subset de 0x90) ne doit PAS court-circuiter le mode
 * paire car le dispatch PAIR passe en premier sur le nibble de mode. */
void test_pipeline_pair_mode_dispatch(void)
{
    pipeline_init();
    uart_tx_reset();
    build_frame_with_flags(0U, PROTO_FLAG_PAIR_MAHA_EWC);
    pipeline_run();
    TEST_ASSERT_EQUAL_INT_MESSAGE(22, uart_tx_count,
        "PAIR_MAHA_EWC (0x90) doit produire 22 B (early return, pas de DEBUG_PRINTF)");

    pipeline_init();
    uart_tx_reset();
    build_frame_with_flags(0U, PROTO_FLAG_PAIR_MAHA_HDC);
    pipeline_run();
    TEST_ASSERT_EQUAL_INT_MESSAGE(22, uart_tx_count,
        "PAIR_MAHA_HDC (0xA0) doit produire 22 B (early return)");
}
