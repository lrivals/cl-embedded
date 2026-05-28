/**
 * pipeline.c — Pipeline MVP : UART frame → Mahalanobis → LED + réponse
 *
 * Protocole UART v2 (little-endian, binary) :
 *   Réception : [MAGIC:2B][VERSION:1B][TASK_ID:1B][TIMESTAMP_MS:4B]
 *               [N:1B][features:f32×N][label:1B][FLAGS:1B][CRC8:1B]
 *   Réponse   : [pred:u8][conf:f32][lat_us:u32][ram_b:u16][thr:u16][status:u8] = 14 B
 *
 * Mesure latence via profiling.c (DWT cycle counter).
 */

#include "pipeline.h"
#include "profiling.h"
#include "model_weights.h"
#include "mahalanobis.h"
#include "metrics.h"
#include "tinyol.h"
#include "ewc_head.h"
#include "stm32f4xx.h"
#include <math.h>
#include <string.h>
#ifdef DEBUG_PRINTF
#include <stdio.h>
#endif

#define PROTO_MAGIC0 0xCDU   /* MAGIC=0xABCD little-endian : octet bas en premier */
#define PROTO_MAGIC1 0xABU
#define PROTO_MAX_N  16U

/* ── Globals statiques des 3 modèles (S2006 — profiling RAM Gap 2) ─────── */
/* MEM: 128 B @ FP32 en .bss                                               */
MahalanobisDetector g_detector;

/* MEM: TinyOLEncoder ~2.88 Ko @ FP32 + TinyOLDecoder ~2.77 Ko @ FP32 en .bss */
TinyOLEncoder g_tinyol_enc;
TinyOLDecoder g_tinyol_dec;

/* MEM: ~9.5 Ko @ FP32 en .bss */
EWCHead g_ewc_head;

/* MEM: métriques on-board — 302 B @ SRAM */
static OnlineAccuracy    g_acc;
static OnlineAUROC       g_auroc;
static ForgettingTracker g_fgt;
static uint8_t           g_current_task_id;

/* Contexte de la trame courante */
static uint8_t  g_recv_label;
static uint8_t  g_recv_version;
static uint8_t  g_recv_task_id;
static uint32_t g_recv_timestamp_ms;
static uint8_t  g_recv_flags;

/* ── Helpers UART polling (USART3, configuré par hw_uart_init) ─────────── */

#ifndef TEST_MODE
static uint8_t uart_getbyte(void)
{
    while (!(USART3->SR & (1U << 5))) {}   /* attendre RXNE */
    return (uint8_t)USART3->DR;
}

static void uart_send_byte(uint8_t b)
{
    while (!(USART3->SR & (1U << 7))) {}   /* attendre TXE */
    USART3->DR = b;
}
#else
/* Mocks définis dans tests/test_pipeline.c */
extern uint8_t uart_getbyte(void);
extern void    uart_send_byte(uint8_t b);
#endif

/* CRC8 polynomial 0x07 — identique à sensor_sim.py */
static uint8_t proto_crc8(const uint8_t *d, int len)
{
    uint8_t crc = 0;
    for (int i = 0; i < len; i++) {
        crc ^= d[i];
        for (int j = 0; j < 8; j++)
            crc = (crc & 0x80U) ? (uint8_t)((crc << 1) ^ 0x07U) : (uint8_t)(crc << 1);
    }
    return crc;
}

/* Réponse v2 : 14 B = [pred:u8][conf:f32][lat_us:u32][ram_b:u16][thr:u16][status:u8] */
static void uart_send_response_v2(uint8_t pred, float conf,
                                   uint32_t lat_us, uint8_t status)
{
    (void)lat_us;   /* lat_us encodé via profiling_encode() ci-dessous */
    union { float f; uint8_t b[4]; } uc;
    uint8_t buf[PROFILING_ENCODED_SIZE];   /* 8 B : [lat_us][ram_b][thr] */

    profiling_encode(buf);

    /* pred */
    uart_send_byte(pred);

    /* conf (f32 little-endian) */
    uc.f = conf;
    uart_send_byte(uc.b[0]); uart_send_byte(uc.b[1]);
    uart_send_byte(uc.b[2]); uart_send_byte(uc.b[3]);

    /* lat_us (u32 little-endian) — depuis profiling */
    uart_send_byte(buf[0]); uart_send_byte(buf[1]);
    uart_send_byte(buf[2]); uart_send_byte(buf[3]);

    /* ram_used_b (u16 little-endian) */
    uart_send_byte(buf[4]); uart_send_byte(buf[5]);

    /* throughput (u16 little-endian) */
    uart_send_byte(buf[6]); uart_send_byte(buf[7]);

    /* status */
    uart_send_byte(status);
}

/* Réponse v3 : 21 B = [pred:u8][conf:f32][lat_us:u32][acc:f32][auroc:f32][forgetting:f32] */
static void uart_send_response_v3(uint8_t pred, float conf,
                                   uint32_t lat_us, uint8_t status,
                                   const MetricsSnapshot *snap)
{
    (void)lat_us;   /* lat_us encodé via profiling_encode() */
    (void)status;   /* non utilisé en v3 — métriques portent l'info */
    union { float f; uint8_t b[4]; } uc;
    uint8_t prof_buf[PROFILING_ENCODED_SIZE];   /* 8 B */
    uint8_t snap_buf[METRICS_SNAPSHOT_SIZE];    /* 12 B */

    profiling_encode(prof_buf);
    metrics_encode_snapshot(snap, snap_buf);

    /* pred */
    uart_send_byte(pred);

    /* conf (f32 little-endian) */
    uc.f = conf;
    uart_send_byte(uc.b[0]); uart_send_byte(uc.b[1]);
    uart_send_byte(uc.b[2]); uart_send_byte(uc.b[3]);

    /* lat_us (u32 little-endian) — depuis profiling */
    uart_send_byte(prof_buf[0]); uart_send_byte(prof_buf[1]);
    uart_send_byte(prof_buf[2]); uart_send_byte(prof_buf[3]);

    /* MetricsSnapshot — 12 B : [acc:f32][auroc:f32][forgetting:f32] */
    for (int i = 0; i < (int)METRICS_SNAPSHOT_SIZE; i++)
        uart_send_byte(snap_buf[i]);
}

/* ── LED ─────────────────────────────────────────────────────────────────── */

void led_set(int state)
{
#ifndef TEST_MODE
    if (state) {
        GPIOA->BSRR = (1U << LED_PIN);           /* PA5 HIGH — anomalie */
    } else {
        GPIOA->BSRR = (1U << (LED_PIN + 16U));   /* PA5 LOW  — normal   */
    }
#else
    (void)state;
#endif
}

/* ── Normalisation Z-score (stats figées en Flash) ─────────────────────── */

static void normalize_zscore(float *x, int dim)
{
    for (int i = 0; i < dim; i++) {
        x[i] = (x[i] - ZSCORE_MEAN[i]) / ZSCORE_STD[i];
    }
}

/* ── Réception trame UART v2 avec synchronisation MAGIC + vérification CRC */

void uart_receive_sample(float *buf)
{
    uint8_t payload[3U + 4U + 1U + PROTO_MAX_N * 4U + 2U];  /* header étendu */
    int pay_idx = 0;

resync:
    while (uart_getbyte() != PROTO_MAGIC0) {}
    if (uart_getbyte() != PROTO_MAGIC1) goto resync;

    payload[pay_idx++] = PROTO_MAGIC0;
    payload[pay_idx++] = PROTO_MAGIC1;

    g_recv_version = uart_getbyte();
    payload[pay_idx++] = g_recv_version;
    if (g_recv_version != PROTO_VERSION_V2) goto resync;   /* filtre v2 uniquement */

    g_recv_task_id = uart_getbyte();
    payload[pay_idx++] = g_recv_task_id;

    /* TIMESTAMP_MS : 4 octets little-endian */
    g_recv_timestamp_ms = 0U;
    for (int k = 0; k < 4; k++) {
        uint8_t b = uart_getbyte();
        payload[pay_idx++] = b;
        g_recv_timestamp_ms |= ((uint32_t)b << (k * 8U));
    }

    uint8_t n = uart_getbyte();
    payload[pay_idx++] = n;
    if (n > PROTO_MAX_N) goto resync;

    /* features */
    for (uint8_t i = 0; i < n; i++) {
        union { float f; uint8_t b[4]; } u;
        for (int k = 0; k < 4; k++) {
            u.b[k] = uart_getbyte();
            payload[pay_idx++] = u.b[k];
        }
        if (i < MAHA_DIM) buf[i] = u.f;
    }
    for (uint8_t i = n; i < MAHA_DIM; i++) buf[i] = 0.0f;

    uint8_t label = uart_getbyte();
    payload[pay_idx++] = label;
    g_recv_label = label;

    g_recv_flags = uart_getbyte();
    payload[pay_idx++] = g_recv_flags;

    uint8_t recv_crc = uart_getbyte();
    if (proto_crc8(payload, pay_idx) != recv_crc) goto resync;
}

/* ── pipeline_set_task ──────────────────────────────────────────────────── */

void pipeline_set_task(uint8_t task_id)
{
    g_current_task_id = task_id;
}

/* ── Init pipeline ─────────────────────────────────────────────────────── */

void pipeline_init(void)
{
    maha_init(&g_detector, MAHA_THRESHOLD_INIT, MAHA_EMA_ALPHA);
    g_ewc_head.lambda = 400.0f;   /* configs/board_ewc.yaml : EWC_LAMBDA */
    acc_init(&g_acc);
    auroc_init(&g_auroc);
    fgt_init(&g_fgt);
    g_current_task_id = 0U;

    for (int i = 0; i < MAHA_DIM; i++) {
        g_detector.mean[i] = MAHA_MEAN_INIT[i];
        for (int j = 0; j < MAHA_DIM; j++) {
            g_detector.precision[i][j] = MAHA_PRECISION_INIT[i][j];
        }
    }

    /* Init TinyOL depuis Flash (poids placeholder — remplacer via export_weights_c.py) */
    memcpy(g_tinyol_enc.w_enc1, TINYOL_W_ENC1, sizeof(g_tinyol_enc.w_enc1));
    memcpy(g_tinyol_enc.b_enc1, TINYOL_B_ENC1, sizeof(g_tinyol_enc.b_enc1));
    memcpy(g_tinyol_enc.w_enc2, TINYOL_W_ENC2, sizeof(g_tinyol_enc.w_enc2));
    memcpy(g_tinyol_enc.b_enc2, TINYOL_B_ENC2, sizeof(g_tinyol_enc.b_enc2));
    memcpy(g_tinyol_dec.w_dec1, TINYOL_W_DEC1, sizeof(g_tinyol_dec.w_dec1));
    memcpy(g_tinyol_dec.b_dec1, TINYOL_B_DEC1, sizeof(g_tinyol_dec.b_dec1));
    memcpy(g_tinyol_dec.w_dec2, TINYOL_W_DEC2, sizeof(g_tinyol_dec.w_dec2));
    memcpy(g_tinyol_dec.b_dec2, TINYOL_B_DEC2, sizeof(g_tinyol_dec.b_dec2));

#ifndef TEST_MODE
    /* Activer GPIOA pour LED LD2 */
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN;
    GPIOA->MODER &= ~(0x3U << (LED_PIN * 2U));
    GPIOA->MODER |=  (0x1U << (LED_PIN * 2U));   /* Output mode */
    GPIOA->OTYPER &= ~(0x1U << LED_PIN);          /* Push-pull */
#endif
}

#ifdef TEST_MODE
/* Expose uart_send_response_v2 pour test direct de la taille de trame */
void test_pipeline_send_response_v2(uint8_t pred, float conf,
                                     uint32_t lat_us, uint8_t status)
{
    uart_send_response_v2(pred, conf, lat_us, status);
}

/* Expose uart_send_response_v3 pour test direct de la taille de trame */
void test_pipeline_send_response_v3(uint8_t pred, float conf,
                                     uint32_t lat_us, uint8_t status,
                                     const MetricsSnapshot *snap)
{
    uart_send_response_v3(pred, conf, lat_us, status, snap);
}
#endif

/* ── Boucle principale ─────────────────────────────────────────────────── */

void pipeline_run(void)
{
    float raw[MAHA_DIM];   /* MEM: 20 B @ FP32 (stack) */

    uart_receive_sample(raw);

    /* RESET : réinitialise EWC avant une nouvelle expérience.
     * raw[0] encode la nouvelle valeur lambda (> 0 pour la modifier). */
    if (g_recv_flags & PROTO_FLAG_RESET) {
        float new_lambda = raw[0];
        if (new_lambda > 0.0f) g_ewc_head.lambda = new_lambda;
        ewc_init(&g_ewc_head);
        acc_init(&g_acc);
        auroc_init(&g_auroc);
        fgt_init(&g_fgt);
        g_current_task_id = 0U;
        MetricsSnapshot snap_reset = { .accuracy = 0.0f, .auroc = 0.0f, .forgetting = 0.0f };
        uart_send_response_v3(0, 1.0f, 0U, PROTO_STATUS_OK, &snap_reset);
        return;
    }

    profiling_start();   /* Démarre le chrono DWT */

    int   pred;
    float confidence;

    if (g_recv_flags & PROTO_FLAG_EWC_MODE) {
        /* ── Chemin EWC : forward softmax + update SGD + consolidation ─────── */
        float logits[EWC_OUT];   /* MEM: 8 B @ FP32 (stack) */
        ewc_forward(&g_ewc_head, raw, logits);
        pred = (logits[1] > logits[0]) ? 1 : 0;

        float e0 = expf(logits[0]);
        float e1 = expf(logits[1]);
        confidence = e1 / (e0 + e1);   /* prob(faulty=1) pour AUROC */

        if (g_recv_flags & PROTO_FLAG_UPDATE) {
            ewc_sgd_step(&g_ewc_head, raw, (int)g_recv_label);
        }
        if (g_recv_flags & PROTO_FLAG_CONSOLIDATE) {
            ewc_consolidate(&g_ewc_head, EWC_FISHER_DECAY);
            g_current_task_id = g_recv_task_id;
        }
        auroc_update(&g_auroc, confidence, (int)g_recv_label);
    } else {
        /* ── Chemin Mahalanobis (comportement historique) ───────────────────── */
        normalize_zscore(raw, MAHA_DIM);
        float score = maha_score(&g_detector, raw);
        int   anomaly = (score > g_detector.threshold) ? 1 : 0;
        pred       = anomaly;
        confidence = 1.0f / (1.0f + score);
        led_set(anomaly ? LED_ON : LED_OFF);

        if ((g_recv_flags & PROTO_FLAG_UPDATE) && !anomaly) {
            maha_update(&g_detector, raw);
        }
        if (g_recv_flags & PROTO_FLAG_CONSOLIDATE) {
            ewc_consolidate(&g_ewc_head, EWC_FISHER_DECAY);
            g_current_task_id = g_recv_task_id;
        }
        auroc_update(&g_auroc, score, (int)g_recv_label);
    }

    profiling_stop();    /* Arrête le chrono, calcule throughput */

    acc_update(&g_acc, pred, (int)g_recv_label);
    fgt_update(&g_fgt, g_current_task_id, acc_compute(&g_acc));

    MetricsSnapshot snap = {
        .accuracy   = acc_compute(&g_acc),
        .auroc      = auroc_compute(&g_auroc),
        .forgetting = fgt_avg_forgetting(&g_fgt),
    };

    uart_send_response_v3((uint8_t)pred, confidence,
                           profiling_get_latency_us(), PROTO_STATUS_OK, &snap);

#ifdef DEBUG_PRINTF
    char dbg[80];
    snprintf(dbg, sizeof(dbg), "score=%.4f pred=%d lat=%lu us\r\n",
             (double)confidence, pred, (unsigned long)profiling_get_latency_us());
    for (int i = 0; dbg[i]; i++) uart_send_byte((uint8_t)dbg[i]);
#endif
}
