/**
 * pipeline.c — Pipeline MVP : UART frame → Mahalanobis → LED + réponse
 *
 * Protocole UART (little-endian, binary) — conforme sensor_sim.py :
 *   Réception : [MAGIC 0xABCD:2B][N:1B][features:f32×N][label:1B][CRC8:1B]
 *   Réponse   : [pred:u8][confidence:f32][latency_us:u32]  = 9 B
 *
 * Mesure latence via DWT cycle counter (activé par hw_dwt_calibrate au boot).
 */

#include "pipeline.h"
#include "model_weights.h"
#include "mahalanobis.h"
#include "stm32f4xx.h"

#define SYSCLK_HZ    180000000U
#define PROTO_MAGIC0 0xCDU   /* MAGIC=0xABCD little-endian : octet bas en premier */
#define PROTO_MAGIC1 0xABU
#define PROTO_MAX_N  16U

#define DWT_CYCCNT  (*(volatile uint32_t *)0xE0001004UL)

/* ── Détecteur global statique ─────────────────────────────────────────── */
/* MEM: 128 B @ FP32 en .bss                                               */
MahalanobisDetector g_detector;

/* Label reçu dans la trame courante (loopback test) */
static uint8_t g_recv_label;

/* ── Helpers UART polling (USART3, configuré par hw_uart_init) ─────────── */

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

/* Réponse firmware → sensor_sim : [pred:u8][conf:f32][lat_us:u32] = 9 B */
static void uart_send_response(uint8_t pred, float conf, uint32_t lat_us)
{
    union { float f; uint8_t b[4]; } uc;

    uart_send_byte(pred);
    uc.f = conf;
    uart_send_byte(uc.b[0]); uart_send_byte(uc.b[1]);
    uart_send_byte(uc.b[2]); uart_send_byte(uc.b[3]);
    uart_send_byte((uint8_t)(lat_us));
    uart_send_byte((uint8_t)(lat_us >>  8));
    uart_send_byte((uint8_t)(lat_us >> 16));
    uart_send_byte((uint8_t)(lat_us >> 24));
}

/* ── LED ─────────────────────────────────────────────────────────────────── */

void led_set(int state)
{
    if (state) {
        GPIOA->BSRR = (1U << LED_PIN);           /* PA5 HIGH — anomalie */
    } else {
        GPIOA->BSRR = (1U << (LED_PIN + 16U));   /* PA5 LOW  — normal   */
    }
}

/* ── Normalisation Z-score (stats figées en Flash) ─────────────────────── */

static void normalize_zscore(float *x, int dim)
{
    for (int i = 0; i < dim; i++) {
        x[i] = (x[i] - ZSCORE_MEAN[i]) / ZSCORE_STD[i];
    }
}

/* ── Réception trame UART avec synchronisation MAGIC + vérification CRC ── */

void uart_receive_sample(float *buf)
{
    /* payload = [MAGIC0 MAGIC1 N features... label] sans CRC */
    uint8_t payload[3U + PROTO_MAX_N * 4U + 1U];

resync:
    while (uart_getbyte() != PROTO_MAGIC0) {}
    if (uart_getbyte() != PROTO_MAGIC1) goto resync;

    payload[0] = PROTO_MAGIC0;
    payload[1] = PROTO_MAGIC1;

    uint8_t n = uart_getbyte();
    if (n > PROTO_MAX_N) goto resync;
    payload[2] = n;

    for (uint8_t i = 0; i < n; i++) {
        union { float f; uint8_t b[4]; } u;
        u.b[0] = uart_getbyte(); payload[3U + i * 4U]      = u.b[0];
        u.b[1] = uart_getbyte(); payload[3U + i * 4U + 1U] = u.b[1];
        u.b[2] = uart_getbyte(); payload[3U + i * 4U + 2U] = u.b[2];
        u.b[3] = uart_getbyte(); payload[3U + i * 4U + 3U] = u.b[3];
        if (i < MAHA_DIM) buf[i] = u.f;
    }
    for (uint8_t i = n; i < MAHA_DIM; i++) buf[i] = 0.0f;  /* pad si n < MAHA_DIM */

    uint8_t label = uart_getbyte();
    payload[3U + n * 4U] = label;
    g_recv_label = label;

    uint8_t recv_crc = uart_getbyte();
    if (proto_crc8(payload, (int)(3U + n * 4U + 1U)) != recv_crc) goto resync;
}

/* ── Init pipeline ─────────────────────────────────────────────────────── */

void pipeline_init(void)
{
    maha_init(&g_detector, MAHA_THRESHOLD_INIT, MAHA_EMA_ALPHA);

    for (int i = 0; i < MAHA_DIM; i++) {
        g_detector.mean[i] = MAHA_MEAN_INIT[i];
        for (int j = 0; j < MAHA_DIM; j++) {
            g_detector.precision[i][j] = MAHA_PRECISION_INIT[i][j];
        }
    }

    /* Activer GPIOA pour LED LD2 */
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN;
    GPIOA->MODER &= ~(0x3U << (LED_PIN * 2U));
    GPIOA->MODER |=  (0x1U << (LED_PIN * 2U));   /* Output mode */
    GPIOA->OTYPER &= ~(0x1U << LED_PIN);          /* Push-pull */
}

/* ── Boucle principale ─────────────────────────────────────────────────── */

void pipeline_run(void)
{
    float raw[MAHA_DIM];   /* MEM: 20 B @ FP32 (stack) */

    uart_receive_sample(raw);

    /* Chrono DWT : mesure uniquement l'inférence (Z-score + Mahalanobis) */
    uint32_t t0 = DWT_CYCCNT;

    normalize_zscore(raw, MAHA_DIM);

    float score   = maha_score(&g_detector, raw);
    int   anomaly = (score > g_detector.threshold) ? 1 : 0;
    led_set(anomaly ? LED_ON : LED_OFF);

    if (!anomaly) maha_update(&g_detector, raw);

    uint32_t dt_cycles = DWT_CYCCNT - t0;
    uint32_t lat_us    = dt_cycles / (SYSCLK_HZ / 1000000U);
    float    confidence = 1.0f / (1.0f + score);

    uart_send_response((uint8_t)anomaly, confidence, lat_us);
}
