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
#include "model_weights_rul.h"
#include "model_weights_multiclass.h"
#include "model_weights_ewc.h"
#include "mahalanobis.h"
#include "metrics.h"
#include "tinyol.h"
#include "ewc_head.h"
#include "ewc_head_regression.h"
#include "ewc_head_multiclass.h"
#include "ewc_head_int8.h"
#ifdef EWC_INT8_V2
#include "ewc_head_int8_v2.h"
#include "ewc_head_int8_v2_weights.h"  /* généré (vide par défaut) — S3908 */
#endif
#include "hdc.h"
#include "meta_head.h"
#include "mahalanobis_q15.h"
#include "mahalanobis_q15_weights.h"   /* généré (vide par défaut) — S3407 */
#ifdef MAHA_INT8
#include "mahalanobis_int8.h"
#include "mahalanobis_int8_weights.h"  /* généré (vide par défaut) — S2912 */
#endif
#ifdef EWC_AUTO_UPDATE
#include "drift_detector.h"            /* gate de mise à jour autonome — S3803 */
#include "drift_thresholds.h"          /* généré (neutre par défaut) — S3803 */
#endif
#include "stm32f4xx.h"
#include <math.h>
#include <string.h>
#ifdef DEBUG_PRINTF
#include <stdio.h>
#endif

#define PROTO_MAGIC0 0xCDU   /* MAGIC=0xABCD little-endian : octet bas en premier */
#define PROTO_MAGIC1 0xABU
/* Nombre max de features par trame UART — surchargeable au build (S3506).
 * Défaut 16U (conditions 5feat/best ≤ 16). La condition `all`×cmapss (21 capteurs)
 * se construit avec `-DPROTO_MAX_N=21`. Coût .bss : g_stream_storage croît de
 * STREAM_BUF_W*(PROTO_MAX_N-16)*4 octets (à W=5 : +100 B pour 21) ; raw[]/payload[]
 * sont sur la pile. Défaut inchangé ⇒ .bss 5feat bit-identique. */
#ifndef PROTO_MAX_N
#define PROTO_MAX_N  16U
#endif

/* Dimension native des poids placeholder de model_weights.h (générés à 5 par
 * export_weights_c.py — header jamais édité à la main). Si la condition de build
 * change MAHA_DIM/TINYOL_IN, ces tableaux figés ne sont plus copiables → init
 * neutre (maha_init = identité ; TinyOL = .bss à zéro). Les poids réels par
 * condition sont regénérés en S3507. EWC suit déjà EWC_HEAD_WEIGHTS_PROVIDED. */
#ifndef WEIGHTS_NATIVE_DIM
#define WEIGHTS_NATIVE_DIM 5
#endif

/* S3507 : dim native PAR MODÈLE des poids exportés (condition all/best → k≠5,
 * et k peut différer par modèle en `best`). Les headers générés par
 * export_weights_c.py définissent MAHA_NATIVE_DIM (model_weights.h) et
 * EWC_HEAD_NATIVE_DIM (model_weights_ewc.h). Fallback sur WEIGHTS_NATIVE_DIM
 * pour les headers historiques (5feat) → 0 régression. */
#ifndef MAHA_NATIVE_DIM
#define MAHA_NATIVE_DIM WEIGHTS_NATIVE_DIM
#endif
#ifndef EWC_HEAD_NATIVE_DIM
#define EWC_HEAD_NATIVE_DIM WEIGHTS_NATIVE_DIM
#endif

/* ── Globals statiques des 3 modèles (S2006 — profiling RAM Gap 2) ─────── */
/* MEM: 128 B @ FP32 en .bss                                               */
MahalanobisDetector g_detector;

/* MEM: TinyOLEncoder ~2.88 Ko @ FP32 + TinyOLDecoder ~2.77 Ko @ FP32 en .bss */
TinyOLEncoder g_tinyol_enc;
TinyOLDecoder g_tinyol_dec;

/* MEM: ~9.5 Ko @ FP32 en .bss */
EWCHead g_ewc_head;

/* MEM: EWCHeadInt8 ~2.4 Ko @ INT8 + ~1.2 Ko biais FP32 en .bss
 * Cf. S2221 — ewc_head_int8.h commentaires MEM détaillés */
EWCHeadInt8 g_ewc_int8;

#ifdef EWC_INT8_V2
/* Sprint 39 (S3915) — tête INT8 v2 : acc int32, scales par-canal + activations calibrées.
 * Sélectionnée à la compilation (-DEWC_INT8_V2) car le nibble protocole est saturé (mirroir
 * -DMAHA_INT8, S2912) : le chemin 0x40 (FRAME_FLAGS_INT8_MODE) exécute le v2 au lieu du v1.
 * Update en ligne (S4002) : SGD sur la tête FP32 maître g_ewc_head puis requantification
 * v2 (les scales par-canal restent la vue quantifiée de g_ewc_head, act_max figé).
 * MEM: EWCHeadInt8V2 ~704 B poids int8 + scales/biais FP32 en .bss. */
EWCHeadInt8V2 g_ewc_int8_v2;
/* act_max de calibration mémorisé à l'init (header EWC_V2_ACT_MAX ou neutre) → réutilisé
 * pour la requantification par échantillon en mode online (S4002). Figé = parité miroir PC. */
static float g_v2_act_max[3] = {1.0f, 1.0f, 1.0f};
#endif

/* MEM: HDCClassifier ~27.7 Ko @ FP32 en .bss
 * ATTENTION : avec g_ewc_head (~9.5 Ko) + g_tinyol (~5.7 Ko) + g_detector (~128 B)
 * + g_hdc (~27.7 Ko) = ~43 Ko → dans le budget 64 Ko. */
HDCClassifier g_hdc;

/* Sprint 29 — modèles INT8 firmware (S2901 HDC INT8, S2902 TinyOL INT8) */
HDCInt8           g_hdc_int8;     /* MEM: ~34.8 Ko @ INT8/INT16 en .bss */
TinyOLEncoderInt8 g_tinyol_int8;  /* MEM: ~0.9 Ko @ INT8 + biais FP32 */
OtOHeadInt8       g_oto_int8;     /* MEM: tête OtO INT8 + maîtres FP32 */
/* Buffer encodage HDC INT8 — fichier-scope pour éviter 2 Ko de pile dans pipeline_run. */
static int8_t g_hv_int8[HDC_I_D];   /* MEM: 2 048 B @ INT8 en .bss */

/* ── Instrumentation streaming/buffer (Sprint 34 S3403) ─────────────────────
 * Buffer circulaire générique (ring_buffer.c) alimenté par chaque trame reçue.
 * Sa taille .bss = STREAM_BUF_W * PROTO_MAX_N * sizeof(float) varie avec W (rebuild
 * `make all STREAM_BUF_W=<W>`) → mesurée par arm-none-eabi-size pour l'étude S3403.
 * N'altère PAS le chemin d'inférence par-trame (remplissage observé uniquement). */
#ifndef STREAM_BUF_W
#define STREAM_BUF_W 5
#endif
static float      g_stream_storage[STREAM_BUF_W * PROTO_MAX_N];  /* MEM: W*16*4 B @ FP32 */
static RingBuffer g_stream_rb;

/* MEM: EWCHeadReg ~8.9 Ko @ FP32 en .bss */
EWCHeadReg g_ewc_reg;

/* MEM: EWCHeadMC ~14 Ko @ FP32 en .bss (EWC_MC_N_CLASSES=10) */
EWCHeadMC g_ewc_mc;

/* MEM: MetaHead ~20 B @ FP32 (logreg) en .bss — méta-modèle stacking (Sprint 31) */
MetaHead g_meta;

/* MEM: MahalanobisQ15 = d×1 B (mu int8) + d²×2 B (sigma int16) + 5×4 B (scales/seuils)
 * = 75 B @ d=5 en .bss — Mahalanobis sigma_inv Q15 (MAHA_Q15_MODE, S3407) */
MahalanobisQ15 g_maha_q15;

#ifdef MAHA_INT8
/* MEM: MahalanobisInt8 = d×1 B (mu) + d²×1 B (sigma) + 6×4 B (scales/zp/seuils)
 * = 54 B @ d=5 en .bss — Mahalanobis INT8 affine (S2912, sélection -DMAHA_INT8).
 * Remplace le chemin Mahalanobis par défaut (FP32) ; le nibble de flags étant saturé,
 * la sélection est à la compilation (driver re-flashe par cellule). */
MahalanobisInt8 g_maha_int8;
#endif

#ifdef EWC_AUTO_UPDATE
/* MEM: DriftDetector ≈ DRIFT_WINDOW_MAX×4 B (~256 B) en .bss — gate autonome S3803.
 * Présent uniquement sous -DEWC_AUTO_UPDATE (build par défaut inchangé). */
static DriftDetector g_drift;
static uint32_t      g_n_updates;   /* SGD réellement déclenchés par le gate (mesure board) */
static DriftVerdict  g_last_verdict; /* verdict du dernier échantillon (0=NORMAL,1=FAULT,2=DRIFT) — S3805 */
#endif

/* MEM: métriques on-board — 302 B @ SRAM + 16 B OnlineRMSE + ~202 B OnlineF1Macro */
static OnlineAccuracy    g_acc;
static OnlineAUROC       g_auroc;
static ForgettingTracker g_fgt;
static OnlineRMSE        g_rmse;
static OnlineF1Macro     g_f1;
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

/* Réponse v3 : 23 B = [pred:u8][conf:f32][lat_us:u32][ram_b:u16][acc:f32][auroc:f32][forgetting:f32] */
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

    /* ram_b (u16 little-endian) — taille .bss calculée au link time */
    uart_send_byte(prof_buf[4]); uart_send_byte(prof_buf[5]);

    /* MetricsSnapshot — 12 B : [acc:f32][auroc:f32][forgetting:f32] */
    for (int i = 0; i < (int)METRICS_SNAPSHOT_SIZE; i++)
        uart_send_byte(snap_buf[i]);
}

/* Sprint 27 — Réponse 25 B pour DUAL_MODE
 * Layout : [pred_fault:u8][conf_fault:f32][rul_pred:f32][lat_us:u32]
 *          [f1_macro:f32][rmse_rul:f32][forgetting:f32]             */
static void uart_send_response_dual(uint8_t pred_fault, float conf_fault,
                                     float rul_pred, float f1_macro,
                                     float rmse_rul, float forgetting)
{
    union { float f; uint8_t b[4]; } uc;
    uint8_t prof_buf[PROFILING_ENCODED_SIZE];
    profiling_encode(prof_buf);

    uart_send_byte(pred_fault);         /* offset 0 — classe faute prédite */

    uc.f = conf_fault;                  /* offset 1–4 — confiance softmax */
    uart_send_byte(uc.b[0]); uart_send_byte(uc.b[1]);
    uart_send_byte(uc.b[2]); uart_send_byte(uc.b[3]);

    uc.f = rul_pred;                    /* offset 5–8 — RUL prédit (float) */
    uart_send_byte(uc.b[0]); uart_send_byte(uc.b[1]);
    uart_send_byte(uc.b[2]); uart_send_byte(uc.b[3]);

    /* offset 9–12 — latence combinée DWT (u32, µs) */
    uart_send_byte(prof_buf[0]); uart_send_byte(prof_buf[1]);
    uart_send_byte(prof_buf[2]); uart_send_byte(prof_buf[3]);

    uc.f = f1_macro;                    /* offset 13–16 — F1-macro faute */
    uart_send_byte(uc.b[0]); uart_send_byte(uc.b[1]);
    uart_send_byte(uc.b[2]); uart_send_byte(uc.b[3]);

    uc.f = rmse_rul;                    /* offset 17–20 — RMSE RUL */
    uart_send_byte(uc.b[0]); uart_send_byte(uc.b[1]);
    uart_send_byte(uc.b[2]); uart_send_byte(uc.b[3]);

    uc.f = forgetting;                  /* offset 21–24 — forgetting moyen */
    uart_send_byte(uc.b[0]); uart_send_byte(uc.b[1]);
    uart_send_byte(uc.b[2]); uart_send_byte(uc.b[3]);
    /* Total : 1 + 4 + 4 + 4 + 4 + 4 + 4 = 25 B */
}

/* Sprint 30 — Réponse 22 B pour PAIR_MODE (Mahalanobis + supervisé)
 * Layout : [pred_maha:u8][score_maha:f32][pred_sup:u8][conf_sup:f32]
 *          [lat_us:u32][auroc_maha:f32][f1_sup:f32]                     */
static void uart_send_response_pair(uint8_t pred_maha, float score_maha,
                                    uint8_t pred_sup, float conf_sup,
                                    float auroc_maha, float f1_sup)
{
    union { float f; uint8_t b[4]; } uc;
    uint8_t prof_buf[PROFILING_ENCODED_SIZE];
    profiling_encode(prof_buf);

    uart_send_byte(pred_maha);          /* offset 0 — anomalie Mahalanobis */

    uc.f = score_maha;                  /* offset 1–4 — score Mahalanobis */
    uart_send_byte(uc.b[0]); uart_send_byte(uc.b[1]);
    uart_send_byte(uc.b[2]); uart_send_byte(uc.b[3]);

    uart_send_byte(pred_sup);           /* offset 5 — classe supervisée */

    uc.f = conf_sup;                    /* offset 6–9 — confiance supervisée */
    uart_send_byte(uc.b[0]); uart_send_byte(uc.b[1]);
    uart_send_byte(uc.b[2]); uart_send_byte(uc.b[3]);

    /* offset 10–13 — latence combinée DWT (u32, µs) */
    uart_send_byte(prof_buf[0]); uart_send_byte(prof_buf[1]);
    uart_send_byte(prof_buf[2]); uart_send_byte(prof_buf[3]);

    uc.f = auroc_maha;                  /* offset 14–17 — AUROC en ligne Maha */
    uart_send_byte(uc.b[0]); uart_send_byte(uc.b[1]);
    uart_send_byte(uc.b[2]); uart_send_byte(uc.b[3]);

    uc.f = f1_sup;                      /* offset 18–21 — F1 en ligne supervisé */
    uart_send_byte(uc.b[0]); uart_send_byte(uc.b[1]);
    uart_send_byte(uc.b[2]); uart_send_byte(uc.b[3]);
    /* Total : 1 + 4 + 1 + 4 + 4 + 4 + 4 = 22 B */
}

/* Sprint 31 — Réponse 27 B pour TRIPLE_MODE (PAIR + méta-modèle de stacking)
 * Layout : [pred_maha:u8][score_maha:f32][pred_sup:u8][conf_sup:f32][lat_us:u32]
 *          [auroc_maha:f32][f1_sup:f32]   ← identique PAIR (22 B)
 *          [pred_meta:u8][prob_meta:f32]  ← verdict méta (5 B)                  */
static void uart_send_response_triple(uint8_t pred_maha, float score_maha,
                                      uint8_t pred_sup, float conf_sup,
                                      float auroc_maha, float f1_sup,
                                      uint8_t pred_meta, float prob_meta)
{
    union { float f; uint8_t b[4]; } uc;
    uint8_t prof_buf[PROFILING_ENCODED_SIZE];
    profiling_encode(prof_buf);

    uart_send_byte(pred_maha);          /* offset 0 */

    uc.f = score_maha;                  /* offset 1–4 */
    uart_send_byte(uc.b[0]); uart_send_byte(uc.b[1]);
    uart_send_byte(uc.b[2]); uart_send_byte(uc.b[3]);

    uart_send_byte(pred_sup);           /* offset 5 */

    uc.f = conf_sup;                    /* offset 6–9 */
    uart_send_byte(uc.b[0]); uart_send_byte(uc.b[1]);
    uart_send_byte(uc.b[2]); uart_send_byte(uc.b[3]);

    /* offset 10–13 — latence combinée DWT (u32, µs) */
    uart_send_byte(prof_buf[0]); uart_send_byte(prof_buf[1]);
    uart_send_byte(prof_buf[2]); uart_send_byte(prof_buf[3]);

    uc.f = auroc_maha;                  /* offset 14–17 */
    uart_send_byte(uc.b[0]); uart_send_byte(uc.b[1]);
    uart_send_byte(uc.b[2]); uart_send_byte(uc.b[3]);

    uc.f = f1_sup;                      /* offset 18–21 */
    uart_send_byte(uc.b[0]); uart_send_byte(uc.b[1]);
    uart_send_byte(uc.b[2]); uart_send_byte(uc.b[3]);

    uart_send_byte(pred_meta);          /* offset 22 — verdict méta */

    uc.f = prob_meta;                   /* offset 23–26 — proba méta */
    uart_send_byte(uc.b[0]); uart_send_byte(uc.b[1]);
    uart_send_byte(uc.b[2]); uart_send_byte(uc.b[3]);
    /* Total : 22 + 1 + 4 = 27 B */
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
    if (g_recv_version != PROTO_VERSION_V2 && g_recv_version != PROTO_VERSION_V3) goto resync;

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
        if (i < PROTO_MAX_N) buf[i] = u.f;
    }
    for (uint8_t i = n; i < PROTO_MAX_N; i++) buf[i] = 0.0f;

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

/* S3205 : initialise g_ewc_head. Si des poids entraînés sont fournis en Flash
 * (model_weights_ewc.h régénéré par export_weights_c.py --ewc), on les charge
 * pour garantir la parité board↔PC ; sinon on retombe sur l'init Xavier
 * historique (apprentissage en ligne — aucune régression). */
static void ewc_head_load_or_init(EWCHead *h)
{
    ewc_init(h);   /* Xavier + zero fisher/star (préserve lambda) */
/* S3507 : poids exportés copiables quand EWC_IN == dim native du header généré
 * (EWC_HEAD_NATIVE_DIM, écrit par export_weights_c.py --ewc-head pour la condition).
 * Sinon (header absent / dim incohérente) → Xavier (apprentissage en ligne). */
#if defined(EWC_HEAD_WEIGHTS_PROVIDED) && (EWC_IN == EWC_HEAD_NATIVE_DIM)
    memcpy(h->w1, EWC_W1_INIT, sizeof(h->w1));
    memcpy(h->b1, EWC_B1_INIT, sizeof(h->b1));
    memcpy(h->w2, EWC_W2_INIT, sizeof(h->w2));
    memcpy(h->b2, EWC_B2_INIT, sizeof(h->b2));
    memcpy(h->w3, EWC_W3_INIT, sizeof(h->w3));
    memcpy(h->b3, EWC_B3_INIT, sizeof(h->b3));
    /* θ* = poids chargés (référence EWC pour la consolidation online) */
    memcpy(h->star_w1, EWC_W1_INIT, sizeof(h->star_w1));
    memcpy(h->star_w2, EWC_W2_INIT, sizeof(h->star_w2));
    memcpy(h->star_w3, EWC_W3_INIT, sizeof(h->star_w3));
#endif
}

void pipeline_init(void)
{
    maha_init(&g_detector, MAHA_THRESHOLD_INIT, MAHA_EMA_ALPHA);
    g_ewc_head.lambda = 400.0f;   /* configs/board_ewc.yaml : EWC_LAMBDA */
    ewc_head_load_or_init(&g_ewc_head);
    ewc_int8_init(&g_ewc_int8);
    /* Sprint 26 : nouvelles têtes EWC régression + multi-classes */
    g_ewc_reg.lambda = 400.0f;
    ewc_reg_init(&g_ewc_reg);
    /* Charger poids entraînés depuis Flash (model_weights_rul.h — export_weights_ewc_rul.py) */
    memcpy(g_ewc_reg.w1, EWC_REG_W1_INIT, sizeof(g_ewc_reg.w1));
    memcpy(g_ewc_reg.b1, EWC_REG_B1_INIT, sizeof(g_ewc_reg.b1));
    memcpy(g_ewc_reg.w2, EWC_REG_W2_INIT, sizeof(g_ewc_reg.w2));
    memcpy(g_ewc_reg.b2, EWC_REG_B2_INIT, sizeof(g_ewc_reg.b2));
    memcpy(g_ewc_reg.w3, EWC_REG_W3_INIT, sizeof(g_ewc_reg.w3));
    memcpy(g_ewc_reg.b3, EWC_REG_B3_INIT, sizeof(g_ewc_reg.b3));

    g_ewc_mc.lambda = 400.0f;
    ewc_mc_init(&g_ewc_mc);
    /* Charger poids entraînés depuis Flash (model_weights_multiclass.h — export_weights_ewc_multiclass.py) */
    memcpy(g_ewc_mc.w1, EWC_MC_W1_INIT, sizeof(g_ewc_mc.w1));
    memcpy(g_ewc_mc.b1, EWC_MC_B1_INIT, sizeof(g_ewc_mc.b1));
    memcpy(g_ewc_mc.w2, EWC_MC_W2_INIT, sizeof(g_ewc_mc.w2));
    memcpy(g_ewc_mc.b2, EWC_MC_B2_INIT, sizeof(g_ewc_mc.b2));
    memcpy(g_ewc_mc.w3, EWC_MC_W3_INIT, sizeof(g_ewc_mc.w3));
    memcpy(g_ewc_mc.b3, EWC_MC_B3_INIT, sizeof(g_ewc_mc.b3));
    online_rmse_init(&g_rmse);
    online_f1_init(&g_f1);
    /* Sprint 31 — méta-modèle stacking (poids depuis meta_weights.h via export_weights_c.py) */
    meta_init(&g_meta);
    /* Sprint 36 (S3610) : charger la tête INT8 depuis les poids FP32 entraînés (résout
     * TODO(dorra)). Sans cela, le chemin 0x40 (FRAME_FLAGS_INT8_MODE) exécutait une tête
     * Xavier non entraînée. ewc_head_load_or_init() a déjà rempli g_ewc_head ci-dessus
     * (poids exportés si EWC_HEAD_WEIGHTS_PROVIDED, Xavier en fallback → 0 régression
     * FP32). La conversion ne touche que g_ewc_int8 ; le chemin FP32 reste inchangé. */
    ewc_int8_from_fp32(&g_ewc_int8, &g_ewc_head);
#ifdef EWC_INT8_V2
    /* Sprint 39 (S3915) : quantifie la tête FP32 entraînée (g_ewc_head) en INT8 v2 avec
     * scales par-canal + bornes d'activation calibrées. act_max provient du header généré
     * (EWC_V2_ACT_MAX) ; à défaut (header vide) → bornes unitaires neutres. Le QMAX dépend
     * de la variante (int8/q15/mixed) → on passe act_max brut, pas scale_act figé.
     * Parité par construction avec l'émulateur forward_quant(per_channel_int8|q15). */
#if defined(EWC_INT8_V2_WEIGHTS_PROVIDED)
    g_v2_act_max[0] = EWC_V2_ACT_MAX[0]; g_v2_act_max[1] = EWC_V2_ACT_MAX[1];
    g_v2_act_max[2] = EWC_V2_ACT_MAX[2];
#else
    g_v2_act_max[0] = 1.0f; g_v2_act_max[1] = 1.0f; g_v2_act_max[2] = 1.0f;
#endif
    ewc_int8_v2_from_fp32_calib(&g_ewc_int8_v2, &g_ewc_head, g_v2_act_max);
#endif
    hdc_init(&g_hdc);
#ifdef EWC_AUTO_UPDATE
    /* S3803 — gate de mise à jour autonome : seuils depuis inc/drift_thresholds.h
     * (généré par export_weights_c.py, neutres par défaut → 0 déclenchement). */
    drift_init(&g_drift, DRIFT_WINDOW_SIZE, DRIFT_FAULT_THRESHOLD,
               DRIFT_DRIFT_THRESHOLD, DRIFT_RATIO);
    g_n_updates = 0;
    g_last_verdict = DRIFT_NORMAL;   /* S3805 */
#endif
    /* S3403 — buffer de streaming instrumenté (ring_buffer générique, taille STREAM_BUF_W) */
    ring_buffer_init(&g_stream_rb, (uint8_t *)g_stream_storage,
                     (int)(PROTO_MAX_N * sizeof(float)), STREAM_BUF_W);
    /* proj initialisée depuis model_weights.h ou générée avec LCG seed fixe :
     * TODO(dorra): mettre les poids de projection en Flash (RODATA) pour économiser 20 Ko SRAM */
    /* Sprint 29 — modèles INT8 firmware */
    hdc_int8_init(&g_hdc_int8);
    tinyol_int8_init(&g_tinyol_int8);
    oto_int8_init(&g_oto_int8);
    acc_init(&g_acc);
    auroc_init(&g_auroc);
    fgt_init(&g_fgt);
    g_current_task_id = 0U;

#if (MAHA_DIM == MAHA_NATIVE_DIM)
    /* Poids Mahalanobis exportés (parité board↔PC) — dim de la condition. */
    for (int i = 0; i < MAHA_DIM; i++) {
        g_detector.mean[i] = MAHA_MEAN_INIT[i];
        for (int j = 0; j < MAHA_DIM; j++) {
            g_detector.precision[i][j] = MAHA_PRECISION_INIT[i][j];
        }
    }
#else
    /* MAHA_DIM ≠ dim native : init neutre (maha_init() a déjà posé identité+0).
     * Poids réels regénérés par condition en S3507. */
#endif

    /* Sprint 34 — Mahalanobis Q15 : init neutre puis poids exportés si fournis (S3407).
     * Sans header généré (MAHA_Q15_WEIGHTS_PROVIDED non défini) → identité, 0 régression. */
    maha_q15_init(&g_maha_q15, MAHA_THRESHOLD_INIT, MAHA_EMA_ALPHA);
#if defined(MAHA_Q15_WEIGHTS_PROVIDED) && (MAHA_Q15_N_FEATURES == MAHA_Q15_NATIVE_DIM)
    g_maha_q15.mu_scale        = MAHA_Q15_MU_SCALE;
    g_maha_q15.mu_zp           = MAHA_Q15_MU_ZP;
    g_maha_q15.sigma_inv_scale = MAHA_Q15_SIGMA_SCALE;
    g_maha_q15.threshold       = MAHA_Q15_THRESHOLD;
    for (int i = 0; i < MAHA_Q15_N_FEATURES; i++) {
        g_maha_q15.mu_q8[i] = MAHA_Q15_MU_Q8[i];
        for (int j = 0; j < MAHA_Q15_N_FEATURES; j++) {
            g_maha_q15.sigma_inv_q15[i][j] = MAHA_Q15_SIGMA_INV[i][j];
        }
    }
#endif

#ifdef MAHA_INT8
    /* Sprint 29 — Mahalanobis INT8 : init neutre puis poids exportés si fournis (S2912).
     * Sans header généré (MAHA_INT8_WEIGHTS_PROVIDED non défini) → identité, 0 régression. */
    maha_int8_init(&g_maha_int8, MAHA_THRESHOLD_INIT, MAHA_EMA_ALPHA);
#if defined(MAHA_INT8_WEIGHTS_PROVIDED) && (MAHA_INT8_N_FEATURES == MAHA_INT8_NATIVE_DIM)
    g_maha_int8.mu_scale        = MAHA_INT8_MU_SCALE;
    g_maha_int8.mu_zp           = MAHA_INT8_MU_ZP;
    g_maha_int8.sigma_inv_scale = MAHA_INT8_SIGMA_SCALE;
    g_maha_int8.sigma_inv_zp    = MAHA_INT8_SIGMA_ZP;
    g_maha_int8.threshold       = MAHA_INT8_THRESHOLD;
    for (int i = 0; i < MAHA_INT8_N_FEATURES; i++) {
        g_maha_int8.mu_q8[i] = MAHA_INT8_MU_Q8[i];
        for (int j = 0; j < MAHA_INT8_N_FEATURES; j++) {
            g_maha_int8.sigma_inv_q8[i][j] = MAHA_INT8_SIGMA_INV[i][j];
        }
    }
#endif
#endif

#if (TINYOL_IN == WEIGHTS_NATIVE_DIM)
    /* Init TinyOL depuis Flash (poids placeholder — remplacer via export_weights_c.py) */
    memcpy(g_tinyol_enc.w_enc1, TINYOL_W_ENC1, sizeof(g_tinyol_enc.w_enc1));
    memcpy(g_tinyol_enc.b_enc1, TINYOL_B_ENC1, sizeof(g_tinyol_enc.b_enc1));
    memcpy(g_tinyol_enc.w_enc2, TINYOL_W_ENC2, sizeof(g_tinyol_enc.w_enc2));
    memcpy(g_tinyol_enc.b_enc2, TINYOL_B_ENC2, sizeof(g_tinyol_enc.b_enc2));
    memcpy(g_tinyol_dec.w_dec1, TINYOL_W_DEC1, sizeof(g_tinyol_dec.w_dec1));
    memcpy(g_tinyol_dec.b_dec1, TINYOL_B_DEC1, sizeof(g_tinyol_dec.b_dec1));
    memcpy(g_tinyol_dec.w_dec2, TINYOL_W_DEC2, sizeof(g_tinyol_dec.w_dec2));
    memcpy(g_tinyol_dec.b_dec2, TINYOL_B_DEC2, sizeof(g_tinyol_dec.b_dec2));
#else
    /* TINYOL_IN ≠ dim native : encodeur/décodeur restent à zéro (.bss).
     * Poids réels regénérés par condition en S3507. */
#endif

#ifndef TEST_MODE
    /* Activer GPIOA pour LED LD2 */
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN;
    GPIOA->MODER &= ~(0x3U << (LED_PIN * 2U));
    GPIOA->MODER |=  (0x1U << (LED_PIN * 2U));   /* Output mode */
    GPIOA->OTYPER &= ~(0x1U << LED_PIN);          /* Push-pull */
#endif

    ENERGY_MARKER_INIT();              /* S3304 — PA8 marqueur phase (no-op si flag absent) */
    energy_marker_phase(PHASE_STARTUP);
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

/* Sprint 27 — Expose uart_send_response_dual pour test taille trame */
void test_pipeline_send_response_dual(uint8_t pred_fault, float conf_fault,
                                       float rul_pred, float f1_macro,
                                       float rmse_rul, float forgetting)
{
    uart_send_response_dual(pred_fault, conf_fault, rul_pred,
                             f1_macro, rmse_rul, forgetting);
}

/* Sprint 30 — Expose uart_send_response_pair pour test taille trame */
void test_pipeline_send_response_pair(uint8_t pred_maha, float score_maha,
                                      uint8_t pred_sup, float conf_sup,
                                      float auroc_maha, float f1_sup)
{
    uart_send_response_pair(pred_maha, score_maha, pred_sup, conf_sup,
                             auroc_maha, f1_sup);
}
#endif

/* ── Boucle principale ─────────────────────────────────────────────────── */

void pipeline_run(void)
{
    /* PROTO_MAX_N=16 > EWC_MC_IN(9) > MAHA_DIM(5) — buffer commun pour tous les modes */
    float raw[PROTO_MAX_N];   /* MEM: 64 B @ FP32 (stack) */

    energy_marker_phase(PHASE_ACQUISITION);   /* S3304 — début réception trame */
    uart_receive_sample(raw);

    /* S3403 — remplissage du buffer de streaming instrumenté (hors chrono d'inférence). */
    ring_buffer_push(&g_stream_rb, raw);

    /* RESET : réinitialise EWC avant une nouvelle expérience.
     * raw[0] encode la nouvelle valeur lambda (> 0 pour la modifier). */
    if (g_recv_flags & PROTO_FLAG_RESET) {
        float new_lambda = raw[0];
        if (new_lambda > 0.0f) g_ewc_head.lambda = new_lambda;
        ewc_head_load_or_init(&g_ewc_head);
        acc_init(&g_acc);
        auroc_init(&g_auroc);
        fgt_init(&g_fgt);
        g_current_task_id = 0U;
        MetricsSnapshot snap_reset = { .accuracy = 0.0f, .auroc = 0.0f, .forgetting = 0.0f };
        uart_send_response_v3(0, 1.0f, 0U, PROTO_STATUS_OK, &snap_reset);
        energy_marker_phase(PHASE_IDLE);   /* S3304 */
        return;
    }

    energy_marker_phase(PHASE_INFERENCE);   /* S3304 — corrélé au DWT (profiling_start adjacent) */
    profiling_start();   /* Démarre le chrono DWT */

    int   pred;
    float confidence;

    /* ── MAHA_Q15_MODE (0xF0) : Mahalanobis seul, sigma_inv int16 Q15 (S3407) ──────
     * DOIT passer AVANT TRIPLE/PAIR/DUAL/MULTICLASS : 0xF0 & 0x70 == 0x70 (DUAL) et
     * 0xF0 & 0x30 == 0x30 (MULTICLASS) matcheraient sinon. Exact-match sur le nibble
     * (masque 0xF0). Réponse V3 (23 B) — aucun nouveau format. mu reste fixe (pas d'EMA :
     * mu quantifié), parité avec MahalanobisDetectorInt8.anomaly_score_q15 (PC). */
    if ((uint8_t)(g_recv_flags & PROTO_PAIR_MODE_MASK) == PROTO_FLAG_MAHA_Q15) {
        normalize_zscore(raw, MAHA_DIM);
        float score   = maha_q15_score(&g_maha_q15, raw);
        int   anomaly = (score > g_maha_q15.threshold) ? 1 : 0;
        led_set(anomaly ? LED_ON : LED_OFF);
        confidence = 1.0f / (1.0f + score);

        if (g_recv_flags & PROTO_FLAG_CONSOLIDATE)
            g_current_task_id = g_recv_task_id;
        auroc_update(&g_auroc, score, (int)g_recv_label);
        acc_update(&g_acc, anomaly, (int)g_recv_label);
        fgt_update(&g_fgt, g_current_task_id, acc_compute(&g_acc));

        profiling_stop();

        MetricsSnapshot snap;
        snap.accuracy   = acc_compute(&g_acc);
        snap.auroc      = auroc_compute(&g_auroc);
        snap.forgetting = fgt_avg_forgetting(&g_fgt);
        uart_send_response_v3((uint8_t)anomaly, confidence,
                              profiling_get_latency_us(), PROTO_STATUS_OK, &snap);
        energy_marker_phase(PHASE_IDLE);   /* S3304 */
        return;
    }

    /* ── TRIPLE_MODE (0xD0/0xE0) : PAIR (Maha + supervisé) + méta-modèle ───────
     * Sprint 31 (S3106). Les 2 sorties de base alimentent meta_head → verdict final.
     * DOIT passer AVANT PAIR/DUAL/MULTICLASS (exact-match sur le nibble de mode).
     * Feature méta = [p_maha, p_sup, disagreement, conf_sup] (ordre DEFAULT_FEATURES,
     * cf. src/ensemble/meta_learner.py), toutes ∈ [0, 1] → pas de scaler embarqué.
     *   p_maha = sigmoid(score - seuil)  (parité ModelPair._maha_proba)
     *   p_sup  : EWC → softmax classe 1 ; HDC → label (HDC sans predict_proba côté PC) */
#if META_N_FEATURES != 4
#error "TRIPLE_MODE attend META_N_FEATURES == 4 ([p_maha, p_sup, disagreement, conf_sup])"
#endif
    uint8_t triple_mode = (uint8_t)(g_recv_flags & PROTO_TRIPLE_MODE_MASK);
    if (triple_mode == PROTO_FLAG_TRIPLE_MAHA_EWC ||
        triple_mode == PROTO_FLAG_TRIPLE_MAHA_HDC) {

        /* (1) Détecteur Mahalanobis sur copie z-scorée */
        float maha_x[MAHA_DIM];   /* MEM: 20 B @ FP32 (stack) */
        memcpy(maha_x, raw, sizeof(maha_x));
        normalize_zscore(maha_x, MAHA_DIM);
        float maha_score_v = maha_score(&g_detector, maha_x);
        int   pred_maha = (maha_score_v > g_detector.threshold) ? 1 : 0;
        float p_maha = 1.0f / (1.0f + expf(-(maha_score_v - g_detector.threshold)));

        /* (2) Modèle supervisé sur raw brut → pred_sup + p_sup ∈ [0, 1] */
        int   pred_sup = 0;
        float p_sup = 0.0f;

        if (triple_mode == PROTO_FLAG_TRIPLE_MAHA_EWC) {
            float logits[EWC_OUT];   /* MEM: 8 B @ FP32 (stack) */
            ewc_forward(&g_ewc_head, raw, logits);
            pred_sup = (logits[1] > logits[0]) ? 1 : 0;
            float e0 = expf(logits[0]);
            float e1 = expf(logits[1]);
            p_sup = e1 / (e0 + e1);            /* proba classe 1 = _supervised_proba (EWC) */
            if (g_recv_flags & PROTO_FLAG_UPDATE)
                ewc_sgd_step(&g_ewc_head, raw, (int)g_recv_label);
            if (g_recv_flags & PROTO_FLAG_CONSOLIDATE)
                ewc_consolidate(&g_ewc_head, EWC_FISHER_DECAY);
        } else { /* PROTO_FLAG_TRIPLE_MAHA_HDC */
            float hv[HDC_DIM];   /* MEM: 4 Ko @ FP32 (stack) */
            hdc_encode(&g_hdc, raw, hv);
            pred_sup = hdc_predict(&g_hdc, hv);
            p_sup = (float)pred_sup;           /* HDC sans predict_proba → label sert de proba (PC) */
            if (g_recv_flags & PROTO_FLAG_UPDATE)
                hdc_update_with_sample(&g_hdc, raw, hv, (int)g_recv_label);
            if (g_recv_flags & PROTO_FLAG_CONSOLIDATE)
                hdc_binarize(&g_hdc);
        }

        /* (3) Méta-modèle : arbitrage des 2 sorties → verdict final */
        float feats[META_N_FEATURES];
        feats[0] = p_maha;
        feats[1] = p_sup;
        feats[2] = (pred_maha != pred_sup) ? 1.0f : 0.0f;   /* disagreement */
        feats[3] = fabsf(p_sup - 0.5f) * 2.0f;              /* conf_sup */
        float prob_meta = meta_forward(&g_meta, feats);
        int   pred_meta = (prob_meta > 0.5f) ? 1 : 0;

        /* (4) Mise à jour détecteur + LED pilotée par le verdict méta */
        if ((g_recv_flags & PROTO_FLAG_UPDATE) && !pred_maha)
            maha_update(&g_detector, maha_x);
        if (g_recv_flags & PROTO_FLAG_CONSOLIDATE)
            g_current_task_id = g_recv_task_id;
        led_set(pred_meta ? LED_ON : LED_OFF);

        /* (5) Métriques en ligne : AUROC (Maha) + F1 (verdict méta) */
        auroc_update(&g_auroc, maha_score_v, (int)g_recv_label);
        online_f1_update(&g_f1, pred_meta, (int)g_recv_label);

        profiling_stop();

        uart_send_response_triple((uint8_t)pred_maha, maha_score_v,
                                  (uint8_t)pred_sup, p_sup,
                                  auroc_compute(&g_auroc),
                                  online_f1_get(&g_f1),
                                  (uint8_t)pred_meta, prob_meta);
        energy_marker_phase(PHASE_IDLE);   /* S3304 */
        return;
    }

    /* ── PAIR_MODE (0x90/0xA0/0xB0) : Mahalanobis + supervisé {EWC,HDC,TinyOL} ──
     * Généralisation Sprint 30 du DUAL_MODE. DOIT passer AVANT DUAL/MULTICLASS :
     * les valeurs de nibble 0x9_/0xA_/0xB_ sont libres, mais on respecte la règle
     * « exact-match (sur le nibble de mode) avant subset ».
     *   Mahalanobis : raw[0..MAHA_DIM-1] z-scoré (copie — ne pas clobber raw)
     *   Supervisé   : raw brut (parité avec les chemins single-mode)            */
    uint8_t pair_mode = (uint8_t)(g_recv_flags & PROTO_PAIR_MODE_MASK);
    if (pair_mode == PROTO_FLAG_PAIR_MAHA_EWC ||
        pair_mode == PROTO_FLAG_PAIR_MAHA_HDC ||
        pair_mode == PROTO_FLAG_PAIR_MAHA_TINYOL) {

        /* (1) Détecteur Mahalanobis sur copie z-scorée */
        float maha_x[MAHA_DIM];   /* MEM: 20 B @ FP32 (stack) */
        memcpy(maha_x, raw, sizeof(maha_x));
        normalize_zscore(maha_x, MAHA_DIM);
        float maha_score_v = maha_score(&g_detector, maha_x);
        int   pred_maha = (maha_score_v > g_detector.threshold) ? 1 : 0;

        /* (2) Modèle supervisé sur raw brut, selon le sous-mode */
        int   pred_sup = 0;
        float conf_sup = 0.0f;

        if (pair_mode == PROTO_FLAG_PAIR_MAHA_EWC) {
            float logits[EWC_OUT];   /* MEM: 8 B @ FP32 (stack) */
            ewc_forward(&g_ewc_head, raw, logits);
            pred_sup = (logits[1] > logits[0]) ? 1 : 0;
            float e0 = expf(logits[0]);
            float e1 = expf(logits[1]);
            conf_sup = e1 / (e0 + e1);
            if (g_recv_flags & PROTO_FLAG_UPDATE)
                ewc_sgd_step(&g_ewc_head, raw, (int)g_recv_label);
            if (g_recv_flags & PROTO_FLAG_CONSOLIDATE)
                ewc_consolidate(&g_ewc_head, EWC_FISHER_DECAY);
        } else if (pair_mode == PROTO_FLAG_PAIR_MAHA_HDC) {
            float hv[HDC_DIM];   /* MEM: 4 Ko @ FP32 (stack) — cf. FIXME(gap2) HDC single */
            hdc_encode(&g_hdc, raw, hv);
            pred_sup = hdc_predict(&g_hdc, hv);
            float score = 0.0f;
            for (int i = 0; i < HDC_DIM; i++) score += g_hdc.am[pred_sup][i] * hv[i];
            conf_sup = (score / (float)HDC_DIM + 1.0f) / 2.0f;
            if (g_recv_flags & PROTO_FLAG_UPDATE)
                hdc_update_with_sample(&g_hdc, raw, hv, (int)g_recv_label);
            if (g_recv_flags & PROTO_FLAG_CONSOLIDATE)
                hdc_binarize(&g_hdc);
        } else { /* PROTO_FLAG_PAIR_MAHA_TINYOL */
            float emb[16];   /* MEM: 64 B (stack) */
            float recon[EWC_IN];
            tinyol_encode(&g_tinyol_enc, raw, emb);
            tinyol_decode(&g_tinyol_dec, emb, recon);
            float mse = tinyol_reconstruction_error(raw, recon, EWC_IN);
            pred_sup = (mse > TINYOL_THRESHOLD) ? 1 : 0;
            conf_sup = 1.0f / (1.0f + mse);
            /* update OtO non disponible en C bare-metal (no-op, cf. chemin single) */
        }

        /* (3) Mise à jour du détecteur + LED pilotée par l'anomalie */
        if ((g_recv_flags & PROTO_FLAG_UPDATE) && !pred_maha)
            maha_update(&g_detector, maha_x);
        if (g_recv_flags & PROTO_FLAG_CONSOLIDATE)
            g_current_task_id = g_recv_task_id;
        led_set(pred_maha ? LED_ON : LED_OFF);

        /* (4) Métriques en ligne : AUROC (Maha, score) + F1 (supervisé) */
        auroc_update(&g_auroc, maha_score_v, (int)g_recv_label);
        online_f1_update(&g_f1, pred_sup, (int)g_recv_label);

        profiling_stop();

        uart_send_response_pair((uint8_t)pred_maha, maha_score_v,
                                 (uint8_t)pred_sup, conf_sup,
                                 auroc_compute(&g_auroc),
                                 online_f1_get(&g_f1));
        energy_marker_phase(PHASE_IDLE);   /* S3304 */
        return;
    }

    /* ── DUAL_MODE (0x70) : EWC_REG (RUL) + EWC_MC (faute) ────────────────────
     * DOIT ÊTRE EN PREMIER — 0x70 & 0x30 == 0x30 matcherait MULTICLASS sinon
     * Encodage labels :
     *   g_recv_task_id = fault_label ∈ [0, EWC_MC_N_CLASSES-1]
     *   g_recv_label   = rul_u8 = round(RUL / 300 × 255) → re-normalisé en [0,1]
     * Features :
     *   raw[0..4] → g_ewc_reg  (EWC_REG_IN = 5)
     *   raw[0..8] → g_ewc_mc   (EWC_MC_IN  = 9) */
    if ((g_recv_flags & PROTO_FLAG_DUAL_MODE) == PROTO_FLAG_DUAL_MODE) {
        uint8_t fault_label    = g_recv_task_id;
        float   rul_label_norm = (float)g_recv_label / 255.0f;

        float rul_pred = ewc_reg_predict(&g_ewc_reg, raw);

        float logits[EWC_MC_N_CLASSES];   /* MEM: N×4 B stack */
        ewc_mc_forward(&g_ewc_mc, raw, logits);
        int fault_pred = ewc_mc_predict(&g_ewc_mc, raw);

        float max_l = logits[0];
        for (int j = 1; j < EWC_MC_N_CLASSES; j++)
            if (logits[j] > max_l) max_l = logits[j];
        float sum_exp = 0.0f;
        for (int j = 0; j < EWC_MC_N_CLASSES; j++)
            sum_exp += expf(logits[j] - max_l);
        float conf_fault = expf(logits[fault_pred] - max_l) / sum_exp;

        if (g_recv_flags & PROTO_FLAG_UPDATE) {
            ewc_reg_sgd_step(&g_ewc_reg, raw, rul_label_norm);
            ewc_mc_sgd_step(&g_ewc_mc, raw, (int)fault_label);
        }
        if (g_recv_flags & PROTO_FLAG_CONSOLIDATE) {
            ewc_reg_consolidate(&g_ewc_reg, EWC_REG_FISHER_DECAY);
            ewc_mc_consolidate(&g_ewc_mc, EWC_MC_FISHER_DECAY);
            g_current_task_id++;
        }

        online_rmse_update(&g_rmse, rul_pred, rul_label_norm);
        online_f1_update(&g_f1, fault_pred, (int)fault_label);

        profiling_stop();

        uart_send_response_dual(
            (uint8_t)fault_pred, conf_fault, rul_pred,
            online_f1_get(&g_f1),
            online_rmse_get(&g_rmse),
            fgt_avg_forgetting(&g_fgt)
        );
        energy_marker_phase(PHASE_IDLE);   /* S3304 */
        return;
    }

    if ((g_recv_flags & PROTO_FLAG_MULTICLASS_MODE) == PROTO_FLAG_MULTICLASS_MODE) {
        /* ── Chemin EWC Multi-class (EWC_MODE|HDC_MODE = 0x30) ─────────────── */
        float logits[EWC_MC_N_CLASSES];   /* MEM: N×4 B stack */
        ewc_mc_forward(&g_ewc_mc, raw, logits);
        pred = ewc_mc_predict(&g_ewc_mc, raw);

        /* Confiance = softmax[pred] numériquement stable */
        float max_l = logits[0];
        for (int j = 1; j < EWC_MC_N_CLASSES; j++)
            if (logits[j] > max_l) max_l = logits[j];
        float sum_exp = 0.0f;
        for (int j = 0; j < EWC_MC_N_CLASSES; j++) sum_exp += expf(logits[j] - max_l);
        confidence = expf(logits[pred] - max_l) / sum_exp;

        if (g_recv_flags & PROTO_FLAG_UPDATE)
            ewc_mc_sgd_step(&g_ewc_mc, raw, (int)g_recv_label);
        if (g_recv_flags & PROTO_FLAG_CONSOLIDATE) {
            ewc_mc_consolidate(&g_ewc_mc, EWC_MC_FISHER_DECAY);
            g_current_task_id = g_recv_task_id;
        }
        online_f1_update(&g_f1, pred, (int)g_recv_label);

    } else if ((g_recv_flags & PROTO_FLAG_RUL_MODE) == PROTO_FLAG_RUL_MODE) {
        /* ── Chemin EWC Régression RUL (EWC_MODE|INT8_MODE = 0x50) ─────────── */
        float rul_pred = ewc_reg_predict(&g_ewc_reg, raw);
        pred       = 0;
        confidence = rul_pred;   /* champ conf transporte le RUL prédit */

        /* label_u8 ∈ [0,255] encodé par simulate_rul_board.py (RUL/RUL_CAP × 255)
         * → re-normaliser en [0,1] pour correspondre à la target d'entraînement */
        float rul_label_norm = (float)g_recv_label / 255.0f;
        if (g_recv_flags & PROTO_FLAG_UPDATE)
            ewc_reg_sgd_step(&g_ewc_reg, raw, rul_label_norm);
        if (g_recv_flags & PROTO_FLAG_CONSOLIDATE) {
            ewc_reg_consolidate(&g_ewc_reg, EWC_REG_FISHER_DECAY);
            g_current_task_id = g_recv_task_id;
        }
        online_rmse_update(&g_rmse, rul_pred, rul_label_norm);

    } else if ((g_recv_flags & PROTO_FLAG_HDC_INT8) == PROTO_FLAG_HDC_INT8) {
        /* ── Chemin HDC INT8 (HDC_MODE|INT8_MODE = 0x60) ──────────────────────
         * DOIT passer avant le check INT8_MODE (0x40) : 0x60 & 0x40 == 0x40. */
        hdc_int8_encode(&g_hdc_int8, raw, g_hv_int8);
        pred = hdc_int8_predict(&g_hdc_int8, g_hv_int8);

        /* Confiance : dot(am[pred], hv) normalisé par (HDC_I_D × max|am|) → [0, 1] */
        int32_t score = 0;
        for (int i = 0; i < HDC_I_D; i++)
            score += (int32_t)g_hdc_int8.am[pred][i] * (int32_t)g_hv_int8[i];
        confidence = (float)score / (float)HDC_I_D;
        confidence = (confidence + 1.0f) / 2.0f;   /* recentrage approximatif [0,1] */

        if (g_recv_flags & PROTO_FLAG_UPDATE)
            hdc_int8_update(&g_hdc_int8, g_hv_int8, (int)g_recv_label);
        if (g_recv_flags & PROTO_FLAG_CONSOLIDATE)
            g_current_task_id = g_recv_task_id;   /* HDC additif : pas de binarize */
        auroc_update(&g_auroc, confidence, (int)g_recv_label);

    } else if ((g_recv_flags & PROTO_FLAG_TINYOL_INT8) == PROTO_FLAG_TINYOL_INT8) {
        /* ── Chemin TinyOL INT8 (TINYOL_MODE|INT8_MODE = 0xC0) ────────────────
         * DOIT passer avant le check INT8_MODE (0x40) : 0xC0 & 0x40 == 0x40. */
        uint8_t emb_u8[TINYOL_EMB];   /* MEM: 16 B @ UINT8 stack */
        tinyol_int8_encode(&g_tinyol_int8, raw, emb_u8);
        pred       = oto_int8_predict(&g_oto_int8, emb_u8);
        confidence = g_oto_int8.last_prob;

        if (g_recv_flags & PROTO_FLAG_UPDATE)
            oto_int8_update(&g_oto_int8, emb_u8, (int)g_recv_label);
        if (g_recv_flags & PROTO_FLAG_CONSOLIDATE)
            g_current_task_id = g_recv_task_id;
        auroc_update(&g_auroc, confidence, (int)g_recv_label);

    } else if (g_recv_flags & PROTO_FLAG_EWC_MODE) {
        /* ── Chemin EWC : forward softmax + update SGD + consolidation ─────── */
        float logits[EWC_OUT];   /* MEM: 8 B @ FP32 (stack) */
        ewc_forward(&g_ewc_head, raw, logits);
        pred = (logits[1] > logits[0]) ? 1 : 0;

        float e0 = expf(logits[0]);
        float e1 = expf(logits[1]);
        confidence = e1 / (e0 + e1);   /* prob(faulty=1) pour AUROC */

#ifdef EWC_AUTO_UPDATE
        /* ── Gate autonome (S3803) : remplace le déclencheur humain PROTO_FLAG_UPDATE.
         * Le score Mahalanobis nourrit le détecteur de dérive ; le verdict décide à
         * bord QUAND et AVEC QUEL label mettre à jour. Protocole UART inchangé. */
        {
            float s = maha_score(&g_detector, raw);
            DriftVerdict v = drift_update(&g_drift, s);
            g_last_verdict = v;   /* S3805 : exposé dans le snapshot (réinterprétation, infra) */
#ifdef GATE_PSEUDO_LABEL
            /* P3 — 100 % autonome (pseudo-label par verdict). */
            if (v == DRIFT_FAULT) {
                ewc_sgd_step(&g_ewc_head, raw, 1);   /* pseudo-label faulty */
                g_n_updates++;
            } else if (v == DRIFT_DRIFT) {
                maha_update(&g_detector, raw);        /* adapte le normal (pas de SGD faute) */
            }
#else
            /* P2 — active learning : vrai label sur déclenchement du gate. */
            if (v != DRIFT_NORMAL) {
                ewc_sgd_step(&g_ewc_head, raw, (int)g_recv_label);
                g_n_updates++;
            }
#endif
        }
#else
        if (g_recv_flags & PROTO_FLAG_UPDATE) {
            ewc_sgd_step(&g_ewc_head, raw, (int)g_recv_label);
        }
#endif
        if (g_recv_flags & PROTO_FLAG_CONSOLIDATE) {
            ewc_consolidate(&g_ewc_head, EWC_FISHER_DECAY);
            g_current_task_id = g_recv_task_id;
        }
        auroc_update(&g_auroc, confidence, (int)g_recv_label);
    } else if (g_recv_flags & PROTO_FLAG_INT8_MODE) {
#ifdef EWC_INT8_V2
        /* ── Chemin EWC INT8 v2 (S3915) : forward inférence, déquant→FP32 ──────────
         * Sélectionné par -DEWC_INT8_V2 (le nibble 0x40 route vers le v2 au lieu du v1).
         * Le kernel v2 quantifie les activations à bord (scales calibrés) ⇒ on lui passe
         * l'entrée FP32 brute, pas x_q7. */
        float logits[EWC_OUT];   /* MEM: 8 B @ FP32 (stack) */
        ewc_int8_v2_forward(&g_ewc_int8_v2, raw, logits);
        pred = (logits[1] > logits[0]) ? 1 : 0;

        float e0 = expf(logits[0]);
        float e1 = expf(logits[1]);
        confidence = e1 / (e0 + e1);

        /* Update online (S4002) : SGD sur la tête FP32 maître puis requantification v2.
         * g_ewc_head est la source ; g_ewc_int8_v2 en est la vue quantifiée (act_max figé).
         * Le coût de requantification par échantillon est inclus dans la latence DWT (honnête). */
        if (g_recv_flags & PROTO_FLAG_UPDATE) {
            ewc_sgd_step(&g_ewc_head, raw, (int)g_recv_label);
            ewc_int8_v2_from_fp32_calib(&g_ewc_int8_v2, &g_ewc_head, g_v2_act_max);
        }
        if (g_recv_flags & PROTO_FLAG_CONSOLIDATE) {
            ewc_consolidate(&g_ewc_head, EWC_FISHER_DECAY);
            ewc_int8_v2_from_fp32_calib(&g_ewc_int8_v2, &g_ewc_head, g_v2_act_max);
            g_current_task_id = g_recv_task_id;
        }
        auroc_update(&g_auroc, confidence, (int)g_recv_label);
#else
        /* ── Chemin EWC INT8 v1 : forward Q7 + update Q7 ──────────────────────────── */
        int8_t x_q7[EWC_IN];   /* MEM: EWC_IN B = 5 B (stack) */
        for (int i = 0; i < EWC_IN; i++) {
            x_q7[i] = float_to_q7(raw[i]);
        }

        float logits[EWC_OUT];   /* MEM: 8 B @ FP32 (stack) */
        ewc_int8_forward(&g_ewc_int8, x_q7, logits);
        pred = (logits[1] > logits[0]) ? 1 : 0;

        float e0 = expf(logits[0]);
        float e1 = expf(logits[1]);
        confidence = e1 / (e0 + e1);

        if (g_recv_flags & PROTO_FLAG_UPDATE) {
            ewc_int8_update(&g_ewc_int8, x_q7, g_recv_label,
                            0.01f, /* lr = EWC_LR */
                            1      /* fisher_ema = true */);
        }
        if (g_recv_flags & PROTO_FLAG_CONSOLIDATE) {
            ewc_int8_consolidate(&g_ewc_int8);
            g_current_task_id = g_recv_task_id;
        }
        auroc_update(&g_auroc, confidence, (int)g_recv_label);
#endif
    } else if (g_recv_flags & PROTO_FLAG_HDC_MODE) {
        /* ── Chemin HDC : encode → predict → update si UPDATE → binarize si CONSOLIDATE ── */
        float hv[HDC_DIM];   /* MEM: 4 Ko @ FP32 (stack — vérifier _Min_Stack_Size dans ld) */
        /* FIXME(gap2): hv[HDC_DIM] = 4 Ko sur pile — valider suffisance pile MCU */
        hdc_encode(&g_hdc, raw, hv);
        pred = hdc_predict(&g_hdc, hv);

        /* Confiance : dot(am[pred], hv) normalisé par HDC_DIM → [0, 1] */
        float score = 0.0f;
        for (int i = 0; i < HDC_DIM; i++) score += g_hdc.am[pred][i] * hv[i];
        confidence = (score / (float)HDC_DIM + 1.0f) / 2.0f;

        if (g_recv_flags & PROTO_FLAG_UPDATE) {
            hdc_update_with_sample(&g_hdc, raw, hv, (int)g_recv_label);
        }
        if (g_recv_flags & PROTO_FLAG_CONSOLIDATE) {
            hdc_binarize(&g_hdc);
            g_current_task_id = g_recv_task_id;
        }
        auroc_update(&g_auroc, confidence, (int)g_recv_label);
    } else if (g_recv_flags & PROTO_FLAG_TINYOL_MODE) {
        /* ── Chemin TinyOL autoencoder (anomaly via reconstruction) ─────────── */
        /* MEM: emb[16] = 64 B + recon[5] = 20 B (stack) */
        float emb[16];
        float recon[EWC_IN];
        tinyol_encode(&g_tinyol_enc, raw, emb);
        tinyol_decode(&g_tinyol_dec, emb, recon);
        float mse = tinyol_reconstruction_error(raw, recon, EWC_IN);
        /* Seuil : depuis model_weights.h (calibré P95×1.5 via export_weights_tinyol.py) */
        pred       = (mse > TINYOL_THRESHOLD) ? 1 : 0;
        confidence = 1.0f / (1.0f + mse);
        led_set(pred ? LED_ON : LED_OFF);

        if (g_recv_flags & PROTO_FLAG_UPDATE) {
            /* Mise à jour OtO : ajuste les poids si pred != label */
            tinyol_encode(&g_tinyol_enc, raw, emb);  /* re-encode pour update */
            (void)emb;  /* update via gradient backward non disponible en C bare-metal — no-op */
        }
        if (g_recv_flags & PROTO_FLAG_CONSOLIDATE) {
            g_current_task_id = g_recv_task_id;
        }
        auroc_update(&g_auroc, mse, (int)g_recv_label);
    } else {
        /* ── Chemin Mahalanobis (comportement historique) ───────────────────── */
        normalize_zscore(raw, MAHA_DIM);
#ifdef MAHA_INT8
        /* Sprint 29 (S2912) : variante INT8 sélectionnée à la compilation (nibble saturé).
         * Fit offline → pas de maha_update (mu/sigma_inv quantifiés figés). Parité PC. */
        float score = maha_int8_score(&g_maha_int8, raw);
        int   anomaly = (score > g_maha_int8.threshold) ? 1 : 0;
        pred       = anomaly;
        confidence = 1.0f / (1.0f + score);
        led_set(anomaly ? LED_ON : LED_OFF);
#else
        float score = maha_score(&g_detector, raw);
        int   anomaly = (score > g_detector.threshold) ? 1 : 0;
        pred       = anomaly;
        confidence = 1.0f / (1.0f + score);
        led_set(anomaly ? LED_ON : LED_OFF);

        if ((g_recv_flags & PROTO_FLAG_UPDATE) && !anomaly) {
            maha_update(&g_detector, raw);
        }
#endif
        if (g_recv_flags & PROTO_FLAG_CONSOLIDATE) {
            ewc_consolidate(&g_ewc_head, EWC_FISHER_DECAY);
            g_current_task_id = g_recv_task_id;
        }
        auroc_update(&g_auroc, score, (int)g_recv_label);
    }

    profiling_stop();    /* Arrête le chrono, calcule throughput */

    /* acc_update / fgt_update uniquement pour les modes classification */
    if ((g_recv_flags & PROTO_FLAG_RUL_MODE) != PROTO_FLAG_RUL_MODE) {
        acc_update(&g_acc, pred, (int)g_recv_label);
        fgt_update(&g_fgt, g_current_task_id, acc_compute(&g_acc));
    }

    MetricsSnapshot snap;
    if ((g_recv_flags & PROTO_FLAG_MULTICLASS_MODE) == PROTO_FLAG_MULTICLASS_MODE) {
        snap.accuracy   = online_f1_get(&g_f1);
        snap.auroc      = 0.0f;
        snap.forgetting = fgt_avg_forgetting(&g_fgt);
    } else if ((g_recv_flags & PROTO_FLAG_RUL_MODE) == PROTO_FLAG_RUL_MODE) {
        snap.accuracy   = 0.0f;
        snap.auroc      = online_rmse_get(&g_rmse);   /* champ auroc transporte RMSE */
        snap.forgetting = 0.0f;
    } else {
        snap.accuracy   = acc_compute(&g_acc);
        snap.auroc      = auroc_compute(&g_auroc);
        snap.forgetting = fgt_avg_forgetting(&g_fgt);
    }

#ifdef EWC_AUTO_UPDATE
    /* S3805 — réinterprétation du snapshot SOUS LE GATE UNIQUEMENT (wire format V3
     * inchangé : sensor_stream lit toujours [acc][auroc][forgetting]). Le gate doit
     * renvoyer le verdict par échantillon (parité S3806) et le compteur de MAJ réel
     * (économie S3805) ; aucun slot dédié dans la réponse → on réutilise auroc/forgetting.
     * accuracy reste l'accuracy online. Build par défaut strictement inchangé. */
    if (g_recv_flags & PROTO_FLAG_EWC_MODE) {
        snap.auroc      = (float)g_last_verdict;   /* 0=NORMAL,1=FAULT,2=DRIFT */
        snap.forgetting = (float)g_n_updates;      /* compteur cumulé de SGD déclenchés */
    }
#endif

    uart_send_response_v3((uint8_t)pred, confidence,
                           profiling_get_latency_us(), PROTO_STATUS_OK, &snap);

#ifdef DEBUG_PRINTF
    char dbg[80];
    snprintf(dbg, sizeof(dbg), "score=%.4f pred=%d lat=%lu us\r\n",
             (double)confidence, pred, (unsigned long)profiling_get_latency_us());
    for (int i = 0; dbg[i]; i++) uart_send_byte((uint8_t)dbg[i]);
#endif

    energy_marker_phase(PHASE_IDLE);   /* S3304 — retour en attente UART */
}
