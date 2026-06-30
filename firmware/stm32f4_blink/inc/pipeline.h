#pragma once
#include <stdint.h>
#include "mahalanobis.h"
#include "mahalanobis_q15.h"
#include "metrics.h"
#include "ewc_head.h"
#include "ewc_head_regression.h"
#include "ewc_head_multiclass.h"
#include "tinyol.h"
#include "hdc.h"
#include "ewc_head_int8.h"
#include "hdc_int8.h"
#include "tinyol_int8.h"
#include "meta_head.h"

/* LED LD2 = PA5 sur NUCLEO-F439ZI */
#define LED_PIN 5U
#define LED_ON  1
#define LED_OFF 0

/* Protocole UART v2 / v3 */
#define PROTO_VERSION_V2         0x02U
#define PROTO_VERSION_V3         0x03U
#define PROTO_FLAG_UPDATE        0x01U
#define PROTO_FLAG_PROFILING     0x02U
#define PROTO_FLAG_CONSOLIDATE   0x04U   /* frontière de tâche → ewc_consolidate() */
#define PROTO_FLAG_RESET         0x08U   /* réinitialise poids EWC → ewc_init() + reset métriques */
#define PROTO_FLAG_EWC_MODE      0x10U   /* utilise EWC head pour inférence (au lieu de Mahalanobis) */
#define PROTO_FLAG_HDC_MODE      0x20U   /* utilise HDCClassifier pour inférence */
#define PROTO_FLAG_INT8_MODE     0x40U   /* utilise EWCHeadInt8 (bit 6) */
#define PROTO_FLAG_TINYOL_MODE   0x80U   /* utilise TinyOL autoencoder (bit 7) */
/* Sprint 26 — combinaisons libres (bits 7 déjà pris par TINYOL) */
#define PROTO_FLAG_RUL_MODE        (PROTO_FLAG_EWC_MODE | PROTO_FLAG_INT8_MODE)  /* 0x50 */
#define PROTO_FLAG_MULTICLASS_MODE (PROTO_FLAG_EWC_MODE | PROTO_FLAG_HDC_MODE)   /* 0x30 */
/* Sprint 27 — DUAL_MODE : EWC_REG (RUL) + EWC_MC (faute) pipeline simultané
 * Valeur : EWC_MODE(0x10) | HDC_MODE(0x20) | INT8_MODE(0x40) = 0x70
 * ATTENTION : 0x70 & 0x30 == 0x30 — le bloc DUAL_MODE doit passer AVANT
 *             le check MULTICLASS_MODE dans pipeline_run() */
#define PROTO_FLAG_DUAL_MODE       (PROTO_FLAG_EWC_MODE | PROTO_FLAG_HDC_MODE | PROTO_FLAG_INT8_MODE)  /* 0x70 */
/* Sprint 29 — modes INT8 firmware (HDC / TinyOL).
 * Valeurs retenues : HDC_INT8=0x60 (HDC|INT8), TINYOL_INT8=0xC0 (TINYOL|INT8).
 * NB : on n'utilise PAS 0x22/0x81 du doc S2900 (collision PROFILING 0x02 / UPDATE 0x01).
 * ATTENTION : 0x60 & 0x40 == 0x40 et 0xC0 & 0x40 == 0x40 — les deux blocs doivent passer
 *             AVANT le check INT8_MODE (EWC INT8) dans pipeline_run(). */
#define PROTO_FLAG_HDC_INT8        (PROTO_FLAG_HDC_MODE    | PROTO_FLAG_INT8_MODE)  /* 0x60 */
#define PROTO_FLAG_TINYOL_INT8     (PROTO_FLAG_TINYOL_MODE | PROTO_FLAG_INT8_MODE)  /* 0xC0 */
#define RESPONSE_DUAL_SIZE          25U  /* [pred_fault:u8][conf_fault:f32][rul_pred:f32]
                                          * [lat_us:u32][f1_macro:f32][rmse_rul:f32][forgetting:f32] */
/* Sprint 30 — PAIR_MODE : généralisation DUAL_MODE aux paires Mahalanobis + supervisé.
 * Le byte FLAGS est saturé en bits individuels (0x01..0x80), mais le nibble HAUT
 * (sélecteur de mode) a des valeurs libres : 0x90/0xA0/0xB0/0xD0/0xE0/0xF0 (cf. S2600).
 * On y place 3 modes paire (valeurs de nibble uniques → aucune collision avec
 * EWC 0x10 / HDC 0x20 / MULTICLASS 0x30 / INT8 0x40 / RUL 0x50 / HDC_INT8 0x60 /
 * DUAL 0x70 / TINYOL 0x80 / TINYOL_INT8 0xC0). Tester via (FLAGS & PROTO_PAIR_MODE_MASK). */
#define PROTO_FLAG_PAIR_MAHA_EWC    0x90U  /* Mahalanobis + EWC binaire  */
#define PROTO_FLAG_PAIR_MAHA_HDC    0xA0U  /* Mahalanobis + HDC          */
#define PROTO_FLAG_PAIR_MAHA_TINYOL 0xB0U  /* Mahalanobis + TinyOL recon */
#define PROTO_PAIR_MODE_MASK        0xF0U  /* isole le nibble de mode    */
#define RESPONSE_PAIR_SIZE          22U  /* [pred_maha:u8][score_maha:f32][pred_sup:u8]
                                          * [conf_sup:f32][lat_us:u32][auroc_maha:f32][f1_sup:f32] */
/* Sprint 31 — TRIPLE_MODE : PAIR (Maha + supervisé) + méta-modèle de stacking.
 * Le méta arbitre les 2 sorties de base → verdict binaire final (meta_head.c).
 * Nibbles de mode libres après PAIR (0x9_/0xA_/0xB_) et TINYOL_INT8 (0xC0) :
 * on prend 0xD0 (maha+EWC) et 0xE0 (maha+HDC). Aucune collision (cf. masque 0xF0). */
#define PROTO_FLAG_TRIPLE_MAHA_EWC  0xD0U  /* Mahalanobis + EWC + méta */
#define PROTO_FLAG_TRIPLE_MAHA_HDC  0xE0U  /* Mahalanobis + HDC + méta */
#define PROTO_TRIPLE_MODE_MASK      0xF0U  /* même masque de nibble que PAIR */
#define RESPONSE_TRIPLE_SIZE        27U  /* PAIR 22 B + [pred_meta:u8][prob_meta:f32] */

/* Sprint 34 — MAHA_Q15_MODE : Mahalanobis seul avec sigma_inv int16 Q15 (fallback grande
 * dynamique, S3407). Le nibble 0xF0 est le SEUL libre : 0x10–0xE0 sont tous pris (EWC 0x10 …
 * TINYOL_INT8 0xC0, TRIPLE 0xD0/0xE0). Dispatch via (FLAGS & PROTO_PAIR_MODE_MASK) == 0xF0,
 * AVANT la chaîne de bits (0xF0 & 0x30 == 0x30 matcherait MULTICLASS sinon). Réponse V3 (23 B). */
#define PROTO_FLAG_MAHA_Q15         0xF0U  /* Mahalanobis sigma_inv Q15 (réutilise masque 0xF0) */

#define PROTO_STATUS_OK          0x00U
#define PROTO_STATUS_CRC_ERR     0x01U
#define PROTO_STATUS_OOB         0x02U
#define PROTO_STATUS_UPDATE_DONE 0x04U

/* Globals modèles (allocation statique, pas de malloc) */
extern MahalanobisDetector g_detector;
extern EWCHead             g_ewc_head;
extern EWCHeadReg          g_ewc_reg;   /* MEM: ~8.9 Ko @ FP32 en .bss */
extern EWCHeadMC           g_ewc_mc;    /* MEM: ~14 Ko @ FP32 en .bss */
extern TinyOLEncoder       g_tinyol_enc;
extern TinyOLDecoder       g_tinyol_dec;
extern HDCClassifier       g_hdc;
extern EWCHeadInt8         g_ewc_int8;
extern HDCInt8             g_hdc_int8;     /* MEM: ~34.8 Ko @ INT8/INT16 en .bss */
extern TinyOLEncoderInt8   g_tinyol_int8;
extern OtOHeadInt8         g_oto_int8;
extern MetaHead            g_meta;       /* méta-modèle stacking (TRIPLE_MODE) */
extern MahalanobisQ15      g_maha_q15;   /* Mahalanobis sigma_inv Q15 (MAHA_Q15_MODE, S3407) */

/* Stubs HAL — implémentés dans pipeline.c pour le MVP */
void uart_receive_sample(float *buf);
void led_set(int state);

/* Pipeline principal */
void pipeline_init(void);
void pipeline_run(void);
void pipeline_set_task(uint8_t task_id);  /* change g_current_task_id */
