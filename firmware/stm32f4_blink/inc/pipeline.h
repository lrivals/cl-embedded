#pragma once
#include <stdint.h>
#include "mahalanobis.h"
#include "metrics.h"
#include "ewc_head.h"
#include "tinyol.h"

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
#define PROTO_STATUS_OK          0x00U
#define PROTO_STATUS_CRC_ERR     0x01U
#define PROTO_STATUS_OOB         0x02U
#define PROTO_STATUS_UPDATE_DONE 0x04U

/* Globals modèles (allocation statique, pas de malloc) */
extern MahalanobisDetector g_detector;
extern EWCHead             g_ewc_head;
extern TinyOLEncoder       g_tinyol_enc;
extern TinyOLDecoder       g_tinyol_dec;

/* Stubs HAL — implémentés dans pipeline.c pour le MVP */
void uart_receive_sample(float *buf);
void led_set(int state);

/* Pipeline principal */
void pipeline_init(void);
void pipeline_run(void);
void pipeline_set_task(uint8_t task_id);  /* change g_current_task_id */
