#pragma once
#include <stdint.h>
#include "mahalanobis.h"

/* LED LD2 = PA5 sur NUCLEO-F439ZI */
#define LED_PIN 5U
#define LED_ON  1
#define LED_OFF 0

/* Détecteur global (allocation statique, pas de malloc) */
extern MahalanobisDetector g_detector;

/* Stubs HAL — implémentés dans pipeline.c pour le MVP */
void uart_receive_sample(float *buf);
void led_set(int state);

/* Pipeline principal */
void pipeline_init(void);
void pipeline_run(void);
