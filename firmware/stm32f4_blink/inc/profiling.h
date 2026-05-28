#pragma once
#include <stdint.h>

/* ── Profiling firmware : latence DWT, empreinte .bss, throughput ──────────
 *
 * Usage :
 *   profiling_start()  — arme le compteur DWT avant inférence
 *   profiling_stop()   — capture et stocke durée + RAM
 *   profiling_report() — encode les métriques dans la réponse UART
 *
 * Pas de malloc. Toutes les données dans ProfilingState statique (.bss).
 */

/* MEM: 20 B @ FP32/uint32 en .bss */
typedef struct {
    uint32_t t_start_cycles;    /* DWT CYCCNT au début de l'inférence */
    uint32_t last_latency_us;   /* Dernière latence mesurée en µs */
    uint16_t bss_bytes;         /* Taille .bss calculée au link time */
    uint16_t throughput_ips;    /* Inférences par seconde (glissant) */
    uint32_t inference_count;   /* Compteur total d'inférences */
    uint32_t total_cycles;      /* Cycles accumulés sur la session */
} ProfilingState;

/* Symboles fournis par le linker script (calculés au link time) */
#ifndef TEST_HOST
extern uint32_t _sbss;   /* Début segment .bss */
extern uint32_t _ebss;   /* Fin segment .bss  */
#endif

extern ProfilingState g_profiling;

void     profiling_init(void);
void     profiling_start(void);
void     profiling_stop(void);

uint32_t profiling_get_latency_us(void);
uint16_t profiling_get_bss_bytes(void);
uint16_t profiling_get_throughput_ips(void);

/* Encode [latency_us:u32][ram_used_b:u16][throughput:u16] = 8 B dans buf */
void profiling_encode(uint8_t *buf);
#define PROFILING_ENCODED_SIZE 8U
