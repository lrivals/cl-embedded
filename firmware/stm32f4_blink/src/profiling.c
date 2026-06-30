/**
 * profiling.c — Mesure automatique latence (DWT), taille .bss, throughput
 *
 * Activé dès le boot par profiling_init(). Chaque appel profiling_start/stop
 * met à jour ProfilingState g_profiling (allocation statique .bss).
 *
 * Compatible STM32F439ZI Cortex-M4 et STM32N6 Cortex-M55.
 * Pas de malloc, pas de stdlib dynamique.
 */

#include "profiling.h"
#ifndef TEST_HOST
#include "stm32f4xx.h"
#endif
#include <string.h>

#define SYSCLK_HZ    180000000U

#ifndef TEST_HOST
#define DWT_CTRL        (*(volatile uint32_t *)0xE0001000UL)
#define DWT_CYCCNT      (*(volatile uint32_t *)0xE0001004UL)
#define CoreDebug_DEMCR (*(volatile uint32_t *)0xE000EDFCUL)
#define TRCENA_BIT      (1U << 24)
#else
volatile uint32_t g_mock_cyccnt = 0U;
#define DWT_CYCCNT      g_mock_cyccnt
#define DWT_CTRL        g_mock_cyccnt
#define CoreDebug_DEMCR g_mock_cyccnt
#define TRCENA_BIT      0U
#endif

/* MEM: 20 B @ FP32/uint32 en .bss */
ProfilingState g_profiling;

void profiling_init(void)
{
    memset(&g_profiling, 0, sizeof(g_profiling));

#ifndef TEST_HOST
    /* Active DWT cycle counter */
    CoreDebug_DEMCR |= TRCENA_BIT;
    DWT_CYCCNT = 0U;
    DWT_CTRL   |= 1U;

    /* Calcule taille .bss au link time via symboles linker */
    g_profiling.bss_bytes = (uint16_t)(
        (uintptr_t)&_ebss - (uintptr_t)&_sbss
    );
#endif
}

void profiling_start(void)
{
    g_profiling.t_start_cycles = DWT_CYCCNT;
}

void profiling_stop(void)
{
    uint32_t elapsed = DWT_CYCCNT - g_profiling.t_start_cycles;
    g_profiling.last_latency_us = elapsed / (SYSCLK_HZ / 1000000U);

    g_profiling.inference_count++;
    g_profiling.total_cycles += elapsed;

    /* Throughput glissant : moyenne sur toute la session */
    if (g_profiling.total_cycles > 0U) {
        uint32_t avg_cycles = g_profiling.total_cycles / g_profiling.inference_count;
        if (avg_cycles > 0U) {
            g_profiling.throughput_ips = (uint16_t)(SYSCLK_HZ / avg_cycles);
        }
    }
}

uint32_t profiling_get_latency_us(void)
{
    return g_profiling.last_latency_us;
}

uint16_t profiling_get_bss_bytes(void)
{
    return g_profiling.bss_bytes;
}

uint16_t profiling_get_throughput_ips(void)
{
    return g_profiling.throughput_ips;
}

/* ── Pic de pile (stack high-water mark) ─────────────────────────────────── */

/* Symboles linker pour .data (le scan .bss utilise déjà _sbss/_ebss). */
#ifndef TEST_HOST
extern uint32_t _sdata, _edata;
#endif

/* Fonction pure, testable sur host : la pile croît vers le bas depuis `high`.
 * Le startup a peint [low, high) avec `sentinel` ; on scanne de `low` (bas)
 * vers le haut jusqu'au premier mot écrasé → tout ce qui est au-dessus a été
 * touché par la pile. Retourne (high - premier_mot_utilisé) en octets. */
uint32_t profiling_stack_peak_from_region(const uint32_t *low,
                                          const uint32_t *high,
                                          uint32_t sentinel)
{
    const uint32_t *p = low;
    while (p < high && *p == sentinel) {
        p++;
    }
    return (uint32_t)((const uint8_t *)high - (const uint8_t *)p);
}

uint32_t profiling_stack_peak_bytes(void)
{
#ifndef TEST_HOST
    return profiling_stack_peak_from_region(&_ebss, &_estack,
                                            STACK_PAINT_SENTINEL);
#else
    return 0U;
#endif
}

uint32_t profiling_ram_peak_bytes(void)
{
#ifndef TEST_HOST
    uint32_t data_bytes = (uint32_t)((uintptr_t)&_edata - (uintptr_t)&_sdata);
    uint32_t bss_bytes  = (uint32_t)((uintptr_t)&_ebss - (uintptr_t)&_sbss);
    return data_bytes + bss_bytes + profiling_stack_peak_bytes();
#else
    return 0U;
#endif
}

/* Encode [latency_us:u32][ram_b:u16][throughput:u16] = 8 B dans buf */
void profiling_encode(uint8_t *buf)
{
    uint32_t lat = g_profiling.last_latency_us;
    uint16_t ram = g_profiling.bss_bytes;
    uint16_t thr = g_profiling.throughput_ips;

    buf[0] = (uint8_t)(lat & 0xFFU);
    buf[1] = (uint8_t)((lat >> 8)  & 0xFFU);
    buf[2] = (uint8_t)((lat >> 16) & 0xFFU);
    buf[3] = (uint8_t)((lat >> 24) & 0xFFU);
    buf[4] = (uint8_t)(ram & 0xFFU);
    buf[5] = (uint8_t)((ram >> 8) & 0xFFU);
    buf[6] = (uint8_t)(thr & 0xFFU);
    buf[7] = (uint8_t)((thr >> 8) & 0xFFU);
}

/* Marqueur de phase énergie (S3304) — toggle GPIO PA8 corrélé au DWT.
 *
 * No-op complet si ENERGY_MARKERS n'est pas défini (macros vides → le
 * compilateur élimine le corps). La corrélation temps↔énergie est assurée
 * par l'adjacence avec profiling_start() côté pipeline.c (le passage en
 * PHASE_INFERENCE est placé juste avant l'appel profiling_start existant).
 */
void energy_marker_phase(EnergyPhase phase)
{
    switch (phase) {
    case PHASE_STARTUP:
    case PHASE_ACQUISITION:
    case PHASE_INFERENCE:
        ENERGY_MARKER_SET();      /* haut : phase active */
        break;
    case PHASE_IDLE:
    default:
        ENERGY_MARKER_CLEAR();    /* bas : retour en attente */
        break;
    }
    (void)phase;  /* évite -Wunused quand les macros sont vides */
}
