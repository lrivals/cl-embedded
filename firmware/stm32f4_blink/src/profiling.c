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
#include "stm32f4xx.h"
#include <string.h>

#define SYSCLK_HZ    180000000U
#define DWT_CTRL    (*(volatile uint32_t *)0xE0001000UL)
#define DWT_CYCCNT  (*(volatile uint32_t *)0xE0001004UL)
#define CoreDebug_DEMCR (*(volatile uint32_t *)0xE000EDFCUL)
#define TRCENA_BIT  (1U << 24)

/* MEM: 20 B @ FP32/uint32 en .bss */
ProfilingState g_profiling;

void profiling_init(void)
{
    memset(&g_profiling, 0, sizeof(g_profiling));

    /* Active DWT cycle counter */
    CoreDebug_DEMCR |= TRCENA_BIT;
    DWT_CYCCNT = 0U;
    DWT_CTRL   |= 1U;

    /* Calcule taille .bss au link time via symboles linker */
    g_profiling.bss_bytes = (uint16_t)(
        (uintptr_t)&_ebss - (uintptr_t)&_sbss
    );
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
