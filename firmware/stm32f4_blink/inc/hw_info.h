#pragma once
#include <stdint.h>

/* MEM: HWInfo = 36 B @ FP32 — alloué en stack main(), libéré après hw_info_print() */
typedef struct {
    uint32_t cpuid;              /* SCB->CPUID  0xE000ED00 */
    uint32_t idcode;             /* DBGMCU->IDCODE  0xE0042000 */
    uint32_t sysclk_hz;          /* Fréquence système calculée depuis RCC */
    uint32_t hclk_hz;            /* HCLK = SYSCLK / AHB prescaler */
    uint32_t pclk1_hz;           /* APB1 clock */
    uint32_t pclk2_hz;           /* APB2 clock */
    uint32_t ram_total_bytes;    /* Constante linker : 256 Ko = 262144 B */
    uint32_t ram_used_bytes;     /* .data + .bss en bytes */
    uint32_t stack_free_bytes;   /* Estimation MSP - fin BSS */
} HWInfo;

/* ── Init hardware (appeler avant hw_info_collect) ── */
void hw_clock_init(void);        /* Configure PLL HSE 8 MHz → SYSCLK 180 MHz */
void hw_uart_init(void);         /* USART3 @ 115200, PD8=TX PD9=RX (ST-LINK VCP) */

/* ── Collecte et rapport ── */
void     hw_info_collect(HWInfo *info);
void     hw_info_print(const HWInfo *info);

/* ── DWT cycle counter (Cortex-M4) ── */
void     dwt_enable(void);
uint32_t dwt_cycles(void);       /* Lecture DWT->CYCCNT */
uint32_t hw_dwt_calibrate(uint32_t sysclk_hz); /* 1M loop → µs + affichage UART */
