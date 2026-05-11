/**
 * hw_info.c — Caractérisation matérielle STM32F439ZI (S1004)
 *
 * - Config PLL : HSE 8 MHz → SYSCLK 180 MHz (HCLK=180, APB1=45, APB2=90)
 * - UART3 TX polling @ 115200 sur PD8/PD9 (ST-LINK Virtual COM Port)
 * - DWT cycle counter pour benchmarks latence (réutilisé dans S1003+)
 * - Lecture registres RCC, SCB, DBGMCU → rapport UART
 *
 * Aucun malloc, aucun printf. Toutes les variables en stack ou Flash.
 * Référence : RM0090 (STM32F439 Reference Manual), Rev 9
 */

#include "hw_info.h"
#include "stm32f4xx.h"

/* ── Registres ARM Cortex-M4 (accès direct par adresse) ─────────────────── */
#define SCB_CPUID        (*(volatile uint32_t *)0xE000ED00UL)
#define CoreDebug_DEMCR  (*(volatile uint32_t *)0xE000EDFCUL)  /* Trace enable */
#define DWT_CTRL         (*(volatile uint32_t *)0xE0001000UL)
#define DWT_CYCCNT       (*(volatile uint32_t *)0xE0001004UL)

/* ── Registres ST-spécifiques ───────────────────────────────────────────── */
#define DBGMCU_IDCODE    (*(volatile uint32_t *)0xE0042000UL)

/* ── Power + Flash (pour config PLL > 168 MHz) ──────────────────────────── */
#define PWR_CR           (*(volatile uint32_t *)0x40007000UL)
#define PWR_CSR          (*(volatile uint32_t *)0x40007004UL)
#define FLASH_ACR        (*(volatile uint32_t *)0x40023C00UL)

/* PWR_CR bits */
#define PWR_CR_VOS_SCALE1   (3UL << 14)   /* Voltage scaling output 1 (max perf) */
#define PWR_CR_ODEN         (1UL << 16)   /* Over-drive enable */
#define PWR_CR_ODSWEN       (1UL << 17)   /* Over-drive switching enable */

/* PWR_CSR bits */
#define PWR_CSR_ODRDY       (1UL << 16)   /* Over-drive ready */
#define PWR_CSR_ODSWRDY     (1UL << 17)   /* Over-drive switching ready */

/* FLASH_ACR bits */
#define FLASH_ACR_LATENCY_5WS  5UL
#define FLASH_ACR_PRFTEN    (1UL << 8)
#define FLASH_ACR_ICEN      (1UL << 9)
#define FLASH_ACR_DCEN      (1UL << 10)

/* ── Linker symbols (définis dans STM32F439ZITx_FLASH.ld) ───────────────── */
extern uint32_t _sdata, _edata, _sbss, _ebss, _estack;

/* ── Helpers UART TX polling ────────────────────────────────────────────── */

static void uart_putchar(char c)
{
    while (!(USART3->SR & USART_SR_TXE)) {}
    USART3->DR = (uint32_t)(uint8_t)c;
}

static void uart_puts(const char *s)
{
    while (*s) uart_putchar(*s++);
}

static void uart_put_uint32(uint32_t n)
{
    char buf[10];
    int idx = 0;
    if (n == 0) { uart_putchar('0'); return; }
    while (n > 0) { buf[idx++] = (char)('0' + (int)(n % 10u)); n /= 10u; }
    while (idx > 0) { uart_putchar(buf[--idx]); }
}

static void uart_put_hex32(uint32_t n)
{
    static const char hex[] = "0123456789ABCDEF";
    uart_puts("0x");
    for (int i = 7; i >= 0; i--) {
        uart_putchar(hex[(n >> ((unsigned)i * 4u)) & 0xFu]);
    }
}

/* Print frequency as "xxx MHz" or "xx MHz" */
static void uart_put_mhz(uint32_t hz)
{
    uart_put_uint32(hz / 1000000u);
    uart_puts(" MHz");
}

/* Print size as "xxx Ko" */
static void uart_put_ko(uint32_t bytes)
{
    uart_put_uint32(bytes / 1024u);
    uart_puts(" Ko");
}

/* ── hw_clock_init — PLL HSI 16 MHz → SYSCLK 180 MHz ───────────────────── */
/*
 * Source : HSI interne (16 MHz) — pas de cristal externe requis.
 * Config : PLLM=8, PLLN=180, PLLP=2, SRC=HSI
 *   VCO input  = 16 / 8       = 2 MHz
 *   VCO output = 2 × 180      = 360 MHz
 *   SYSCLK     = 360 / 2      = 180 MHz
 *   HCLK       = 180 / 1      = 180 MHz  (HPRE=0)
 *   PCLK1      = 180 / 4      = 45 MHz   (PPRE1=101b)  → BRR UART inchangé
 *   PCLK2      = 180 / 2      = 90 MHz   (PPRE2=100b)
 */
void hw_clock_init(void)
{
    /* 0. Active le FPU (CP10/CP11 full access) — requis avant toute instruction VFP */
    *(volatile uint32_t *)0xE000ED88UL |= (0xFUL << 20);

    /* 1. HSI est déjà actif après reset — pas besoin d'activer HSE */

    /* 2. Active le contrôleur d'alimentation (PWREN = APB1ENR bit 28) */
    RCC->APB1ENR |= RCC_APB1ENR_PWREN;
    (void)RCC->APB1ENR;    /* barrière : garantit que l'horloge PWR est active */

    /* 3. Voltage scaling 1 (requis > 168 MHz) */
    PWR_CR = (PWR_CR & ~(3UL << 14)) | PWR_CR_VOS_SCALE1;

    /* 4. Overdrive (requis > 168 MHz sur STM32F42x/43x) */
    PWR_CR |= PWR_CR_ODEN;
    while (!(PWR_CSR & PWR_CSR_ODRDY)) {}
    PWR_CR |= PWR_CR_ODSWEN;
    while (!(PWR_CSR & PWR_CSR_ODSWRDY)) {}

    /* 5. Flash : 5 wait states + prefetch + instruction/data cache */
    FLASH_ACR = FLASH_ACR_LATENCY_5WS | FLASH_ACR_PRFTEN | FLASH_ACR_ICEN | FLASH_ACR_DCEN;

    /* 6. Prescalers : AHB=/1 (0000), APB1=/4 (101), APB2=/2 (100) */
    RCC->CFGR = (RCC->CFGR & ~(0xFFCUL << 4))
              | (0x0UL  <<  4)   /* HPRE  : /1 */
              | (0x5UL  << 10)   /* PPRE1 : /4 */
              | (0x4UL  << 13);  /* PPRE2 : /2 */

    /* 7. Configure PLL : PLLM=8, PLLN=180, PLLP=2, SRC=HSI (bit22=0) */
    RCC->PLLCFGR = (8UL   <<  0)   /* PLLM : HSI/8 = 2 MHz VCO_in */
                 | (180UL <<  6)   /* PLLN : 2×180 = 360 MHz VCO_out */
                 | (0UL   << 16)   /* PLLP : 00 = /2 → SYSCLK 180 MHz */
                 | (0UL   << 22);  /* PLLSRC : HSI (0) */

    /* 8. Active PLL et attend prêt */
    RCC->CR |= RCC_CR_PLLON;
    while (!(RCC->CR & RCC_CR_PLLRDY)) {}

    /* 9. Bascule sur PLL (SW=10) et attend confirmation (SWS=10) */
    RCC->CFGR = (RCC->CFGR & ~3UL) | 2UL;
    while ((RCC->CFGR & (3UL << 2)) != (2UL << 2)) {}
}

/* ── hw_uart_init — USART3 @ 115200 sur PD8/PD9 ────────────────────────── */
/*
 * PD8 = USART3_TX (AF7), PD9 = USART3_RX (AF7)
 * BRR @ PCLK1=45 MHz, OVER8=0 (16x) : USARTDIV = fCK / (16 × baud)
 *   USARTDIV = 45000000 / (16 × 115200) = 24.414
 *   Mantissa = 24 = 0x18, Fraction = round(0.414×16) = 7
 *   BRR = (0x18 << 4) | 7 = 0x0187  → 115089 baud (<0.1% error)
 */
void hw_uart_init(void)
{
    /* Active horloges GPIOD + USART3 */
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIODEN;
    RCC->APB1ENR |= RCC_APB1ENR_USART3EN;

    /* PD8/PD9 : mode Alternate Function (MODER = 10b) */
    GPIOD->MODER &= ~((3UL << (8u * 2u)) | (3UL << (9u * 2u)));
    GPIOD->MODER |=  ( (2UL << (8u * 2u)) | (2UL << (9u * 2u)));

    /* PD8/PD9 : AF7 (USART3) dans AFRH (AFR[1]) */
    GPIOD->AFR[1] &= ~((0xFUL << 0u) | (0xFUL << 4u));   /* bits [7:0] = PD8/PD9 */
    GPIOD->AFR[1] |=  ( (7UL  << 0u) | (7UL  << 4u));    /* AF7 */

    /* PD8/PD9 : vitesse haute, pas de pull */
    GPIOD->OSPEEDR |= (3UL << (8u * 2u)) | (3UL << (9u * 2u));
    GPIOD->PUPDR   &= ~((3UL << (8u * 2u)) | (3UL << (9u * 2u)));

    /* USART3 config : 115200 baud, 8N1, TX+RX */
    USART3->BRR = 0x0187UL;               /* 45 MHz / (16 × 115200) = 24.414 → 0x0187 */
    USART3->CR1 = USART_CR1_UE | USART_CR1_TE | USART_CR1_RE;
}

/* ── DWT cycle counter ──────────────────────────────────────────────────── */

void dwt_enable(void)
{
    CoreDebug_DEMCR |= (1UL << 24);   /* TRCENA : active le trace/debug */
    DWT_CYCCNT = 0UL;
    DWT_CTRL  |= 1UL;                 /* CYCCNTENA */
}

uint32_t dwt_cycles(void)
{
    return DWT_CYCCNT;
}

/* ── hw_info_collect ────────────────────────────────────────────────────── */

/* Tables de diviseurs AHB/APB (index = valeur du champ HPRE/PPRE) */
static const uint16_t k_ahb_div[16] = {
    1,1,1,1,1,1,1,1, 2,4,8,16,64,128,256,512
};
static const uint8_t k_apb_div[8] = {
    1,1,1,1,2,4,8,16
};

void hw_info_collect(HWInfo *info)
{
    /* -- Identification MCU -- */
    info->cpuid  = SCB_CPUID;
    info->idcode = DBGMCU_IDCODE;

    /* -- Horloges depuis RCC -- */
    uint32_t cfgr    = RCC->CFGR;
    uint32_t pllcfgr = RCC->PLLCFGR;
    uint32_t sws     = (cfgr >> 2u) & 3u;   /* SWS[3:2] = source active */

    uint32_t sysclk;
    if (sws == 0u) {
        sysclk = 16000000u;   /* HSI */
    } else if (sws == 1u) {
        sysclk = 8000000u;    /* HSE */
    } else {
        uint32_t pllm   =  pllcfgr & 0x3Fu;
        uint32_t plln   = (pllcfgr >> 6u)  & 0x1FFu;
        uint32_t pllp   = (((pllcfgr >> 16u) & 3u) + 1u) * 2u;  /* 00→2, 01→4, 10→6, 11→8 */
        uint32_t pllsrc = (pllcfgr >> 22u) & 1u;
        uint32_t fin    = pllsrc ? 8000000u : 16000000u;
        sysclk = (fin / pllm) * plln / pllp;
    }

    uint32_t hpre  = (cfgr >>  4u) & 0xFu;
    uint32_t ppre1 = (cfgr >> 10u) & 0x7u;
    uint32_t ppre2 = (cfgr >> 13u) & 0x7u;

    info->sysclk_hz = sysclk;
    info->hclk_hz   = sysclk / k_ahb_div[hpre];
    info->pclk1_hz  = info->hclk_hz / k_apb_div[ppre1];
    info->pclk2_hz  = info->hclk_hz / k_apb_div[ppre2];

    /* -- RAM -- */
    uint32_t data_sz = (uint32_t)(&_edata) - (uint32_t)(&_sdata);
    uint32_t bss_sz  = (uint32_t)(&_ebss)  - (uint32_t)(&_sbss);
    info->ram_total_bytes = 256u * 1024u;
    info->ram_used_bytes  = data_sz + bss_sz;

    uint32_t msp;
    __asm volatile("mrs %0, msp" : "=r"(msp));
    uint32_t ebss_addr = (uint32_t)(&_ebss);
    info->stack_free_bytes = (msp > ebss_addr) ? (msp - ebss_addr) : 0u;
}

/* ── hw_info_print ──────────────────────────────────────────────────────── */

/* Affiche la liste des périphériques dont le clock enable est actif */
static void print_active_periph(void)
{
    uint32_t ahb1 = RCC->AHB1ENR;
    uint32_t apb1 = RCC->APB1ENR;
    uint32_t apb2 = RCC->APB2ENR;

    if (ahb1 & RCC_AHB1ENR_GPIOAEN)  uart_puts("GPIOA ");
    if (ahb1 & RCC_AHB1ENR_GPIOBEN)  uart_puts("GPIOB ");
    if (ahb1 & RCC_AHB1ENR_GPIOCEN)  uart_puts("GPIOC ");
    if (ahb1 & RCC_AHB1ENR_GPIODEN)  uart_puts("GPIOD ");
    if (ahb1 & RCC_AHB1ENR_DMA1EN)   uart_puts("DMA1 ");
    if (ahb1 & RCC_AHB1ENR_DMA2EN)   uart_puts("DMA2 ");
    if (apb1 & RCC_APB1ENR_TIM2EN)   uart_puts("TIM2 ");
    if (apb1 & RCC_APB1ENR_TIM3EN)   uart_puts("TIM3 ");
    if (apb1 & RCC_APB1ENR_TIM4EN)   uart_puts("TIM4 ");
    if (apb1 & RCC_APB1ENR_SPI2EN)   uart_puts("SPI2 ");
    if (apb1 & RCC_APB1ENR_SPI3EN)   uart_puts("SPI3 ");
    if (apb1 & RCC_APB1ENR_USART2EN) uart_puts("USART2 ");
    if (apb1 & RCC_APB1ENR_USART3EN) uart_puts("USART3 ");
    if (apb1 & RCC_APB1ENR_I2C1EN)   uart_puts("I2C1 ");
    if (apb1 & RCC_APB1ENR_I2C2EN)   uart_puts("I2C2 ");
    if (apb1 & RCC_APB1ENR_PWREN)    uart_puts("PWR ");
    if (apb2 & RCC_APB2ENR_TIM1EN)   uart_puts("TIM1 ");
    if (apb2 & RCC_APB2ENR_USART1EN) uart_puts("USART1 ");
    if (apb2 & RCC_APB2ENR_USART6EN) uart_puts("USART6 ");
    if (apb2 & RCC_APB2ENR_ADC1EN)   uart_puts("ADC1 ");
    if (apb2 & RCC_APB2ENR_SPI1EN)   uart_puts("SPI1 ");
}

void hw_info_print(const HWInfo *info)
{
    uart_puts("\r\n=== HW INFO -- STM32F439ZI ===\r\n");

    uart_puts("MCU : STM32F439ZI  IDCODE=");
    uart_put_hex32(info->idcode);
    uart_puts("\r\n");

    uart_puts("CPU : Cortex-M4  CPUID=");
    uart_put_hex32(info->cpuid);
    uart_puts("\r\n");

    uart_puts("SYSCLK : ");  uart_put_mhz(info->sysclk_hz);
    uart_puts("  HCLK : ");  uart_put_mhz(info->hclk_hz);
    uart_puts("  PCLK1 : "); uart_put_mhz(info->pclk1_hz);
    uart_puts("  PCLK2 : "); uart_put_mhz(info->pclk2_hz);
    uart_puts("\r\n");

    uart_puts("RAM total : "); uart_put_ko(info->ram_total_bytes);
    uart_puts("  BSS+data : ");
    uart_put_uint32(info->ram_used_bytes);
    uart_puts(" B  Stack libre : ");
    uart_put_ko(info->stack_free_bytes);
    uart_puts("\r\n");

    uart_puts("Peripheriques actifs : ");
    print_active_periph();
    uart_puts("\r\n");

    uart_puts("=============================\r\n");
}

/* ── dwt_calibrate — mesure 1M itérations, retourne µs ─────────────────── */

uint32_t hw_dwt_calibrate(uint32_t sysclk_hz)
{
    dwt_enable();
    uint32_t t0 = dwt_cycles();
    for (volatile uint32_t i = 0u; i < 1000000u; i++) {}
    uint32_t elapsed = dwt_cycles() - t0;
    /* Conversion en µs : elapsed / (sysclk_hz / 1_000_000) */
    uint32_t us = elapsed / (sysclk_hz / 1000000u);

    uart_puts("DWT calib : 1 000 000 cycles = ");
    uart_put_uint32(us);
    uart_puts(" us @ ");
    uart_put_mhz(sysclk_hz);
    uart_puts("\r\n");

    return us;
}
