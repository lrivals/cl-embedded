/**
 * stm32f4xx.h — Registres STM32F439ZI pour le projet CL-Embedded Phase 2
 *
 * Contient : GPIO A/D, RCC (complet AHB1/APB1/APB2), USART3.
 * Référence : STM32F439xx Reference Manual (RM0090), Rev 9
 */

#ifndef STM32F4XX_H
#define STM32F4XX_H

#include <stdint.h>

/* ── Adresses de base ──────────────────────────────────────────────────── */
#define PERIPH_BASE         0x40000000UL
#define APB1PERIPH_BASE     (PERIPH_BASE + 0x00000000UL)
#define AHB1PERIPH_BASE     (PERIPH_BASE + 0x00020000UL)

#define GPIOA_BASE          (AHB1PERIPH_BASE + 0x0000UL)
#define GPIOD_BASE          (AHB1PERIPH_BASE + 0x0C00UL)
#define RCC_BASE            (AHB1PERIPH_BASE + 0x3800UL)

#define USART3_BASE         (APB1PERIPH_BASE + 0x4800UL)

/* ── GPIO ──────────────────────────────────────────────────────────────── */
typedef struct {
    volatile uint32_t MODER;
    volatile uint32_t OTYPER;
    volatile uint32_t OSPEEDR;
    volatile uint32_t PUPDR;
    volatile uint32_t IDR;
    volatile uint32_t ODR;
    volatile uint32_t BSRR;
    volatile uint32_t LCKR;
    volatile uint32_t AFR[2];   /*!< [0]=AFRL pins 0-7, [1]=AFRH pins 8-15 */
} GPIO_TypeDef;

#define GPIOA   ((GPIO_TypeDef *) GPIOA_BASE)
#define GPIOD   ((GPIO_TypeDef *) GPIOD_BASE)

/* ── RCC ───────────────────────────────────────────────────────────────── */
typedef struct {
    volatile uint32_t CR;           /*!< 0x00 — Clock control */
    volatile uint32_t PLLCFGR;      /*!< 0x04 — PLL configuration */
    volatile uint32_t CFGR;         /*!< 0x08 — Clock configuration */
    volatile uint32_t CIR;          /*!< 0x0C — Clock interrupt */
    volatile uint32_t AHB1RSTR;     /*!< 0x10 */
    volatile uint32_t AHB2RSTR;     /*!< 0x14 */
    volatile uint32_t AHB3RSTR;     /*!< 0x18 */
    volatile uint32_t RESERVED0;    /*!< 0x1C */
    volatile uint32_t APB1RSTR;     /*!< 0x20 */
    volatile uint32_t APB2RSTR;     /*!< 0x24 */
    volatile uint32_t RESERVED1[2]; /*!< 0x28, 0x2C */
    volatile uint32_t AHB1ENR;      /*!< 0x30 — AHB1 clock enable */
    volatile uint32_t AHB2ENR;      /*!< 0x34 */
    volatile uint32_t AHB3ENR;      /*!< 0x38 */
    volatile uint32_t RESERVED2;    /*!< 0x3C */
    volatile uint32_t APB1ENR;      /*!< 0x40 — APB1 clock enable */
    volatile uint32_t APB2ENR;      /*!< 0x44 — APB2 clock enable */
} RCC_TypeDef;

#define RCC     ((RCC_TypeDef *)  RCC_BASE)

/* RCC CR bits */
#define RCC_CR_HSION        (1UL <<  0)
#define RCC_CR_HSIRDY       (1UL <<  1)
#define RCC_CR_HSEON        (1UL << 16)
#define RCC_CR_HSERDY       (1UL << 17)
#define RCC_CR_PLLON        (1UL << 24)
#define RCC_CR_PLLRDY       (1UL << 25)

/* RCC AHB1ENR bits */
#define RCC_AHB1ENR_GPIOAEN (1UL << 0)
#define RCC_AHB1ENR_GPIOBEN (1UL << 1)
#define RCC_AHB1ENR_GPIOCEN (1UL << 2)
#define RCC_AHB1ENR_GPIODEN (1UL << 3)
#define RCC_AHB1ENR_DMA1EN  (1UL << 21)
#define RCC_AHB1ENR_DMA2EN  (1UL << 22)

/* RCC APB1ENR bits */
#define RCC_APB1ENR_TIM2EN  (1UL <<  0)
#define RCC_APB1ENR_TIM3EN  (1UL <<  1)
#define RCC_APB1ENR_TIM4EN  (1UL <<  2)
#define RCC_APB1ENR_SPI2EN  (1UL << 14)
#define RCC_APB1ENR_SPI3EN  (1UL << 15)
#define RCC_APB1ENR_USART2EN (1UL << 17)
#define RCC_APB1ENR_USART3EN (1UL << 18)
#define RCC_APB1ENR_I2C1EN  (1UL << 21)
#define RCC_APB1ENR_I2C2EN  (1UL << 22)
#define RCC_APB1ENR_PWREN   (1UL << 28)

/* RCC APB2ENR bits */
#define RCC_APB2ENR_TIM1EN  (1UL <<  0)
#define RCC_APB2ENR_USART1EN (1UL << 4)
#define RCC_APB2ENR_USART6EN (1UL << 5)
#define RCC_APB2ENR_ADC1EN  (1UL <<  8)
#define RCC_APB2ENR_SPI1EN  (1UL << 12)
#define RCC_APB2ENR_SYSCFGEN (1UL << 14)

/* ── USART ─────────────────────────────────────────────────────────────── */
typedef struct {
    volatile uint32_t SR;    /*!< 0x00 — Status */
    volatile uint32_t DR;    /*!< 0x04 — Data */
    volatile uint32_t BRR;   /*!< 0x08 — Baud rate */
    volatile uint32_t CR1;   /*!< 0x0C — Control 1 */
    volatile uint32_t CR2;   /*!< 0x10 — Control 2 */
    volatile uint32_t CR3;   /*!< 0x14 — Control 3 */
    volatile uint32_t GTPR;  /*!< 0x18 — Guard time and prescaler */
} USART_TypeDef;

#define USART3  ((USART_TypeDef *) USART3_BASE)

/* USART SR bits */
#define USART_SR_TXE    (1UL << 7)   /* TX data register empty */
#define USART_SR_TC     (1UL << 6)   /* Transmission complete */

/* USART CR1 bits */
#define USART_CR1_UE    (1UL << 13)  /* USART enable */
#define USART_CR1_TE    (1UL <<  3)  /* Transmitter enable */
#define USART_CR1_RE    (1UL <<  2)  /* Receiver enable */

/* ── Instruction NOP ───────────────────────────────────────────────────── */
#define __NOP()  __asm volatile ("nop")

#endif /* STM32F4XX_H */
