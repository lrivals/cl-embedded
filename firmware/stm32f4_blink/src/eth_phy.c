/**
 * eth_phy.c — Mise en veille du PHY Ethernet via SMI/MDIO (cf. inc/eth_phy.h).
 *
 * Aucune variable statique : empreinte `.bss` nulle.
 */

#include "eth_phy.h"
#include "stm32f4xx.h"

/* Délai d'attente du bit MB (busy) — borne franche pour ne jamais bloquer le
 * démarrage si le PHY ne répond pas (carte sans PHY, SB160/SB164 ouverts…). */
#define ETH_PHY_MII_TIMEOUT  100000U

/* ── Configuration des broches SMI : MDIO = PA2, MDC = PC1, AF11 ────────── */
static void eth_phy_gpio_init(void)
{
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN | RCC_AHB1ENR_GPIOCEN;
    (void)RCC->AHB1ENR;   /* barrière : horloges effectivement actives */

    /* PA2 — mode Alternate Function (10b), AF11 (Ethernet) */
    GPIOA->MODER   &= ~(3UL << (2U * 2U));
    GPIOA->MODER   |=  (2UL << (2U * 2U));
    GPIOA->OTYPER  &= ~(1UL << 2U);              /* push-pull */
    GPIOA->OSPEEDR |=  (3UL << (2U * 2U));       /* très haute vitesse */
    GPIOA->AFR[0]  &= ~(0xFUL << (2U * 4U));
    GPIOA->AFR[0]  |=  (11UL << (2U * 4U));

    /* PC1 — mode Alternate Function (10b), AF11 (Ethernet) */
    GPIOC->MODER   &= ~(3UL << (1U * 2U));
    GPIOC->MODER   |=  (2UL << (1U * 2U));
    GPIOC->OTYPER  &= ~(1UL << 1U);
    GPIOC->OSPEEDR |=  (3UL << (1U * 2U));
    GPIOC->AFR[0]  &= ~(0xFUL << (1U * 4U));
    GPIOC->AFR[0]  |=  (11UL << (1U * 4U));
}

/* ── Écriture d'un registre du PHY par SMI ──────────────────────────────── */
static void eth_phy_write(uint32_t reg, uint16_t value)
{
    uint32_t guard = ETH_PHY_MII_TIMEOUT;

    /* Attend la fin d'un éventuel échange en cours */
    while ((ETH_MAC->MACMIIAR & 0x1U) && guard--) {}

    ETH_MAC->MACMIIDR = (uint32_t)value;
    ETH_MAC->MACMIIAR = eth_phy_miiar_word(ETH_PHY_ADDR, reg,
                                           ETH_MACMIIAR_CR_HCLK_180MHZ, 1);

    guard = ETH_PHY_MII_TIMEOUT;
    while ((ETH_MAC->MACMIIAR & 0x1U) && guard--) {}
}

void eth_phy_power_down(void)
{
    /* L'horloge du MAC est nécessaire au seul bloc SMI (MACMIIAR/MACMIIDR) :
     * ni la transmission ni la réception ne sont activées. */
    RCC->AHB1ENR |= RCC_AHB1ENR_ETHMACEN;
    (void)RCC->AHB1ENR;

    eth_phy_gpio_init();
    eth_phy_write(ETH_PHY_REG_BCR, (uint16_t)ETH_PHY_BCR_POWERDOWN);
}
