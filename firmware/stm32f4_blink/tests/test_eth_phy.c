/**
 * test_eth_phy.c — Tests Unity de la mise en veille du PHY Ethernet (S5001).
 *
 * Seule la composition du mot MACMIIAR est testable sur host : les accès
 * registre eux-mêmes se valident sur carte (courant mesuré avant/après).
 * Référence des champs : RM0090, ETH_MACMIIAR.
 */

#include "unity.h"
#include "eth_phy.h"

/* ── Test 1 : champs PA / MR correctement placés ────────────────────────── */
void test_eth_phy_miiar_fields(void)
{
    /* PHY 3, registre 5, CR 0b101, écriture :
     *   PA 15:11 = 3 → 0x1800, MR 10:6 = 5 → 0x0140, CR 4:2 = 5 → 0x0014,
     *   MW bit 1 → 0x2, MB bit 0 → 0x1  ⇒  0x1800|0x140|0x14|0x3 = 0x1957. */
    uint32_t w = eth_phy_miiar_word(3U, 5U, 0x5U, 1);

    TEST_ASSERT_EQUAL_UINT32(3U, (w >> 11) & 0x1FU);
    TEST_ASSERT_EQUAL_UINT32(5U, (w >> 6) & 0x1FU);
    TEST_ASSERT_EQUAL_UINT32(0x5U, (w >> 2) & 0x7U);
    TEST_ASSERT_EQUAL_UINT32(0x1957U, w);
}

/* ── Test 2 : le bit MB (busy) arme toujours l'échange ──────────────────── */
void test_eth_phy_miiar_busy_always_set(void)
{
    TEST_ASSERT_EQUAL_UINT32(1U, eth_phy_miiar_word(0U, 0U, 0U, 0) & 0x1U);
    TEST_ASSERT_EQUAL_UINT32(1U, eth_phy_miiar_word(0U, 0U, 0U, 1) & 0x1U);
}

/* ── Test 3 : le bit MW distingue lecture et écriture ───────────────────── */
void test_eth_phy_miiar_write_bit(void)
{
    TEST_ASSERT_EQUAL_UINT32(0U, eth_phy_miiar_word(0U, 0U, 0U, 0) & 0x2U);
    TEST_ASSERT_EQUAL_UINT32(0x2U, eth_phy_miiar_word(0U, 0U, 0U, 1) & 0x2U);
}

/* ── Test 4 : cible réelle du banc — PHY 0, BCR, bit Power Down ─────────── */
void test_eth_phy_powerdown_target(void)
{
    /* Le bit Power Down est le bit 11 du registre Basic Control (adresse 0). */
    TEST_ASSERT_EQUAL_UINT32(0x0800U, ETH_PHY_BCR_POWERDOWN);
    TEST_ASSERT_EQUAL_UINT32(0x00U, ETH_PHY_REG_BCR);

    /* Adresse SMI du LAN8742A sur Nucleo-144 : 0. */
    uint32_t w = eth_phy_miiar_word(ETH_PHY_ADDR, ETH_PHY_REG_BCR,
                                    ETH_MACMIIAR_CR_HCLK_180MHZ, 1);
    TEST_ASSERT_EQUAL_UINT32(0U, (w >> 11) & 0x1FU);
    TEST_ASSERT_EQUAL_UINT32(0U, (w >> 6) & 0x1FU);
    /* HCLK = 180 MHz → plage 168–216 MHz, MDC = HCLK/124 ≈ 1,45 MHz ≤ 2,5 MHz. */
    TEST_ASSERT_EQUAL_UINT32(0x5U, (w >> 2) & 0x7U);
}
