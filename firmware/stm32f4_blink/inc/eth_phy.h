/**
 * eth_phy.h — Mise en veille du PHY Ethernet (LAN8742A) de la NUCLEO-144.
 *
 * POURQUOI (mesure d'énergie, S5001/S5002) :
 *   UM1974 §6.7 : « pour obtenir une consommation correcte [via JP5], le PHY
 *   Ethernet doit être mis en mode power-down, ou SB13 doit être retiré ».
 *   Le PHY est alimenté par le rail VDD_MCU — celui que mesure le PowerShield —
 *   alors que le firmware du projet n'utilise pas Ethernet : sans cette mise en
 *   veille, les µJ mesurés incluent un consommateur étranger aux modèles CL.
 *
 *   Contrainte supplémentaire mesurée sur banc (2026-08-04) : la carte tire
 *   ~59 mA, au-dessus du plafond du mode d'acquisition dynamique du LPM01A
 *   (« Overcurrent >59mA »). Retirer le PHY du rail est aussi la piste pour
 *   repasser sous ce seuil et débloquer le profilage temporel.
 *
 * MÉTHODE : écriture SMI/MDIO du bit Power Down (bit 11) du registre Basic
 *   Control (adresse 0x00) du PHY — UM1974 §6.11 note 3. Broches RMII de la
 *   Nucleo-144 : MDIO = PA2 (AF11), MDC = PC1 (AF11).
 *
 * ACTIVATION : compilée en dur, mais appelée uniquement sous
 *   `-DETH_PHY_POWERDOWN` (cf. main.c) — le build par défaut reste strictement
 *   inchangé, conformément à la convention du projet pour les options banc
 *   (`-DENERGY_MARKERS`, `-DDRIFT_DETECT`, `-DMAHA_INT8`…).
 */

#ifndef ETH_PHY_H
#define ETH_PHY_H

#include <stdint.h>

/* Adresse SMI du LAN8742A sur les cartes Nucleo-144 (strap par défaut). */
#define ETH_PHY_ADDR            0U

/* Registre Basic Control du PHY et son bit Power Down (IEEE 802.3 clause 22). */
#define ETH_PHY_REG_BCR         0x00U
#define ETH_PHY_BCR_POWERDOWN   (1U << 11)

/*
 * Champ CR de MACMIIAR = plage de HCLK, il fixe le diviseur d'horloge MDC
 * (qui doit rester ≤ 2,5 MHz). Le projet tourne à HCLK = 180 MHz (PLL 180 MHz,
 * HPRE /1, cf. hw_clock_init) → plage 168–216 MHz, code 0b101, MDC = HCLK/124
 * ≈ 1,45 MHz. RM0090, description de ETH_MACMIIAR.
 */
#define ETH_MACMIIAR_CR_HCLK_180MHZ  0x5U

/**
 * @brief Compose le mot de commande MACMIIAR d'un accès SMI.
 *
 * Fonction pure (aucun accès registre) : c'est elle qui porte la mise en forme
 * des champs, et elle est testable sur host.
 *
 * Format RM0090 : PA (adresse PHY) bits 15:11, MR (registre) bits 10:6,
 * CR (plage d'horloge) bits 4:2, MW (write) bit 1, MB (busy) bit 0.
 *
 * @param phy_addr Adresse SMI du PHY (0–31).
 * @param reg      Numéro de registre du PHY (0–31).
 * @param cr       Code de plage d'horloge (champ CR).
 * @param write    Non nul pour une écriture, nul pour une lecture.
 * @return Mot à écrire dans ETH_MACMIIAR, bit MB (busy) déjà armé.
 */
static inline uint32_t eth_phy_miiar_word(uint32_t phy_addr, uint32_t reg,
                                          uint32_t cr, int write)
{
    uint32_t word = ((phy_addr & 0x1FU) << 11)
                  | ((reg      & 0x1FU) <<  6)
                  | ((cr       & 0x07U) <<  2)
                  | 0x1U;                        /* MB : démarre l'échange */
    if (write) {
        word |= 0x2U;                            /* MW : écriture */
    }
    return word;
}

#ifndef TEST_HOST
/**
 * @brief Met le PHY Ethernet en veille (bit Power Down du registre BCR).
 *
 * À appeler après l'initialisation de l'horloge système (le champ CR dépend de
 * HCLK). Sans effet fonctionnel sur le projet : le firmware n'utilise pas
 * Ethernet, et l'horloge de référence RMII que le PHY cesse de fournir n'est
 * pas une source du système.
 */
void eth_phy_power_down(void);
#endif /* TEST_HOST */

#endif /* ETH_PHY_H */
