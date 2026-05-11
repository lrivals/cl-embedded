# S1004 — Caractérisation matérielle : inventaire ressources embarquées

| Champ | Valeur |
|-------|--------|
| **ID** | S1004 |
| **Sprint** | Sprint 16 — Semaine 4 (11–17 juin 2026) |
| **Priorité** | Important |
| **Durée estimée** | 5h |
| **Dépendances** | S1001 (toolchain + UART opérationnel) |
| **Fichiers cibles** | `firmware/stm32f4_blink/src/hw_info.c`, `firmware/stm32f4_blink/inc/hw_info.h` |

---

## Objectif

Lire et rapporter depuis le MCU lui-même l'état réel des ressources hardware disponibles sur la **NUCLEO-F439ZI** (STM32F439ZI, Cortex-M4 @ 180 MHz, 256 Ko RAM).

**Motivations** :
- Valider le budget mémoire réel (Gap 2) avant de porter les modèles ML
- Disposer d'une primitive de mesure de latence (DWT cycle counter) pour les benchmarks S1003+
- Préparer une procédure reproductible transposable sur STM32N6

**Sortie cible** (via UART, lisible sur minicom/screen) :

```
=== HW INFO — STM32F439ZI ===
MCU : STM32F439ZI  IDCODE=0x20036413
CPU : Cortex-M4  CPUID=0x410FC241
SYSCLK : 180 MHz  HCLK : 180 MHz  PCLK1 : 45 MHz  PCLK2 : 90 MHz
RAM total : 256 Ko  BSS+data : ~1.2 Ko  Stack libre estimé : ~254 Ko
DWT calib : 1 000 000 cycles = 5 555 us @ 180 MHz
Périphériques actifs : USART3 SPI1 I2C1 ADC1 TIM2 TIM3
=============================
```

---

## Tâches

| ID | Tâche | Priorité | Fichier cible | Durée | Dépendances |
|----|-------|:---:|---------------|:---:|-------------|
| S1004-01 | Lire CPUID + IDCODE → afficher référence MCU via UART | 🔴 | `src/hw_info.c` | 1h | S1001 |
| S1004-02 | Lire registres RCC → calculer SYSCLK / HCLK / PCLK1 / PCLK2 | 🔴 | `src/hw_info.c` | 1h | S1004-01 |
| S1004-03 | Estimer RAM disponible (MSP initial − BSS/data − stack actuel) | 🔴 | `src/hw_info.c` | 1h | S1004-01 |
| S1004-04 | Activer DWT cycle counter + mesurer latence calibration (boucle vide) | 🟡 | `src/hw_info.c` | 1h | S1004-02 |
| S1004-05 | Inventaire périphériques activés via registres RCC (UART/SPI/I2C/ADC/TIM) | 🟢 | `src/hw_info.c` | 1h | S1004-01 |

---

## Interface `hw_info.h`

```c
#pragma once
#include <stdint.h>

/* Résultat de l'inventaire hardware — alloué en stack (usage ponctuel) */
typedef struct {
    uint32_t cpuid;        /* SCB->CPUID */
    uint32_t idcode;       /* DBGMCU->IDCODE */
    uint32_t sysclk_hz;    /* Fréquence système calculée depuis RCC */
    uint32_t hclk_hz;
    uint32_t pclk1_hz;
    uint32_t pclk2_hz;
    uint32_t ram_total_bytes;    /* Constante linker (fin SRAM - début SRAM) */
    uint32_t ram_used_bytes;     /* BSS + data segments */
    uint32_t stack_free_bytes;   /* Estimation MSP vs. __stack_start__ */
} HWInfo;

void hw_info_collect(HWInfo *info);
void hw_info_print(const HWInfo *info);   /* Sortie via UART */
void dwt_enable(void);
uint32_t dwt_cycles(void);               /* Lecture DWT->CYCCNT */
```

**Budget RAM** : `HWInfo` = 9 × 4 = 36 B (alloué en stack, libéré après affichage)

---

## Implémentation `hw_info.c`

### Identification MCU (S1004-01)

Lire les registres ARM et ST standards :

```c
/* ARM Cortex-M : System Control Block */
#define SCB_CPUID   (*(volatile uint32_t *)0xE000ED00)

/* ST Debug MCU Component Register */
#define DBGMCU_IDCODE (*(volatile uint32_t *)0xE0042000)

info->cpuid  = SCB_CPUID;
info->idcode = DBGMCU_IDCODE;
/* IDCODE[11:0] = DEV_ID, [31:16] = REV_ID */
/* STM32F439 attendu : DEV_ID = 0x413, REV_ID = 0x2003 */
```

### Fréquences horloge (S1004-02)

Recalculer depuis les registres RCC (PLL config, prescalers AHB/APB) :

```c
/* RCC_CFGR : SWS[3:2] source, HPRE[7:4] AHB div, PPRE1[12:10], PPRE2[15:13] */
/* RCC_PLLCFGR : PLLM[5:0], PLLN[14:6], PLLP[17:16] */
/* Formule : SYSCLK = HSE_freq / PLLM * PLLN / PLLP */
```

Ne pas utiliser `SystemCoreClock` de CMSIS — le calculer manuellement pour valider
la compréhension de la config PLL (reproductible sur STM32N6 avec registres différents).

### Estimation RAM libre (S1004-03)

```c
/* Symboles linker définis dans STM32F439ZITx_FLASH.ld */
extern uint32_t _sdata, _edata, _sbss, _ebss;
extern uint32_t _estack;  /* Adresse haute SRAM = top stack */

uint32_t used = (uint32_t)(&_edata - &_sdata) * 4   /* .data */
              + (uint32_t)(&_ebss  - &_sbss)  * 4;  /* .bss  */
uint32_t msp  = __get_MSP();
info->stack_free_bytes = msp - (uint32_t)&_sbss;    /* approximation */
info->ram_used_bytes   = used;
info->ram_total_bytes  = 256 * 1024;
```

### DWT cycle counter (S1004-04)

```c
/* CoreDebug + DWT — ARM Cortex-M4 */
#define CoreDebug_DEMCR  (*(volatile uint32_t *)0xE000EDFC)
#define DWT_CTRL         (*(volatile uint32_t *)0xE0001000)
#define DWT_CYCCNT       (*(volatile uint32_t *)0xE0001004)

void dwt_enable(void) {
    CoreDebug_DEMCR |= (1u << 24);  /* TRCENA */
    DWT_CYCCNT = 0;
    DWT_CTRL  |= 1u;                /* CYCCNTENA */
}

uint32_t dwt_cycles(void) { return DWT_CYCCNT; }
```

Calibration : mesurer 1 000 000 itérations de boucle vide, convertir en µs avec SYSCLK.

### Inventaire périphériques (S1004-05)

Lire `RCC->AHB1ENR`, `RCC->APB1ENR`, `RCC->APB2ENR` bit à bit et afficher les
périphériques dont le clock enable est à 1.

---

## Critères d'acceptation

- [x] `hw_info.c` compile sans warning avec `-Wall -Wextra`
- [ ] IDCODE affiché = `0x20036413` (STM32F439ZI — DEV_ID 0x413, REV_ID 0x2003) — en attente accès board
- [ ] SYSCLK affiché = 180 MHz (cohérent avec config PLL de `main.c`) — en attente accès board
- [ ] RAM totale affichée = 256 Ko, RAM utilisée < 2 Ko pour le blink seul — en attente accès board
- [ ] DWT calibration : latence boucle vide ≤ 10 ns/iter @ 180 MHz — en attente accès board
- [ ] Sortie UART lisible sur `minicom -b 115200 -D /dev/ttyACM0` — en attente accès board

---

## Questions ouvertes

- `TODO(dorra)` : Registres équivalents sur STM32N6 (CPUID Cortex-M55, IDCODE N6, DWT Helium) — même structure SCB/DWT ou offsets différents ?
- `TODO(arnaud)` : Faut-il intégrer `hw_info_print` dans le pipeline S1003 comme header de rapport UART, ou garder séparé ?
- `TODO(fred)` : Sur la cible industrielle Edge Spectrum, UART disponible pour debug ou nécessite SWD-only ?

---

## Notes

- Pas de malloc, pas de stdlib — tout en stack ou Flash
- `HWInfo` est déclaré en stack dans `main()` et libéré après `hw_info_print()`
- Le DWT cycle counter est le seul timer de précision sans configuration périphérique supplémentaire — à réutiliser dans tous les benchmarks S1003+
- Annotations MEM obligatoires sur toute struct ajoutée (conforme CLAUDE.md)

**Complété le** : `hw_info.c` implémenté — validation UART sur board en attente
