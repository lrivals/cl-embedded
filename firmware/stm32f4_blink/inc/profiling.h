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
extern uint32_t _estack; /* Sommet de la pile (ORIGIN(RAM)+LENGTH(RAM)) */
#endif

extern ProfilingState g_profiling;

/* ── Mesure du pic de RAM (stack high-water mark) ─────────────────────────
 *
 * `.bss` (profiling_get_bss_bytes) ne compte PAS la pile. Le chemin HDC alloue
 * p.ex. float hv[HDC_DIM] = 4 Ko sur la pile : le pic RAM réel vaut donc
 *   .data + .bss + pic_de_pile.
 * Le startup peint [_ebss, _estack) au boot avec une SENTINELLE POSITION-DÉPENDANTE :
 * chaque mot reçoit **sa propre adresse** (canary = &mot). Après exécution d'une
 * charge, le plus bas mot dont la valeur ≠ son adresse marque la profondeur de pile
 * atteinte. Avantage vs constante fixe : un buffer de pile rempli d'une constante
 * répétée ne peut PAS masquer une zone (chaque mot devrait valoir son adresse unique).
 * Caveat résiduel : le seul mot-frontière valant par hasard sa propre adresse
 * sous-estimerait le pic (~1/2³², négligeable). Voir docs/context/ram_measurement.md.
 */

void     profiling_init(void);
void     profiling_start(void);
void     profiling_stop(void);

uint32_t profiling_get_latency_us(void);
uint16_t profiling_get_bss_bytes(void);
uint16_t profiling_get_throughput_ips(void);

/* Fonction pure (testable sur host) : scanne [low, high) de bas en haut et
 * retourne le nombre d'octets entre le premier mot dont la valeur ≠ son adresse
 * (canary position-dépendant) et `high` — c.-à-d. la pile maximale utilisée
 * (la pile croît vers le bas depuis high). */
uint32_t profiling_stack_peak_from_region(const uint32_t *low,
                                          const uint32_t *high);

/* Pic de pile mesuré (octets) via les symboles linker _ebss/_estack. */
uint32_t profiling_stack_peak_bytes(void);

/* Pic de RAM total honnête : .data + .bss + pic_de_pile (octets). */
uint32_t profiling_ram_peak_bytes(void);

/* Encode [latency_us:u32][ram_used_b:u16][throughput:u16] = 8 B dans buf */
void profiling_encode(uint8_t *buf);
#define PROFILING_ENCODED_SIZE 8U

/* ── Marqueurs de phase énergie (S3304) ──────────────────────────────────
 *
 * Toggle GPIO synchronisé avec le DWT pour segmenter le courant capté par le
 * PowerShield LPM01A (S3305) en phases : démarrage / acquisition / inférence /
 * veille. Compilation CONDITIONNELLE : sans -DENERGY_MARKERS, toutes les
 * macros sont vides et le build standard (make all) est strictement inchangé
 * (taille .bss, comportement, flux UART). Aucun print de debug (cf. bug
 * DEBUG_PRINTF du Sprint 18 — ne pas polluer l'UART).
 *
 * Broche : PA8 — DÉDIÉE, distincte de l'UART (PD8/PD9) et de la LED (PA5).
 * GPIOA est déjà cadencé et configuré par pipeline_init() ; PA8 est libre
 * d'après le pinout STM32CubeMX (.ioc : PA5=LD2, PA6=TIM3, PA13/14=SWD).
 * Accessible sur le connecteur NUCLEO pour le branchement de la sonde.
 * Les #define ci-dessous sont modifiables si une autre broche est requise.
 */
#ifdef ENERGY_MARKERS
#define ENERGY_MARKER_PORT   GPIOA
#define ENERGY_MARKER_PIN    8U

#ifndef TEST_HOST
/* Config PA8 en sortie push-pull (GPIOA déjà clocké par pipeline_init). */
#define ENERGY_MARKER_INIT()                                                   \
    do {                                                                       \
        RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN;                                   \
        ENERGY_MARKER_PORT->MODER &= ~(0x3U << (ENERGY_MARKER_PIN * 2U));      \
        ENERGY_MARKER_PORT->MODER |=  (0x1U << (ENERGY_MARKER_PIN * 2U));      \
        ENERGY_MARKER_PORT->OTYPER &= ~(0x1U << ENERGY_MARKER_PIN);            \
    } while (0)
/* Toggle haut : début de phase. */
#define ENERGY_MARKER_SET()                                                    \
    (ENERGY_MARKER_PORT->BSRR = (1U << ENERGY_MARKER_PIN))
/* Toggle bas : fin de phase. */
#define ENERGY_MARKER_CLEAR()                                                  \
    (ENERGY_MARKER_PORT->BSRR = (1U << (ENERGY_MARKER_PIN + 16U)))
#else  /* TEST_HOST : pas de registres matériels */
#define ENERGY_MARKER_INIT()
#define ENERGY_MARKER_SET()
#define ENERGY_MARKER_CLEAR()
#endif /* TEST_HOST */

#else  /* ENERGY_MARKERS absent : build standard inchangé */
#define ENERGY_MARKER_INIT()
#define ENERGY_MARKER_SET()
#define ENERGY_MARKER_CLEAR()
#endif /* ENERGY_MARKERS */

typedef enum {
    PHASE_STARTUP = 0,   /* init système, avant la boucle d'inférence */
    PHASE_ACQUISITION,   /* réception de la trame UART */
    PHASE_INFERENCE,     /* forward (+ update) du/des modèle(s) */
    PHASE_IDLE,          /* retour en attente UART */
} EnergyPhase;

/* Toggle GPIO de phase corrélé au DWT (timestamp via profiling_start au passage
 * en PHASE_INFERENCE). No-op si ENERGY_MARKERS n'est pas défini. */
void energy_marker_phase(EnergyPhase phase);
