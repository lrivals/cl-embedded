# S1804 — Firmware auto-profiling : latence DWT, taille .bss, throughput ops/s

| Champ | Valeur |
|-------|--------|
| **ID** | S1804 |
| **Sprint** | Sprint 18 — 25 mai – 1er juin 2026 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 5h |
| **Dépendances** | S1801 (pipeline v2) |
| **Fichiers cibles** | `firmware/stm32f4_blink/src/profiling.c`, `firmware/stm32f4_blink/inc/profiling.h` |
| **Statut** | ✅ Implémenté — à intégrer dans pipeline_run() |

---

## Objectif

Mesurer automatiquement à chaque inférence :
- **Latence** (µs) : temps exact de l'inférence via le compteur de cycles matériel DWT
- **Empreinte RAM** (octets) : taille du segment `.bss` calculée au link time
- **Throughput** (inférences/s) : moyenne glissante sur toute la session

Encoder ces 3 métriques en 8 B et les inclure dans la réponse UART v2.

---

## Structures de données

### `ProfilingState` (`profiling.h`)

```c
/* MEM: 20 B @ FP32/uint32 en .bss */
typedef struct {
    uint32_t t_start_cycles;    /* DWT CYCCNT au début de l'inférence */
    uint32_t last_latency_us;   /* Dernière latence mesurée en µs */
    uint16_t bss_bytes;         /* Taille .bss calculée au link time */
    uint16_t throughput_ips;    /* Inférences par seconde (glissant) */
    uint32_t inference_count;   /* Compteur total d'inférences */
    uint32_t total_cycles;      /* Cycles accumulés sur la session */
} ProfilingState;
```

La structure est instanciée en variable globale statique (`g_profiling`) : **pas de malloc**, allocation en `.bss`.

---

## Mécanisme DWT (Data Watchpoint and Trace)

### Registres ARM Cortex-M4 utilisés

| Registre | Adresse | Rôle |
|----------|---------|------|
| `CoreDebug->DEMCR` | `0xE000EDFC` | Active le debug (bit TRCENA = bit 24) |
| `DWT->CYCCNT` | `0xE0001004` | Compteur de cycles 32 bits, libre-service |
| `DWT->CTRL` | `0xE0001000` | Active le compteur (bit 0 = CYCCNTENA) |

### Initialisation

```c
void profiling_init(void)
{
    /* Active le bloc DWT */
    CoreDebug_DEMCR |= TRCENA_BIT;  /* 0xE000EDFC bit 24 */
    DWT_CYCCNT = 0U;
    DWT_CTRL   |= 1U;               /* CYCCNTENA */

    /* Empreinte .bss : symboles fournis par le linker script */
    g_profiling.bss_bytes = (uint16_t)(
        (uintptr_t)&_ebss - (uintptr_t)&_sbss
    );
}
```

### Mesure de latence

```c
void profiling_start(void)  { g_profiling.t_start_cycles = DWT_CYCCNT; }

void profiling_stop(void)
{
    uint32_t elapsed = DWT_CYCCNT - g_profiling.t_start_cycles;
    g_profiling.last_latency_us = elapsed / (SYSCLK_HZ / 1000000U);
    /* À 180 MHz : 1 cycle = 5.56 ns → diviseur = 180 */

    g_profiling.inference_count++;
    g_profiling.total_cycles += elapsed;

    /* Throughput moyen glissant sur toute la session */
    uint32_t avg = g_profiling.total_cycles / g_profiling.inference_count;
    if (avg > 0U)
        g_profiling.throughput_ips = (uint16_t)(SYSCLK_HZ / avg);
}
```

> **Note** : `SYSCLK_HZ = 180000000U` correspond au STM32F439ZI à sa fréquence max.
> Sur STM32N6 (Cortex-M55 @ 800 MHz), mettre à jour cette constante via `SystemCoreClock`.

### Encodage 8 B pour la réponse UART

```c
/* Encode [latency_us:u32][ram_b:u16][throughput:u16] en little-endian */
void profiling_encode(uint8_t *buf)  // buf doit avoir >= 8 B
```

| Offset | Valeur | Type |
|--------|--------|------|
| 0–3 | `last_latency_us` | u32 LE |
| 4–5 | `bss_bytes` | u16 LE |
| 6–7 | `throughput_ips` | u16 LE |

Correspond exactement aux champs `latency_us`, `ram_used_b`, `throughput` de la réponse v2.

---

## Symboles linker (`profiling.h`)

```c
extern uint32_t _sbss;   /* Début segment .bss */
extern uint32_t _ebss;   /* Fin segment .bss  */
```

Ces symboles sont définis dans le linker script `STM32F439ZITX_FLASH.ld` (généré par STM32CubeMX). La différence `_ebss - _sbss` donne la taille statique de la RAM utilisée par les variables globales et statiques non initialisées.

---

## Intégration dans `pipeline_run()`

```c
void pipeline_run(void)
{
    float raw[MAHA_DIM];
    uart_receive_sample(raw);       /* Réception UART hors mesure */

    profiling_start();              /* ARM le compteur DWT */

    normalize_zscore(raw, MAHA_DIM);
    float score   = maha_score(&g_detector, raw);
    int   anomaly = score > g_detector.threshold;

    if ((g_recv_flags & PROTO_FLAG_UPDATE) && !anomaly)
        maha_update(&g_detector, raw);

    profiling_stop();               /* Capture cycles, calcule throughput */

    float conf = 1.0f / (1.0f + score);
    uart_send_response_v2((uint8_t)anomaly, conf,
                           profiling_get_latency_us(), status);
}
```

La mesure encadre uniquement l'inférence + update (pas la réception UART ni l'émission).

---

## Empreinte mémoire

| Symbole | RAM | Localisation |
|---------|-----|-------------|
| `ProfilingState g_profiling` | 20 B | `.bss` |
| Stack local `pipeline_run()` | 20 B (float raw[5]) | Stack |
| `MahalanobisDetector g_detector` | ~128 B | `.bss` |

---

## Critères d'acceptation

- [ ] `profiling_init()` appelé dans `main()` avant la boucle principale
- [ ] `profiling_start()` / `profiling_stop()` encadrent uniquement l'inférence
- [ ] `profiling_get_latency_us()` retourne > 0 après le premier `profiling_stop()`
- [ ] `profiling_encode(buf)` produit exactement 8 B dans le bon ordre little-endian
- [ ] `bss_bytes` ≤ 52 000 (seuil d'alerte profiling_config.yaml ; contrainte absolue < 65 536 = u16 max)
- [ ] `throughput_ips` > 10 (seuil minimum `profiling_config.yaml`)

---

## Questions ouvertes

- `TODO(dorra)` : Confirmer la valeur de `SystemCoreClock` sur la board réelle (pourrait différer de 180 MHz si PLL mal configurée).
- `TODO(dorra)` : Sur STM32N6, DWT CYCCNT disponible sur Cortex-M55 ? Vérifier TRM.
