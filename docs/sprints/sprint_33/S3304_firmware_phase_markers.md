# S3304 — Marqueurs de phase GPIO dans `pipeline.c`

| Champ | Valeur |
|-------|--------|
| **Sprint** | 33 |
| **Priorité** | 🔴 Critique — bloquant pour S3305 (energy_capture.py segmente sur ces marqueurs) |
| **Statut** | ✅ Implémenté (23 juin 2026) |
| **Durée estimée** | 3h |
| **Dépendances** | `firmware/stm32f4_blink/src/profiling.c` ✅ (DWT cycle-counter) |
| **Fichiers cibles** | `firmware/stm32f4_blink/src/pipeline.c`, `firmware/stm32f4_blink/inc/profiling.h` |
| **Références** | `profiling.c:35` (`profiling_init`), `:52` (`profiling_start`), `:57` (`profiling_stop`), `pipeline.c:12-30` (includes), `:36-74` (instances modèles globales) |

---

## Contexte

`profiling.c`/`.h` mesure déjà la latence via DWT (`profiling_start/stop()`, ~20 B `.bss`),
mais **aucun mécanisme GPIO ou marqueur de phase n'existe** (confirmé par lecture du code).
Pour segmenter le courant capté par le PowerShield LPM01A en phases (démarrage / acquisition
/ inférence / veille), il faut un toggle GPIO synchronisé avec le DWT, sans perturber l'UART
(le bug `DEBUG_PRINTF` du Sprint 18 avait pollué le flux UART avec des prints de debug — ne
pas répéter l'erreur).

---

## Spec header (extension `profiling.h`)

```c
#ifdef ENERGY_MARKERS
#define ENERGY_MARKER_PORT   GPIOx          /* broche dédiée, PAS la même que l'UART/LED debug */
#define ENERGY_MARKER_PIN    <N>
#define ENERGY_MARKER_INIT()      /* config GPIO sortie push-pull */
#define ENERGY_MARKER_SET()       /* toggle haut : début de phase */
#define ENERGY_MARKER_CLEAR()     /* toggle bas : fin de phase */
#else
#define ENERGY_MARKER_INIT()
#define ENERGY_MARKER_SET()
#define ENERGY_MARKER_CLEAR()
#endif

typedef enum {
    PHASE_STARTUP = 0,
    PHASE_ACQUISITION,
    PHASE_INFERENCE,
    PHASE_IDLE,
} EnergyPhase;

void energy_marker_phase(EnergyPhase phase);  /* toggle GPIO + profiling_start() corrélé DWT */
```

**Règles** :
- Compilation **conditionnelle** `#ifdef ENERGY_MARKERS` — le build standard
  (`make all`) ne doit ni grossir ni changer de comportement quand le flag est absent.
- Broche **dédiée**, distincte de toute broche déjà utilisée par l'UART ou la LED de debug
  — vérifier le pinout `STM32CubeMX` (`.ioc`) avant de choisir la broche.
- Réutiliser le DWT existant (`profiling_start()`/`profiling_stop()`) pour corréler
  temps↔énergie : chaque toggle GPIO doit avoir un timestamp DWT associé, exploité ensuite
  par `energy_capture.py` (S3305) pour aligner les segments de courant aux phases.
- Insertion dans `pipeline.c` aux points de transition naturels du dispatch existant
  (`pipeline_run()`) : début de trame reçue (acquisition), avant le forward (inférence),
  retour à l'attente UART (veille).

---

## Vérification

```bash
cd firmware/stm32f4_blink
make all CFLAGS+=-DENERGY_MARKERS    # build avec marqueurs actifs, 0 erreur
make all                              # build standard inchangé (pas de régression .bss)
make test                             # 0 nouvelle régression Unity
arm-none-eabi-size build/stm32f4_blink.elf
```
