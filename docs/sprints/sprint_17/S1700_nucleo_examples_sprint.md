# Sprint 17 — NUCLEO-F439ZI : Exemples existants + Simulation PC

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 17 |
| **Semaine** | 20–27 mai 2026 |
| **Statut** | 🔄 En cours (O1 ✅ O2 ✅ O3 ⬜ O4 ⬜) |
| **Priorité globale** | 🟡 Important — maîtrise board + préparation portage modèles CL |
| **Durée estimée totale** | ~20h |
| **Dépendances** | Sprint 16 terminé — toolchain ARM GCC + OpenOCD opérationnelle, projet blink `firmware/stm32f4_blink/` fonctionnel |

---

## Objectif

Explorer et tester les programmes existants pour NUCLEO-F439ZI : exemples officiels STM32CubeF4, simulation sur PC via Renode, exploration Mbed OS. Ce sprint couvre les **4 objectifs** de prise en main progressive de la board, en s'appuyant sur des exemples fonctionnels pré-existants avant d'écrire du code custom.

**Critère de succès** : les 4 exemples cibles (GPIO, UART, TIM PWM, FreeRTOS) compilent et s'exécutent sur la board réelle OU dans Renode, avec traces UART capturées.

> **Contexte** : Sprint 16 a validé la toolchain avec un blink custom (Makefile direct). Ce sprint exploite les exemples HAL officiels STM32CubeF4 pour couvrir des périphériques supplémentaires et établir les patterns de code utilisés dans les exemples industriels.

---

## Objectif 1 — STM32CubeMX + HAL : LED LD2 et GPIO

### Tâches

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. | Dépendances |
|----|-------|:---:|:---:|---------------------|:---:|-------------|
| S17-01 | Générer projet NUCLEO-F439ZI dans STM32CubeMX (`.ioc` → HAL init → CMakeLists) | 🔴 | ✅ | `firmware/stm32f4_cubemx/` | 2h | S16-01 ✅ |
| S17-02 | LED LD2 (PA5) blink via `HAL_GPIO_TogglePin()` + build CMake + flash | 🔴 | ✅ | `firmware/stm32f4_cubemx/Core/Src/main.c` | 1h | S17-01 |
| S17-03 | Importer exemple `STM32CubeF4/Projects/NUCLEO-F439ZI/Examples/GPIO/GPIO_IOToggle/` + compiler + comparer avec custom | 🟡 | ✅ | `firmware/stm32f4_cubemx_examples/GPIO_IOToggle/` | 1h | S17-01 |
| S17-04 | Doc mini : différences `HAL_GPIO_TogglePin` vs `GPIOx->ODR ^= PIN` (registre direct) | 🟢 | ⬜ | `docs/sprints/sprint_17/S1701_gpio_hal_vs_register.md` | 0.5h | S17-02 |

**Livrable O1** : projet CubeMX fonctionnel, LED LD2 blink via HAL ≤ 2 Ko Flash.

---

## Objectif 2 — UART printf / debug terminal

### Tâches

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. | Dépendances |
|----|-------|:---:|:---:|---------------------|:---:|-------------|
| S17-05 | Configurer USART3 (PD8/PD9 = ST-LINK VCP) dans CubeMX + retarget `printf` via `_write()` | 🔴 | ✅ | `firmware/stm32f4_cubemx/Core/Src/main.c`, `retarget_io.c` | 2h | S17-01 |
| S17-06 | Exemple `UART_Printf` : affichage `"Hello NUCLEO\r\n"` + valeur compteur via terminal série | 🔴 | ✅ | `firmware/stm32f4_cubemx/Core/Src/main.c` | 1h | S17-05 |
| S17-07 | Importer exemple `STM32CubeF4/Examples/UART/UART_Printf/` + valider dans Minicom/PuTTY | 🟡 | ✅ | `firmware/stm32f4_cubemx_examples/UART_Printf/` | 1h | S17-05 |
| S17-08 | Intégrer `printf` dans projet blink existant `firmware/stm32f4_blink/` (compléter pipeline UART S16) | 🟡 | ✅ | `firmware/stm32f4_blink/src/pipeline.c` | 1h | S17-06 |

**Livrable O2** : `printf("score=%.4f latency=%lu us\r\n", score, cycles)` opérationnel — remplace le monitoring binaire existant par du texte lisible.

---

## Objectif 3 — TIM PWM (base pour signaux capteurs)

### Tâches

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. | Dépendances |
|----|-------|:---:|:---:|---------------------|:---:|-------------|
| S17-09 | Configurer TIM3 en mode PWM (PA6 = CH1) dans CubeMX + duty cycle 50% | 🔴 | ⬜ | `firmware/stm32f4_cubemx/Core/Src/main.c` | 2h | S17-01 |
| S17-10 | Importer exemple `STM32CubeF4/Examples/TIM/TIM_PWMOutput/` + mesurer signal à l'oscilloscope ou via loopback | 🟡 | ⬜ | `firmware/stm32f4_cubemx_examples/TIM_PWMOutput/` | 1h | S17-09 |
| S17-11 | Variation duty cycle dynamique (simulation signal vibration synthétique) | 🟢 | ⬜ | `firmware/stm32f4_cubemx/Core/Src/main.c` | 1h | S17-09 |

**Livrable O3** : signal PWM configurable — base pour simuler un capteur vibration en entrée d'un ADC futur.

> **Motivation MCU** : les futurs capteurs de la démo Edge Spectrum enverront des signaux analogiques. TIM + DMA + ADC est le pipeline d'acquisition type.

---

## Objectif 4 — Renode : simulation PC sans hardware

### Tâches

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. | Dépendances |
|----|-------|:---:|:---:|---------------------|:---:|-------------|
| S17-12 | Installer Renode (`apt` ou `.deb` GitHub releases) + vérifier `renode --version` | 🔴 | ⬜ | — | 0.5h | — |
| S17-13 | Script Renode `.resc` pour NUCLEO-F439ZI + charger `firmware/stm32f4_blink/*.elf` | 🔴 | ⬜ | `firmware/renode/nucleo_f439zi.resc` | 2h | S17-12 |
| S17-14 | Valider UART output Mahalanobis dans Renode (sans board réelle) — score + latence DWT | 🔴 | ⬜ | `firmware/renode/run_mahalanobis_sim.sh` | 2h | S17-13 |
| S17-15 | Tester exemple GPIO blink dans Renode (LED toggle observable via log) | 🟡 | ⬜ | `firmware/renode/nucleo_f439zi.resc` | 1h | S17-13 |
| S17-16 | Doc Renode : commandes essentielles + workflow `compile → .elf → Renode → validate` | 🟡 | ⬜ | `docs/sprints/sprint_17/S1704_renode_workflow.md` | 1h | S17-14 |

**Livrable O4** : pipeline `make → .elf → Renode → UART assert` opérationnel — permet de valider le firmware sans accès hardware (utile pour CI et tests futurs STM32N6).

---

## Tâches transverses

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. | Dépendances |
|----|-------|:---:|:---:|---------------------|:---:|-------------|
| S17-17 | Tests unitaires C : ajouter scénario UART mock dans Unity (pipeline.c) | 🟡 | ⬜ | `firmware/stm32f4_blink/tests/test_pipeline.c` | 1h | S17-08 |
| S17-18 | MAJ `.github/workflows/firmware.yml` : ajouter step Renode CI | 🟡 | ⬜ | `.github/workflows/firmware.yml` | 1h | S17-14 |
| S17-19 | MAJ `docs/roadmap_phase2.md` : Sprint 17 ✅ + état S1008/S1009 | 🟢 | ⬜ | `docs/roadmap_phase2.md` | 0.5h | — |

---

## Numérotation expériences

Ce sprint ne génère pas d'expériences Python (pas de modèle CL). Les résultats firmware sont tracés dans `experiments/exp_161/`.

| Exp | Contenu | Fichier |
|-----|---------|---------|
| exp_161 | Profiling GPIO/UART/TIM via DWT — cycles/µs par opération HAL | `experiments/exp_161/hw_timing_hal.json` |

---

## Critères d'acceptation Sprint 17

- [x] S17-02 : LED LD2 blink via CubeMX HAL ≤ 2 Ko Flash, buildé avec `cmake --build`
- [x] S17-06 : `printf` opérationnel sur terminal série Minicom/PuTTY (USART3 @ 115200 baud) — validé hardware 21 mai 2026
- [ ] S17-09 : signal PWM sur PA6, fréquence mesurable
- [ ] S17-14 : Renode exécute `stm32f4_blink.elf` + UART Mahalanobis score visible dans log simulation

---

## Livrable Sprint 17

- **`firmware/stm32f4_cubemx/`** — projet CubeMX complet (GPIO + UART + TIM)
- **`firmware/stm32f4_cubemx_examples/`** — exemples STM32CubeF4 officiels importés
- **`firmware/renode/`** — scripts `.resc` + workflow simulation
- **`experiments/exp_161/hw_timing_hal.json`** — profiling timing HAL
- **`.github/workflows/firmware.yml`** — CI étendue avec step Renode

---

## Questions ouvertes

- `TODO(dorra)` : STM32Cube.AI CLI ≥ 9.x — toujours bloqué ; Renode (S17-12) est le fallback simulation en attendant
- `TODO(arnaud)` : les exemples CubeF4 GPIO/UART/TIM sont-ils suffisants pour la démo encadrants prévue, ou faut-il intégrer un modèle CL dans le projet CubeMX dès Sprint 17 ?
- `TODO(fred)` : format des données capteurs Edge Spectrum — ADC continu ou frames UART ? Influence le choix TIM/DMA (S17-11)

---

## Sous-documents

- [S1701_cubemx_gpio_setup.md](S1701_cubemx_gpio_setup.md) — O1 : CubeMX + GPIO LED LD2 (S17-01 à S17-04)
- [S1702_uart_printf_setup.md](S1702_uart_printf_setup.md) — O2 : UART printf + retarget `_write()` (S17-05 à S17-08)
- [S1703_tim_pwm_setup.md](S1703_tim_pwm_setup.md) — O3 : TIM3 PWM 1 kHz sur PA6 (S17-09 à S17-11)
- [S1704_renode_workflow.md](S1704_renode_workflow.md) — O4 : Simulation PC Renode + CI (S17-12 à S17-16)
- [S1705_tests_ci_roadmap.md](S1705_tests_ci_roadmap.md) — Transverses : tests Unity mock + CI + MAJ roadmap (S17-17 à S17-19)

---

> **Après ce sprint** : Sprint 18 — portage C du modèle EWC one-class dans le projet CubeMX (INT8 update, Gap 3 initial). Ou P2-05 (INT8 backprop) selon disponibilité STM32Cube.AI.
