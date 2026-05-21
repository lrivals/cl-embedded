# Sprint 16 — Portage MCU Phase 2 : toolchain + infrastructure C embarquée

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 16 |
| **Semaine** | 20 mai – 17 juin 2026 (exécuté en avance le 11 mai 2026) |
| **Statut** | ✅ CLÔTURÉ — 11 mai 2026 |
| **Priorité globale** | 🔴 Critique — démarrage Phase 2 Portage MCU |
| **Durée estimée totale** | ~44h |
| **Dépendances** | Sprint 15 terminé (Phase 1 anomaly detection Pronostia) ; accès board NUCLEO-F439ZI ; réponse `TODO(dorra)` sur version STM32Cube.AI |

---

## Objectif

Mettre en place la chaîne outillage complète pour le portage des modèles CL Python sur microcontrôleur **NUCLEO-F439ZI** (STM32F439ZI, Cortex-M4 @ 180 MHz, **192 Ko SRAM** @ 0x20000000 + 64 Ko CCM @ 0x10000000). Ce sprint couvre la toolchain ARM, l'export ONNX des modèles Phase 1, le premier portage C MVP, la caractérisation matérielle et l'infrastructure de test C.

**Critère de succès** : pipeline complet `PyTorch → ONNX → code C → flash NUCLEO → validation UART` opérationnel pour Mahalanobis et la tête EWC.

> **Note SRAM** : le linker script initial avait `LENGTH = 256K` (erroné). La NUCLEO-F439ZI expose 192 Ko de SRAM contiguë sur le bus AHB (0x20000000–0x2002FFFF). Les 64 Ko CCM (0x10000000) sont inaccessibles par DMA. Corrigé lors de ce sprint.

---

## Tâches

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Dépendances |
| -- | ----- | :------: | :----: | ------------------- | ----------- |
| S1001 | Setup environnement STM32 : ARM GCC + OpenOCD + VS Code + Cortex-Debug + projet blink | 🔴 | ✅ | `firmware/stm32f4_blink/` | — |
| S1002 | Export ONNX EWC backbone + PTQ INT8 (onnxruntime quantize_dynamic) | 🔴 | ✅ | `scripts/export_onnx.py`, `experiments/exp_160/` | S1001 |
| S1003 | Portage C MVP : Mahalanobis (128 B RAM) + pipeline UART complet | 🔴 | ✅ | `firmware/stm32f4_blink/src/` | S1001, S1002 |
| S1004 | Caractérisation matérielle : CPUID, SYSCLK, RAM libre, DWT cycle counter | 🟡 | ✅ | `firmware/stm32f4_blink/src/hw_info.c` | S1001 |
| S1005 | Setup STM32Cube.AI CLI + validation compatibilité opérateurs ONNX | 🔴 | ⏸ | `scripts/check_onnx_compat.py`, `docs/embedded_ops_compat.md` | TODO(dorra) |
| S1006 | Infrastructure de test C : Unity + CI GitHub Actions | 🟡 | ✅ | `firmware/stm32f4_blink/tests/`, `.github/workflows/firmware.yml` | S1001 |
| S1007 | Simulateur capteur UART + métriques online (board réelle) | 🟡 | ✅ | `scripts/sensor_sim.py`, `src/evaluation/online_metrics.py` | S1001 |
| S1008 | STM32CubeMX projet NUCLEO-F439ZI + CMake build + démo encadrants | 🟡 | 🆕 | `firmware/stm32f4_cubemx/`, `docs/embedded_demo_guide.md` | S1001, S1004 |
| S1009 | LED blink LD2 (PA5) via HAL CubeMX — Hello World embarqué | 🟡 | 🆕 | `firmware/stm32f4_cubemx/Core/Src/main.c` | S1008 |

> Détail : S1601_stm32_env_setup.md · S1602_onnx_export.md · S1603_portage_c_mvp.md · S1604_hardware_characterization.md · S1605_stm32cubeai_setup.md · S1606_c_test_infrastructure.md · S1607_sensor_simulator.md · S1608_stm32cubemx_cmake_demo.md · S1609_led_blink_cubemx.md

---

## Outils installés

| Outil | Version | Chemin | Statut | Usage |
|-------|---------|--------|--------|-------|
| ARM GCC | 13.x | `arm-none-eabi-gcc` | ✅ | Compilation C embarqué |
| OpenOCD | — | `openocd` | ✅ | Flash + debug JTAG/SWD |
| **STM32CubeMX** | **6.17.0** | `~/STM32CubeMX/STM32CubeMX` | **✅ Installé** | Génération HAL, config pins/horloges |
| **CMake** | **4.3.2** | `/usr/local/bin/cmake` | **✅ Installé** | Build system C embarqué (VSCode intégré) |
| STM32Cube.AI | ≥ 9.x requis | `stm32ai` | ⏸ `TODO(dorra)` | Conversion ONNX → code C INT8 |

> ⚠️ **STM32CubeMX ≠ STM32Cube.AI** — CubeMX génère le code d'initialisation HAL (horloges, pins). Cube.AI convertit les réseaux neuronaux en C INT8. Ce sont deux outils distincts.

---

## Architecture MCU cible

```
NUCLEO-F439ZI (board intermédiaire)          STM32N6 (cible finale)
STM32F439ZI — Cortex-M4 @ 180 MHz           Cortex-M55 @ ??? MHz
256 Ko RAM, 2 Mo Flash, pas de NPU           ~64 Ko RAM modèle, NPU inference-only
ST-LINK intégré                               accès hardware TODO(dorra/fred)
```

---

## Pipeline Phase 2

```
STM32CubeMX (.ioc)                           PyTorch .pt
  → firmware/stm32f4_cubemx/ (HAL)            → scripts/export_onnx.py → .onnx
  → CMakeLists.txt                             → STM32Cube.AI (⏸) → code C INT8
  → cmake --build build → .elf
  → openocd flash → NUCLEO-F439ZI
  → scripts/sensor_sim.py → UART frames
  → src/evaluation/online_metrics.py → AUROC streaming
```

---

## Critères d'acceptation Sprint 16

- [x] S1001 : LED LD2 (PA5) clignote, breakpoint `main()` atteignable depuis VS Code ✅ (complété 11 mai 2026)
- [x] S1002 : `ewc_backbone.onnx` passe `onnx.checker`, max|Δ| FP32 < 1e-5 ✅ (code C STM32Cube.AI bloqué TODO(dorra))
- [ ] S1003 : pipeline complet sans hardfault, score Mahalanobis identique Python↔C (tol 1e-4) — implémentation C ✅, test on-board en attente
- [ ] S1004 : IDCODE = `0x20036413`, SYSCLK = 180 MHz affichés via UART — `hw_info.c` compile ✅, test on-board en attente
- [ ] S1005 : `stm32ai --version` ≥ 9.x OU pipeline TFLite validé — `check_onnx_compat.py` ✅, CLI non installé (TODO(dorra))
- [x] S1006 : `make test` passe sur x86, CI GitHub Actions verte ✅ (16/16 tests PASS)
- [x] S1007 : `sensor_sim.py --dry-run` sans erreur, `OnlineAccuracy` == sklearn (tol 1e-9) ✅

---

## Livrable Sprint 16

- **`firmware/stm32f4_blink/`** — projet C complet avec Mahalanobis + tête EWC + pipeline UART
- **`scripts/export_onnx.py`** + **`scripts/sensor_sim.py`**
- **`src/evaluation/online_metrics.py`**
- **`scripts/check_onnx_compat.py`** + **`docs/embedded_ops_compat.md`**
- **`firmware/stm32f4_blink/tests/`** + **`.github/workflows/firmware.yml`**

---

## Questions ouvertes

- `TODO(dorra)` : version STM32Cube.AI minimale pour STM32N6 / NeuralART Turbo — bloquant pour S1005
- `TODO(dorra)` : planning accès STM32N6 réel — à partir de quel sprint ?
- `TODO(fred)` : Edge Spectrum a-t-il une board STM32N6 pour les tests de P2-06 ?
- `TODO(arnaud)` : évaluations intermédiaires prévues sur NUCLEO-F439ZI ou bascule directe STM32N6 ?

---

> **Après ce sprint** : mettre à jour `docs/roadmap_phase2.md` (S16 ✅, Phase 2 toolchain opérationnelle). Commencer Sprint 17 (CWRU anomaly detection) en parallèle si délai sur accès STM32N6.
