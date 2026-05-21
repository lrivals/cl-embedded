# Sprint 19 — Adaptation et déploiement des modèles Phase 1 sur carte

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 19 |
| **Semaine** | 1–8 juin 2026 |
| **Statut** | ⬜ À faire |
| **Priorité globale** | 🔴 Critique — validation modèles CL sur MCU |
| **Durée estimée totale** | ~42h |
| **Dépendances** | Sprint 16 (Mahalanobis C ✅, EWC head C esquissé ✅), Sprint 18 (pipeline données ✅) |

---

## Objectif

Valider les 3 modèles Phase 1 en C sur NUCLEO-F439ZI, avec évaluation automatique enregistrée dans `experiments/` au **même format que Phase 1** (Python) :

```
PyTorch weights  →  export_weights_c.py  →  model_weights.h (Flash)
                                                    ↓
mock_data.h (tests host)  →  firmware C models  →  board UART
                                                    ↓
board_experiment_recorder.py  →  experiments/exp_S19_XX/results.json
(acc_final, AF, BWT, ram_peak_bytes, inference_latency_ms, n_params)
```

**Critère de succès** : `make test` passe 100% des tests Unity sur host (sans board), ET `python scripts/board_experiment_recorder.py --dry-run` produit un `results.json` avec les 6 métriques obligatoires.

---

## Tâches

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Dépendances |
|----|-------|:--------:|:------:|--------------------|-------------|
| S1901 | Validation Mahalanobis C : end-to-end avec streaming S18, comparaison PC vs C | 🔴 | ⬜ | `firmware/stm32f4_blink/src/mahalanobis.c` | S18 done |
| S1902 | Compléter EWC head C : Fisher EMA update, `ewc_consolidate()`, annotations MEM | 🔴 | ⬜ | `firmware/stm32f4_blink/src/ewc_head.c` (compléter) | S1901 |
| S1903 | TinyOL encoder C skeleton : forward pass seulement (NPU-simulated), poids Flash | 🔴 | ⬜ | `firmware/stm32f4_blink/src/tinyol.c`, `inc/tinyol.h` | S1901 |
| S1904 | Mock data framework C : samples synthétiques en dur pour tests host sans board | 🔴 | ⬜ | `firmware/stm32f4_blink/tests/mock_data.h` | — |
| S1905 | Firmware metrics : accuracy online, AUROC sliding window, forgetting tracker | 🔴 | ⬜ | `firmware/stm32f4_blink/src/metrics.c`, `inc/metrics.h` | S1904 |
| S1906 | Response protocol v3 : ajoute metrics_snapshot (acc, auroc, forgetting) | 🔴 | ⬜ | `firmware/stm32f4_blink/src/pipeline.c` | S1905 |
| S1907 | Experiment recorder Python : capture résultats board → experiments/ unifié Phase 1 | 🔴 | ⬜ | `scripts/board_experiment_recorder.py` | S1906 |
| S1908 | Configs YAML modèles embarqués : dims, seuils, Fisher decay, LR | 🟡 | ⬜ | `configs/board_mahalanobis.yaml`, `configs/board_ewc.yaml`, `configs/board_tinyol.yaml` | S1902–S1903 |
| S1909 | Tests Unity : tous modèles sur mock_data, vérification pas de malloc | 🟡 | ⬜ | `firmware/stm32f4_blink/tests/test_models.c` | S1902–S1904 |
| S1910 | Tests Python recorder : JSON output, champs obligatoires, format unifié | 🟡 | ⬜ | `tests/test_board_recorder.py` | S1907 |
| S1911 | Expérience E19-01 : Mahalanobis 500 samples CWRU, auto-enregistré, PC vs carte | 🟡 | ⬜ | `experiments/exp_S19_01/` | S1907–S1908 |
| S1912 | Expérience E19-02 : EWC head 3 tâches Monitoring, forgetting mesuré on-board | 🟡 | ⬜ | `experiments/exp_S19_02/` | S1902, S1911 |
| S1913 | RAM profiling statique : `-Wl,-Map` + parser map file → valide < 64 Ko | 🟢 | ⬜ | `scripts/parse_map_file.py` | S1902–S1903 |

> Détail : [S1901](S1901_mahalanobis_validation.md) · [S1902](S1902_ewc_head_complete.md) · [S1903](S1903_tinyol_skeleton.md) · [S1904](S1904_mock_data.md) · [S1905](S1905_firmware_metrics.md) · [S1906](S1906_protocol_v3.md) · [S1907](S1907_experiment_recorder.md) · [S1908](S1908_board_configs.md) · [S1909](S1909_unity_model_tests.md) · [S1910](S1910_tests_recorder.md) · [S1911](S1911_exp_mahalanobis.md) · [S1912](S1912_exp_ewc.md) · [S1913](S1913_ram_profiling_static.md)

---

## Budget RAM récapitulatif (NUCLEO / STM32N6)

| Modèle | Poids (Flash) | Activation (stack) | CL overhead | Total SRAM | Marge / 64 Ko |
|--------|:------------:|:------------------:|:-----------:|:----------:|:-------------:|
| Mahalanobis | 128 B | 40 B | 0 | **~200 B** | ✅ 63.8 Ko free |
| EWC head MLP | 3 Ko Flash | 200 B stack | 6 Ko Fisher | **~9.5 Ko** | ✅ 54.5 Ko free |
| TinyOL encoder | 6 Ko Flash | 512 B stack | 40 B OtO | **~7 Ko** | ✅ 57 Ko free |

> **Note** : mesures à valider sur Cortex-M55 réel — NUCLEO-F439ZI (192 Ko SRAM) est indicatif (`FIXME(gap2)`).

---

## Format résultats experiments/ (unifié Phase 1)

```json
{
  "exp_id": "S19_01",
  "model": "mahalanobis",
  "dataset": "cwru",
  "platform": "nucleo_f439zi",
  "date": "2026-06-02",
  "acc_final": 0.94,
  "avg_forgetting": 0.02,
  "backward_transfer": -0.01,
  "ram_peak_bytes": 210,
  "inference_latency_ms": 0.003,
  "n_params": 30,
  "n_tasks": 3,
  "n_samples_total": 500,
  "config_snapshot": "configs/board_mahalanobis.yaml"
}
```

---

## Questions ouvertes

- `TODO(arnaud)` : Priorité TinyOL skeleton (M1) vs INT8 backprop exploration (Gap 3) dans ce sprint ?
- `TODO(dorra)` : Format poids exportés `model_weights.h` — FP32 array C statique ou struct nommée ?
- `TODO(dorra)` : NeuralART Turbo (NPU STM32N6) accepte ONNX opset 17 ou format propriétaire `.nef` ?
- `FIXME(gap2)` : Validation RAM < 64 Ko requise sur Cortex-M55 réel — NUCLEO indicatif seulement
