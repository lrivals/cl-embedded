# Sprint 18 — Données capteurs sur carte : streaming, dataset builder, auto-profiling

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 18 |
| **Semaine** | 25 mai – 1er juin 2026 |
| **Statut** | ⬜ À faire |
| **Priorité globale** | 🔴 Critique — pipeline données bout-en-bout |
| **Durée estimée totale** | ~38h |
| **Dépendances** | Sprint 16 (toolchain + UART MVP ✅), Sprint 17 (NUCLEO périphériques ✅/en cours) |

---

## Objectif

Construire le pipeline complet **PC → carte → PC** :

```
PC streame un dataset Phase 1 (CWRU / Monitoring)
    ↓  via UART (protocole binaire étendu)
MCU traite chaque sample (inférence + update)
    ↓  renvoie réponse étendue (pred + conf + latence + RAM)
PC collecte → dataset CSV / HDF5 + profiling JSON
    ↓
experiments/exp_S18_XX/ (format unifié Phase 1)
```

**Critère de succès** : `python scripts/board_dataset_builder.py --dry-run` produit un `experiments/exp_S18_01/results.json` et un `dataset.csv` valides en moins de 5 minutes, sans board.

---

## Tâches

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Dépendances |
|----|-------|:--------:|:------:|--------------------|-------------|
| S1801 | Protocole UART étendu : timestamps, séquences continues, champ task_id | 🔴 | ⬜ | `firmware/stm32f4_blink/src/pipeline.c`, `inc/pipeline.h` | S17 done |
| S1802 | Script PC streaming continu : rate-limité, séquences multi-tâches, dry-run | 🔴 | ⬜ | `scripts/sensor_stream.py` | S1801 |
| S1803 | Dataset builder : collecte réponses → CSV/HDF5, métadonnées exp auto | 🔴 | ⬜ | `scripts/board_dataset_builder.py` | S1802 |
| S1804 | Firmware auto-profiling : latence DWT, taille .bss, throughput ops/s | 🔴 | ⬜ | `firmware/stm32f4_blink/src/profiling.c`, `inc/profiling.h` | S1801 |
| S1805 | Host profiling reader : parse métriques UART, sauve JSON experiments/ auto | 🔴 | ⬜ | `scripts/profiling_reader.py` | S1804 |
| S1806 | Config YAML profiling : seuils RAM, latence max, format sortie | 🟡 | ⬜ | `configs/profiling_config.yaml` | S1804 |
| S1807 | Tests Unity firmware : DWT latence, CRC frames, buffer overflow guards | 🟡 | ⬜ | `firmware/stm32f4_blink/tests/test_profiling.c` | S1804 |
| S1808 | Tests Python : streaming dry-run, mock serial, format CSV/JSON output | 🟡 | ⬜ | `tests/test_sensor_stream.py`, `tests/test_dataset_builder.py` | S1802–S1803 |
| S1809 | Expérience E18-01 : stream 500 samples CWRU → CSV + profiling JSON auto | 🟡 | ⬜ | `experiments/exp_S18_01/` | S1801–S1806 |
| S1810 | Expérience E18-02 : stream Monitoring, compare latence Mahalanobis vs dummy | 🟢 | ⬜ | `experiments/exp_S18_02/` | S1809 |
| S1811 | Documentation protocole binaire + guide "première session carte" | 🟢 | ⬜ | `docs/sprints/sprint_18/S1811_protocol_guide.md` | S1801 |

> Détail : [S1801](S1801_uart_protocol_extended.md) · [S1802](S1802_sensor_stream.md) · [S1803](S1803_dataset_builder.md) · [S1804](S1804_firmware_profiling.md) · [S1805](S1805_profiling_reader.md) · [S1806](S1806_profiling_config.md) · [S1807](S1807_tests_profiling.md) · [S1808](S1808_tests_python.md) · [S1809](S1809_exp_cwru_stream.md) · [S1810](S1810_exp_monitoring_stream.md) · [S1811](S1811_protocol_guide.md)

---

## Protocole UART étendu (Sprint 18)

### Trame envoyée PC → MCU (protocole v2)

```c
[MAGIC 0xABCD : 2B]
[VERSION : 1B]       // 0x02 pour protocole v2
[TASK_ID : 1B]       // identifiant de tâche CL (0–7)
[TIMESTAMP_MS : 4B]  // uint32, ms depuis début session
[N_FEATURES : 1B]    // nombre de features (≤ 16)
[features : f32×N]   // features normalisées
[label : 1B]         // ground truth
[FLAGS : 1B]         // bit0=update_requested, bit1=profiling_requested
[CRC8 : 1B]          // polynomial 0x07
```

### Réponse MCU → PC (protocole v2)

```c
[pred_label : 1B]    // prédiction
[confidence : f32]   // score de confiance
[latency_us : u32]   // latence DWT en µs
[ram_used_b : u16]   // octets .bss + stack peak estimé
[throughput : u16]   // inférences par seconde (entier)
[status : 1B]        // 0=OK, bit0=CRC_ERR, bit1=OOB, bit2=UPDATE_DONE
```

Total réponse : **14 B** (vs 9 B protocole v1)

---

## Livrable

- Pipeline `PC streams data → MCU traite → profiling auto → dataset + résultats PC`
- `scripts/sensor_stream.py` opérationnel en dry-run sans board
- `scripts/board_dataset_builder.py` produit des CSV + JSON compatibles `evaluate_all.py`
- `firmware/stm32f4_blink/src/profiling.c` mesurant latence, RAM, throughput

---

## Questions ouvertes

- `TODO(dorra)` : NUCLEO-F439ZI disponible pour tests série complète ? ou attendre STM32N6 ?
- `TODO(dorra)` : Format ADC STM32N6 (12-bit, DMA, fréquence échantillonnage max) pour calibrer le protocole maintenant
- `TODO(fred)` : Format données capteurs réels Edge Spectrum (CSV export / format propriétaire ?)
- `TODO(arnaud)` : Fréquence d'échantillonnage cible pour le streaming continu (10 Hz / 100 Hz ?)
