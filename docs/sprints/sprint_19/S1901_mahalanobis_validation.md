# S1901 — Validation Mahalanobis C : end-to-end avec streaming S18

| Champ | Valeur |
|-------|--------|
| **Sprint** | 19 |
| **Priorité** | 🔴 Critique |
| **Statut** | ⬜ À faire |
| **Durée estimée** | 4h |
| **Dépendances** | Sprint 18 pipeline données ✅ |
| **Fichiers cibles** | `firmware/stm32f4_blink/src/pipeline.c`, `scripts/sensor_stream.py` |

---

## Contexte

Le détecteur Mahalanobis C a été implémenté en Sprint 16 (`S1603_portage_c_mvp.md`) et testé unitairement (16/16 Unity PASS). Le pipeline UART est fonctionnel avec `sensor_sim.py`. Sprint 18 a livré un `board_dataset_builder.py` qui produit des streams CWRU multi-tâches.

Cette tâche valide le chemin **complet** de bout en bout : données Python → UART → firmware C → réponse → enregistrement — et compare les scores avec la version Python de référence (`src/models/unsupervised/mahalanobis_detector.py`).

---

## Objectif

Obtenir un `experiments/exp_S19_01/results.json` avec `acc_final`, `avg_forgetting`, `backward_transfer`, `ram_peak_bytes`, `inference_latency_ms`, `n_params` via `board_experiment_recorder.py`, ET vérifier que les scores Mahalanobis C s'écartent de moins de **1%** des scores Python sur les mêmes samples.

---

## État actuel (code existant)

### Côté firmware C

**`firmware/stm32f4_blink/src/mahalanobis.c`**
- `maha_init()` — initialisation avec seuil et alpha EMA
- `maha_score()` — distance de Mahalanobis : `sqrt(xᵀ Σ⁻¹ x)` (précision pré-calculée en Flash)
- `maha_update()` — mise à jour EMA de la moyenne (online, sans malloc)
- Budget RAM : ~200 B total (mean: 20 B, precision: 100 B, threshold: 4 B, alpha: 4 B)

**`firmware/stm32f4_blink/src/pipeline.c`**
- `pipeline_init()` — initialise Mahalanobis depuis `model_weights.h` (MAHA_MEAN_INIT, MAHA_PRECISION_INIT, MAHA_THRESHOLD_INIT)
- `pipeline_run()` — boucle : recv frame UART → Z-score → maha_score → LED + réponse 9 B
- Protocole v1 : `[pred:u8][conf:f32][lat_us:u32]` = 9 octets, mesure DWT cycle counter

**`firmware/stm32f4_blink/inc/model_weights.h`**
- Tableaux C statiques (Flash) : ZSCORE_MEAN, ZSCORE_STD, MAHA_MEAN_INIT, MAHA_PRECISION_INIT

### Côté Python

**`scripts/sensor_stream.py`** — streaming multi-tâches via UART, collecte réponses firmware
**`scripts/board_dataset_builder.py`** — construit un stream CWRU segmenté en 3 tâches (normal task 0→1→2)
**`src/models/unsupervised/mahalanobis_detector.py`** — référence Python (même algorithme)

---

## Ce qui manque / Ce qu'il faut faire

### 1. Script de comparaison PC vs C

Créer ou vérifier `scripts/compare_mahalanobis_pc_vs_board.py` :
- Charger les mêmes 500 samples CWRU (3 tâches)
- Calculer scores via `mahalanobis_detector.py` (Python)
- Streamer les mêmes samples via `sensor_stream.py` → firmware → collecter scores C
- Calculer delta moyen : `mean(|score_python - score_C|) < 0.01 * mean(score_python)`

### 2. Validation dry-run

Confirmer que la chaîne suivante produit un JSON valide sans board :
```bash
python scripts/board_experiment_recorder.py \
    --model mahalanobis --dataset cwru \
    --dry-run --output experiments/exp_S19_01
```

### 3. Exécution sur board (si disponible)

```bash
python scripts/board_experiment_recorder.py \
    --model mahalanobis --dataset cwru \
    --port /dev/ttyACM0 --n-samples 500 --n-tasks 3 \
    --output experiments/exp_S19_01
```

---

## Fichiers cibles

| Fichier | Action |
|---------|--------|
| `firmware/stm32f4_blink/src/pipeline.c` | Lecture seule — vérifier protocole v1 |
| `firmware/stm32f4_blink/src/mahalanobis.c` | Lecture seule — MVP validé Sprint 16 |
| `firmware/stm32f4_blink/inc/model_weights.h` | Régénérer si poids CWRU mis à jour |
| `scripts/board_experiment_recorder.py` | Valider dry-run |
| `configs/board_mahalanobis.yaml` | Vérifier seuil, alpha EMA, n_tasks=3 |
| `experiments/exp_S19_01/` | Sortie (créer dossier) |

---

## Budget RAM (indicatif NUCLEO-F439ZI)

| Composant | RAM |
|-----------|-----|
| `MahalanobisDetector` struct (.bss) | ~200 B |
| Stack `pipeline_run` | ~20 B (float raw[5]) |
| **Total** | **~220 B / 192 Ko SRAM** |

`FIXME(gap2)` : à re-mesurer sur Cortex-M55 réel (STM32N6, 64 Ko contrainte projet).

---

## Vérification

- [ ] `make test` dans `firmware/stm32f4_blink/` → `test_mahalanobis.c` : 16/16 PASS
- [ ] Dry-run recorder : JSON présent avec les 6 métriques obligatoires
- [ ] (optionnel) Comparaison PC vs C : delta score < 1% sur 500 samples CWRU

---

## Questions ouvertes

- `TODO(arnaud)` : Accepter un delta PC vs C de 1% ou exiger une tolérance plus stricte (1e-4 comme les tests Unity) ?
- `TODO(dorra)` : Les poids MAHA_PRECISION_INIT sont FP32 en Flash — confirmer que l'ABI Cortex-M55 n'exige pas d'alignement particulier pour les tableaux `const float[]` en Flash XIP ?
