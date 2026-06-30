# CL-Embedded — Continual Learning for Resource-Constrained Embedded Systems

> M2 Research Internship — ISAE-SUPAERO (DISC) × ENAC (LII) × Edge Spectrum  
> Author: Léonard Rivals | March–August 2026

## Overview

This repository implements and compares continual learning (CL) methods for deployment on microcontrollers with severely limited resources (**active board: NUCLEO-F439ZI**, Cortex-M4 @ 180 MHz, 256 KB SRAM). The application domain is industrial predictive maintenance.

The implementations target **PC-first development** with explicit portability constraints (no dynamic allocation, SGD-only optimizer, ReLU activations, fixed normalization statistics), then are ported to C and validated on hardware. All three scientific gaps have been addressed as of Sprint 22.

> **Status (30 June 2026)**: Sprint 38 implemented. The project now spans the full pipeline — PC continual-learning models, native dataset tasks (RUL regression / multiclass), INT8 + Q15 quantization, multi-model UART protocols (DUAL / PAIR / TRIPLE), stacking meta-model, energy profiling scaffold, paired PC↔board comparison, autonomous gate-triggered EWC updates, and a reproducible sanitized GitLab-release pipeline. See the [Sprint Status](#sprint-status) table.
>
> **Note**: The original target was STM32N6 (Cortex-M55, NPU) but is unavailable. All firmware development targets NUCLEO-F439ZI. Do not design for 64 KB.

## Models

### Supervised CL Methods (board-validated ✅)

| ID | Method | CL Family | Datasets | Board RAM (.bss) |
|----|--------|-----------|----------| -----------------|
| **M2** | EWC Online + MLP | Regularization-based | Monitoring, CWRU, CMAPSS, Paderborn | ~6.7 KB |
| **M2b** | EWC INT8 | Regularization + quantization | CMAPSS, Paderborn (Sprint 22) | ~1.7 KB |
| **M3** | HDC Hyperdimensional | Architecture-based | Monitoring, Pronostia, CWRU | ~14.0 KB |
| **M1** | TinyOL + OtO Head | Architecture-based | Pump, Monitoring | ~5.8 KB |

### Native Dataset Tasks (Sprint 25 — beyond binary CL)

The datasets were originally designed for richer tasks than binary normal/fault. These heads recover that signal:

| Model | Task | Datasets | Metrics |
|-------|------|----------|---------|
| **EWC Regression** (`ewc_mlp_regression.py`) | RUL regression (MSE + EWC) | CMAPSS, Pronostia, Battery | RMSE, MAE, PHM-2008 Horizon Score |
| **EWC Multiclass** (`ewc_mlp_multiclass.py`) | N-class fault classification (softmax + EWC) | CWRU (10 cls), Paderborn (3 cls) | F1-macro, confusion matrix, AF_f1 |
| **HDC Regressor** (`hdc_regressor.py`) | RUL via weighted prototype sum | CMAPSS | RMSE vs EWC |

Board ports (Sprint 26+): RMSE_RUL=21.23 on CMAPSS (ratio 0.94 ✅), EWC multiclass with exact board↔PC parity.

### Ensembles & Stacking (Sprints 30–31)

| Construct | Description | Board mode |
|-----------|-------------|-----------|
| **Model pairs** (Sprint 30) | Mahalanobis + supervised (HDC/EWC/TinyOL) run in parallel, disagreement tracking (κ, origin) | `PAIR_MODE` (22 B response) |
| **Stacking meta-model** (Sprint 31, `src/ensemble/meta_learner.py` + `meta_head.c`) | logreg/MLP arbitrating a pair's two outputs (`[p_maha, p_sup, disagreement, conf_sup]`) | `TRIPLE_MODE` (27 B response, board↔PC parity 1.000) |

### Unsupervised / Anomaly Detection Baselines

| ID | Method | MCU-compatible | Datasets |
|----|--------|:--------------:|---------|
| **Mahalanobis** | Mahalanobis distance | ✅ | All — 80 B model, 3 µs @ 180 MHz |
| **KMeans** | K-Means clustering | ❌ | Monitoring, CWRU, Pronostia |
| **KNN** | KNN anomaly detection | ❌ | Monitoring, CWRU, Pronostia |
| **PCA** | PCA reconstruction error | ❌ | Monitoring, CWRU, Pronostia |
| **DBSCAN** | Density-based clustering | ❌ | Monitoring, CWRU, Pronostia |

### One-Class Anomaly Detection Variants

| Model | MCU-compatible | Notes |
|-------|:--------------:|-------|
| HDC (`one_class_mode`) | ✅ | Percentile threshold on normal data |
| TinyOL Autoencoder | ✅ | MSE reconstruction score |
| EWC OneClass Detector | ✅ | MLP autoencoder + EWC MSE regularization |
| Mahalanobis one-class | ✅ | Mahalanobis distance from normal distribution |
| Mahalanobis Q15 (Sprint 34) | ✅ | int16 Q15 `sigma_inv_` — recovers AUROC vs INT8 on high-dynamic-range covariance |
| KMeans / DBSCAN one-class | ❌ | Distance to centroid / noise points as anomalies |

All one-class models train on normal data only, evaluated with AUROC [T×T] matrices.

## Scientific Positioning — Triple Gap

This work addresses three gaps simultaneously absent from the literature:

| Gap | Claim | Status |
|-----|-------|--------|
| **Gap 1** | Validation on real industrial time-series data | ✅ CWRU, Pronostia, CMAPSS, Paderborn validated (Sprints 18–23) |
| **Gap 2** | CL demonstrated under 100 KB RAM with precise measurements | ✅ Formally proven Sprint 20 — RAM 1 000 B (.bss), latency 3.7 µs P50 / 4.0 µs P99 on NUCLEO-F439ZI |
| **Gap 3** | INT8 quantization during incremental training (not just inference) | ✅ EWC INT8 + HDC INT8 board-validated (Sprint 22); INT8 vs FP32 benchmark (Sprint 28, BOPs gain ×16, RAM ×2.33–4.00); Q15 Mahalanobis fallback for high-dynamic-range covariance (Sprint 34) |

See [`docs/triple_gap.md`](docs/triple_gap.md) for the full analysis.

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# --- Python training ---
python scripts/train_ewc.py --config configs/ewc_config.yaml
python scripts/train_hdc.py --config configs/hdc_config.yaml
python scripts/train_tinyol.py --config configs/tinyol_config.yaml
python scripts/train_mahalanobis.py --config configs/board_mahalanobis.yaml

# CWRU / Pronostia / CMAPSS / Paderborn
python scripts/train_ewc.py --config configs/cwru_by_fault_config.yaml
python scripts/train_hdc.py --config configs/pronostia_config.yaml
python scripts/train_ewc.py --config configs/cmapss_config.yaml       # FP32
python scripts/train_ewc.py --config configs/board_cmapss.yaml        # INT8 (M2b)

# Native dataset tasks (Sprint 25)
python scripts/train_ewc_rul.py --config configs/cmapss_rul_config.yaml          # RUL regression
python scripts/train_ewc_multiclass.py --config configs/cwru_multiclass_config.yaml  # N-class fault

# --- Firmware (NUCLEO-F439ZI) ---
make -C firmware/stm32f4_blink/ all             # Compile for Cortex-M4
make -C firmware/stm32f4_blink/ test            # Run Unity tests on host (TEST_MODE=1)
make -C firmware/stm32f4_blink/ flash           # Flash via OpenOCD ST-LINK
make -C firmware/stm32f4_blink/ size            # Flash/RAM size report

# Export model weights → C header
python scripts/export_weights_c.py --model ewc --config configs/board_ewc.yaml
python scripts/export_weights_tinyol.py --config configs/board_tinyol.yaml

# Board streaming & experiments
python scripts/sensor_stream.py --port /dev/ttyACM0 --dataset cwru
python scripts/sensor_stream.py --port /dev/ttyACM0 --condition all --update   # online CL, native dims
python scripts/board_dataset_builder.py --dry-run
python scripts/board_experiment_recorder.py --exp exp_S23_01

# --- Evaluation ---
python scripts/evaluate_all.py --exp_dir experiments/
python scripts/profile_memory.py --model ewc --dataset monitoring
python scripts/compare_board_sim.py --exp exp_S20_01

# --- GitLab release (sanitized export, Sprint 37) ---
make gitlab-release-dry     # list exclusions / neutral docs, write nothing
make gitlab-check           # CI-style AI-trace guard on a throwaway export
make gitlab-release         # build clean export in a separate repo (hard gate)
```

## Repository Structure

```
cl-embedded/
├── CLAUDE.md                   # Context for Claude Code (read first)
├── README.md                   # This file
├── Makefile                    # gitlab-release / gitlab-release-dry / gitlab-check (Sprint 37)
├── configs/                    # ~102 YAML configs: base / board / dataset / anomaly / single-task /
│                               #   feature-subset / rul / multiclass / sweep / best_features / gitlab_release
├── data/                       # Raw + processed data (gitignored)
├── docs/
│   ├── models/                 # Specs M1–M4 + unsupervised
│   ├── context/                # Hardware, datasets
│   ├── sprints/                # 37 sprint directories with detailed tasks
│   ├── gitlab/                 # Neutral README/CONTRIBUTING templates for the GitLab export
│   ├── triple_gap.md           # Triple-gap scientific positioning
│   ├── roadmap_phase1.md       # Sprints 1–15 (✅ complete)
│   └── roadmap_phase2.md       # Sprints 16–37 + manuscript (active)
├── firmware/
│   └── stm32f4_blink/          # NUCLEO-F439ZI firmware (Cortex-M4)
│       ├── inc/                # 23 headers (ewc_head{,_int8,_multiclass,_regression}, hdc{,_int8},
│       │                       #   mahalanobis{,_q15}, tinyol{,_int8}, meta_head, ring_buffer, pipeline…)
│       ├── src/                # 18 sources (main, pipeline, ewc_head*, hdc*, mahalanobis*, tinyol*,
│       │                       #   meta_head, profiling, metrics…)
│       ├── tests/              # 22 Unity test files (incl. q15, int8, multiclass, meta, pair)
│       ├── startup/            # startup_stm32f439xx.s
│       └── Makefile            # arm-none-eabi-gcc, targets: all / test / flash / size
├── src/
│   ├── data/                   # 8 loaders: pump, monitoring, cwru, pronostia, cmapss, paderborn, battery
│   │                           #   (binary + native rul / multiclass modes, Sprint 25)
│   ├── models/
│   │   ├── ewc/                # ewc_mlp, ewc_mlp_int8, ewc_mlp_regression, ewc_mlp_multiclass,
│   │   │                       #   ewc_oneclass, fisher
│   │   ├── hdc/                # hdc_classifier, hdc_regressor, base_vectors
│   │   ├── tinyol/             # autoencoder, oto_head, tinyol_anomaly_detector
│   │   └── unsupervised/       # KMeans, KNN, PCA, Mahalanobis (+ Q15), DBSCAN detectors
│   ├── ensemble/               # meta_learner.py (stacking, Sprint 31), model pairs / disagreement
│   ├── training/               # CL scenarios, anomaly detection scenarios
│   ├── evaluation/             # metrics, memory_profiler, anomaly_metrics, online_metrics,
│   │                           #   rul_metrics, multiclass_metrics, feature_importance, feature_conditions,
│   │                           #   drift_detector, compute_cost, hw_cost_model, autonomy, plots
│   └── utils/                  # Reproducibility, config loader, quantization helpers
├── experiments/                # 320+ outputs (exp_001–160, exp_S18–exp_S36)
├── notebooks/                  # Exploration + visualization
│   └── cl_eval/                # CL evaluation notebooks (Sprints 7–36: threshold_impact, energy_cost,
│                               #   feature conditions, pc_board_ewc, pairs/disagreement…)
├── tests/                      # ~52 Python unit test files
└── scripts/                    # ~76 CLI entry points (train, eval, export, board drivers,
                                #   sensor_stream, energy_capture, gitlab release, visualize)
```

## Key Design Constraints

All implementations respect MCU portability requirements:

- **SGD optimizer only** (Adam prohibited — memory overhead)
- **ReLU activations only** (INT8-friendly for CMSIS-NN)
- **Fixed normalization statistics** (computed offline, stored in configs/)
- **No dynamic tensor allocation** in forward passes
- **Memory annotations** (`# MEM: X B @ FP32`) on every layer

## Notebooks

| Notebook | Description |
|----------|-------------|
| [`notebooks/01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb) | EDA — Dataset 1 (Pump) + Dataset 2 (Equipment Monitoring) |
| [`notebooks/02_baseline_comparison.ipynb`](notebooks/02_baseline_comparison.ipynb) | EWC vs HDC vs Fine-tuning — Equipment Monitoring |
| [`notebooks/03_cl_evaluation.ipynb`](notebooks/03_cl_evaluation.ipynb) | CL evaluation — Pump Maintenance (TinyOL) |
| [`notebooks/cl_eval/`](notebooks/cl_eval/) | Granular single-task + scenario comparisons (Sprints 7–19) |
| [`notebooks/cl_eval/monitoring_anomaly_detection/`](notebooks/cl_eval/monitoring_anomaly_detection/) | Anomaly detection one-class — AUROC matrices + ROC curves (Sprint 13) |
| [`notebooks/cl_eval/cwru_anomaly_detection/`](notebooks/cl_eval/cwru_anomaly_detection/) | Anomaly detection CWRU — 6 modèles × by_severity (Sprint 17) |
| [`notebooks/cl_eval/equipment_monitoring_anomaly_detection/`](notebooks/cl_eval/equipment_monitoring_anomaly_detection/) | Anomaly detection Equipment Monitoring — by_equipment_type (Sprint 18) |
| [`notebooks/cl_eval/pronostia_anomaly_detection/`](notebooks/cl_eval/pronostia_anomaly_detection/) | Anomaly detection Pronostia — by_bearing_condition (Sprint 19) |
| [`notebooks/cl_eval/summary_anomaly_detection.ipynb`](notebooks/cl_eval/summary_anomaly_detection.ipynb) | Synthèse cross-dataset finale — AUROC × 6 modèles × 3 datasets (Sprint 19) |

## Results

> PC results: seed=42, CPU, RAM = tracemalloc peak (Python overhead included).
> Board results: NUCLEO-F439ZI @ 180 MHz, DWT cycle counter, .bss measured via `arm-none-eabi-size`.

### PC — Equipment Monitoring, by_equipment (Pump → Turbine → Compressor)

| Method | AA | AF | BWT | RAM peak | Latency |
|--------|:--:|:--:|:---:|:--------:|:-------:|
| EWC Online (M2) | **0.9824** | 0.0010 | +0.0000 | 1.1 KB | 0.036 ms |
| HDC (M3) | 0.8698 | **0.0000** | +0.0019 | 14.2 KB | 0.048 ms |
| Mahalanobis | 0.9524 | 0.0010 | −0.0010 | 1.5 KB | **0.018 ms** |
| Fine-tuning naïf | 0.9811 | 0.0000 | +0.0010 | — | — |

### PC — Pump Maintenance, by_id (5 pumps)

| Method | AA | AF | BWT | RAM peak | Latency |
|--------|:--:|:--:|:---:|:--------:|:-------:|
| TinyOL (M1) | 0.5629 | 0.0071 | −0.0030 | 5.8 KB | **0.010 ms** |
| EWC Online (M2) | **0.5658** | 0.0099 | −0.0099 | 1.1 KB | 0.036 ms |
| Fine-tuning naïf | 0.5339 | 0.0595 | −0.0496 | — | — |

> Low AA on Pump is expected — inter-pump distributions are very similar (weak domain shift). See `docs/roadmap_phase1.md`.

### Board — NUCLEO-F439ZI (Sprint 18 baseline, CWRU 3 tasks)

| Metric | Measured | Gap 2 budget | Margin |
|--------|:--------:|:------------:|:------:|
| RAM (.bss) | **1 000 B** | < 64 KB | ×64 |
| Latency P50 | **3.7 µs** | < 100 ms | ×27 000 |
| Latency P99 | **4.0 µs** | < 100 ms | ×25 000 |
| Throughput | **34 235 ips** | — | — |

### Board — advanced features (NUCLEO-F439ZI, measured)

All combined latencies stay far under the 100 ms Gap-2 budget; board↔PC prediction parity is exact for EWC + Mahalanobis (HDC/TinyOL remain HW-only by construction).

| Feature | Sprint | Combined latency | `.bss` | Notes |
|---------|:------:|:----------------:|:------:|-------|
| RUL regression on board (CMAPSS) | 26 | 130 µs inf / 403 µs inf+update | — | RMSE_RUL=21.23 (ratio 0.94 ✅) |
| `DUAL_MODE` (EWC_REG + EWC_MC, 1 frame) | 27 | 637 µs | 66.7 KB | RMSE_RUL=22.59 preserved |
| `PAIR_MODE` Maha+EWC / Maha+HDC | 30 | 256 µs / 651 µs | 104.6 KB | ~0 co-execution overhead, 22 B response |
| `TRIPLE_MODE` (pair + stacking meta) | 31 | 258 µs / 593 µs | 104.6 KB | meta parity board↔PC = 1.000, 27 B response |
| Mahalanobis Q15 | 34 | 5 µs P50 | 105.0 KB | exact parity 300/300, AUROC recovered vs INT8 |
| Feature-count study (`5feat`/`all`/`best`) | 35 | ≤ 1.6 ms (HDC worst) | 105–184 KB | EWC+Maha parity on all dims k=1→21 |
| Paired PC↔board EWC comparison | 36 | 48–65 µs inf / 239–340 µs inf+update | 100–145 KB | frozen parity 1.000, Δacc ≤ 0.007 |
| Autonomous gate-triggered EWC update | 38 | 79–82 µs gated / 238–251 µs always | +300 B gate | ~97 % fewer updates, verdict parity 1.000, F1 preserved |

Full experiment outputs: [`experiments/`](experiments/)

## Progress

| Component | Status | Key results |
|-----------|:------:|-------------|
| M2 EWC + MLP | ✅ | AA=0.982, AF=0.001 — Equipment Monitoring |
| M3 HDC | ✅ | AA=0.870, AF=0.000 (by construction) |
| M1 TinyOL | ✅ | AA=0.563 — Pump (weak domain shift documented) |
| Mahalanobis baseline | ✅ | 80 B params, 3 µs @ 180 MHz — MCU-optimal |
| KMeans / KNN / PCA / DBSCAN | ✅ | Anomaly detection baselines |
| One-class anomaly detection | ✅ | All 5 models, 3 datasets, AUROC matrices |
| CWRU multi-scenario | ✅ | by_fault_type + by_severity (exp_100–160) |
| Pronostia anomaly detection | ✅ | by_bearing_condition, 6 models (exp_S21) |
| CMAPSS + Paderborn CL | ✅ | Domain-incr. FD001–FD004, damage levels (exp_S22) |
| EWC INT8 (M2b) | ✅ | Sprint 22 — Gap 3 closed, board-validated |
| HDC INT8 | ✅ | Sprint 22 — board-validated |
| NUCLEO HAL + Renode CI | ✅ | Sprint 17 — 24/24 Unity PASS, CI operational |
| UART pipeline v3 | ✅ | Sprint 18 — 34 235 ips, 3.7 µs latency |
| 3 models on board (EWC/Mahal/TinyOL) | ✅ | Sprint 19 — exp_S19_01–02 |
| Gap 2 formal proof | ✅ | Sprint 20 — RAM 1 000 B, latency 3.7 µs P50 |
| HDC C on board | ✅ | Sprint 23 O1 — complete |
| CMAPSS/Paderborn board | ✅ | Sprint 23 O2 |
| Retro UINT8 + comparative notebook | ✅ | Sprint 24 — EWC/HDC UINT8, ONNX ×20, RAM report |
| Native tasks (RUL / multiclass) | ✅ | Sprint 25 — EWC regression/multiclass, HDC regressor |
| INT8 vs FP32 benchmark | ✅ | Sprint 28 — BOPs ×16, RAM ×2.33–4.00, Gap 3 RAM 18/18 |
| Model pairs + stacking meta | ✅ | Sprints 30–31 — PAIR_MODE / TRIPLE_MODE, meta parity 1.000 |
| Mahalanobis Q15 | ✅ | Sprint 34 — AUROC recovered vs INT8, exact board parity |
| Threshold-sweep & feature studies | ✅ | Sprints 32 & 35 — RUL→fault threshold, feature-count heatmaps |
| Energy profiling scaffold | ✅ | Sprint 33 — LPM01A pipeline ready, values "à mesurer" |
| Paired PC↔board EWC comparison | ✅ | Sprint 36 — frozen parity 1.000, Δacc ≤ 0.007 |
| GitLab sanitized-release pipeline | ✅ | Sprint 37 — `make gitlab-release`, AI-trace guard |
| Autonomous gate-triggered EWC update | ✅ | Sprint 38 — Mahalanobis + drift gate, ~97 % fewer updates, verdict parity 1.000 |
| Manuscript (P2-07 → P2-10) | ⬜ | July–August 2026 |

### Sprint Status

| Sprint | Status | Focus |
|--------|:------:|-------|
| Sprints 1–15 | ✅ | PC models + anomaly detection + CWRU/Pronostia |
| Sprint 16 | ✅ | Toolchain ARM + Mahalanobis C + UART pipeline |
| Sprint 17 | ✅ | NUCLEO HAL (GPIO/UART/TIM/Renode) — 24/24 tests |
| Sprint 18 | ✅ | UART pipeline v3 + board dataset builder + DWT profiling |
| Sprint 19 | ✅ | 3 models in C on board (Mahalanobis, EWC, TinyOL) |
| Sprint 20 | ✅ | TinyOL weights + EWC fix + Gap 2 formal proof |
| Sprint 21 | ✅ | Multi-dataset board (Monitoring + Pronostia) |
| Sprint 22 | ✅ | CMAPSS + Paderborn CL + Gap 3 INT8 |
| Sprint 23 | ✅ | Board new datasets + HDC C + benchmark |
| Sprint 24 | ✅ | Retro-apply UINT8 (EWC/HDC) + ONNX + comprehensive comparison notebook |
| Sprint 25 | ✅ | Native dataset tasks — RUL regression + multiclass classification (PC) |
| Sprint 26 | ✅ | Board RUL (RMSE=21.23) + catastrophic-forgetting diagnosis (gap1 resolved) |
| Sprint 27 | ✅ | `DUAL_MODE` UART — EWC_REG + EWC_MC in one frame (637 µs) |
| Sprint 28 | ✅ | INT8 vs FP32 benchmark — Gap 3 RAM (BOPs ×16, RAM ×2.33–4.00) |
| Sprint 29 | ⬜ | INT8 firmware on board + Gap 3 multi-model (not started) |
| Sprint 30 | ✅ | Parallel model pairs (Maha + supervised) — `PAIR_MODE` |
| Sprint 31 | ✅ | Stacking meta-model — `TRIPLE_MODE`, parity 1.000 |
| Sprint 32 | ✅ | RUL→fault threshold impact study (60 runs + board sweep) |
| Sprint 33 | ✅ | Energy & cost profiling scaffold (LPM01A; values "à mesurer") |
| Sprint 34 | ✅ | Streaming/ring-buffer + Q15 Mahalanobis (AUROC recovered) |
| Sprint 35 | ✅ | Feature-count impact study (5feat/all/best) — 12 heatmaps |
| Sprint 36 | ✅ | Paired PC↔board EWC comparison (Pronostia + Monitoring) |
| Sprint 37 | ✅ | Sanitized GitLab-release pipeline + AI-trace guard |
| Sprint 38 | ✅ | Autonomous EWC update via embedded novelty gate (4 policies, board-validated) |

## Evaluation Metrics

For every CL experiment:

| Metric | Description |
|--------|-------------|
| `aa` | Average Accuracy across all tasks |
| `af` | Average Forgetting (0 = no forgetting) |
| `bwt` | Backward Transfer (negative = forgetting) |
| `auroc` | Area Under ROC Curve (anomaly detection only) |
| `avg_precision` | Average Precision / AP score (anomaly detection only) |
| `ram_peak_bytes` | Peak RAM measured via tracemalloc |
| `inference_latency_ms` | Forward pass latency (mean over 100 runs) |
| `n_params` | Total trainable parameters |

## Hardware Target

**NUCLEO-F439ZI** (Cortex-M4 @ 180 MHz, 192 KB SRAM + 64 KB CCM = 256 KB total, no NPU)

- Backpropagation runs in FP32 software on Cortex-M4
- Board validated since Sprint 17 (May 2026); all firmware targets this board
- Gap 2 measured result: **RAM 1 000 B (.bss), latency 3.7 µs P50 / 4.0 µs P99**

> The original target was STM32N6 (Cortex-M55, NPU) but is unavailable for this internship. Do not design for 64 KB or assume NPU availability.

See [`docs/context/hardware_constraints.md`](docs/context/hardware_constraints.md) for full details.

## Supervisors

- **Arnaud Dion** — ISAE-SUPAERO (DISC), primary supervisor
- **Dorra Ben Khalifa** — ENAC (LII), hardware & quantization
- **Frédéric Zbierski** — Edge Spectrum, industrial application

## License

MIT License — see `LICENSE` file.

## Citation

```bibtex
@mastersthesis{rivals2026cl_embedded,
  author  = {Léonard Rivals},
  title   = {Apprentissage Incrémental pour Systèmes Embarqués à Ressources Limitées},
  school  = {ISAE-SUPAERO},
  year    = {2026},
  note    = {M2 internship — DISC department}
}
```
