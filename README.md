# CL-Embedded — Continual Learning for Resource-Constrained Embedded Systems

> M2 Research Internship — ISAE-SUPAERO (DISC) × ENAC (LII) × Edge Spectrum  
> Author: Léonard Rivals | March–August 2026

## Overview

This repository implements and compares continual learning (CL) methods designed for deployment on microcontrollers with severely limited resources (target: STM32N6, ~64 KB RAM). The application domain is industrial predictive maintenance.

The implementations target **PC-first development** with explicit design constraints ensuring portability to the MCU target (no dynamic allocation, SGD-only optimizer, ReLU activations, fixed normalization statistics).

## Models

### Supervised CL Methods

| ID | Method | CL Family | Dataset | RAM (update) |
|----|--------|-----------|---------|--------------|
| **M2** | EWC Online + MLP | Regularization-based | Equipment Monitoring + Pump | ~6.7 KB |
| **M3** | Hyperdimensional Computing (HDC) | Architecture-based | Equipment Monitoring + Pump | ~14.0 KB |
| **M1** | TinyOL + OtO Head | Architecture-based | Pump Maintenance | ~5.8 KB |

### Unsupervised Baselines

| ID | Method | MCU-compatible | Dataset | RAM (params) |
|----|--------|:--------------:|---------|--------------|
| **M6** | Mahalanobis distance | ✅ | Both | 80 B |
| **M4** | K-Means clustering | ❌ | Both | ~4 KB |
| **M4b** | KNN anomaly detection | ❌ | Both | ~4 KB |
| **M5** | PCA reconstruction error | ❌ | Both | ~4 KB |
| — | DBSCAN | ❌ | Both | — |

### One-Class Anomaly Detection (Sprints 13–19)

| Model | MCU-compatible | Config | Notes |
|-------|:--------------:|--------|-------|
| HDC (one_class_mode) | ✅ | `hdc_anomaly_detection_config.yaml` | Seuil au percentile sur données normales |
| TinyOL Autoencoder | ✅ | `tinyol_anomaly_detection_config.yaml` | Reconstruction MSE comme score d'anomalie |
| EWC OneClass Detector | ✅ | `ewc_oneclass_config.yaml` | Autoencoder MLP + régularisation EWC sur MSE reconstruction |
| K-Means one-class | ❌ | `unsupervised_anomaly_detection_config.yaml` | Distance au centroïde le plus proche |
| Mahalanobis one-class | ✅ | `unsupervised_anomaly_detection_config.yaml` | Distance de Mahalanobis depuis données normales |
| DBSCAN one-class | ❌ | `unsupervised_anomaly_detection_config.yaml` | Points bruit = anomalies (RAM variable selon volume normal) |

Tous les modèles s'entraînent sur données normales uniquement et évaluent via matrice AUROC [T×T].

All models operate within the 64 KB RAM budget of the STM32N6 target (M6/HDC are the only methods viable on MCU).

## Scientific Positioning — Triple Gap

This work addresses three gaps simultaneously absent from the literature:

- **Gap 1**: Validation on real industrial time-series data
- **Gap 2**: CL demonstrated under 100 KB RAM with precise per-component measurements
- **Gap 3**: INT8 quantization during incremental training (not just inference)

See [`docs/context/triple_gap.md`](docs/context/triple_gap.md) for the full analysis.

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Train M2 (EWC) — Equipment Monitoring
python scripts/train_ewc.py --config configs/ewc_config.yaml

# Generate HDC base vectors (once, before training HDC)
python -m src.models.hdc.base_vectors

# Train M3 (HDC) — Equipment Monitoring
python scripts/train_hdc.py --config configs/hdc_config.yaml

# Pre-train TinyOL backbone (Dataset 1 — normal samples only)
python scripts/pretrain_tinyol.py --config configs/tinyol_config.yaml

# Train M1 (TinyOL) — Pump Maintenance
python scripts/train_tinyol.py --config configs/tinyol_config.yaml

# Train unsupervised baselines (K-Means, Mahalanobis, DBSCAN, PCA, KNN)
python scripts/train_unsupervised.py --config configs/unsupervised_config.yaml

# Anomaly detection — one-class CL (Sprints 13–19)
python scripts/train_hdc.py --config configs/hdc_anomaly_detection_config.yaml
python scripts/train_tinyol.py --config configs/tinyol_anomaly_detection_config.yaml
python scripts/train_unsupervised.py --config configs/unsupervised_anomaly_detection_config.yaml
python scripts/run_anomaly_detection.py --config configs/ewc_oneclass_config.yaml  # Sprint 14

# Run anomaly detection batch (Monitoring / Pronostia)
python scripts/run_anomaly_detection.py --model hdc --dataset pronostia --scenario by_bearing_condition --strategy refit --exp_id exp_155

# Firmware (Phase 2 — NUCLEO-F439ZI)
make -C firmware/stm32f4_blink/ test             # Unity tests host x86 (16/16 PASS)
make -C firmware/stm32f4_blink/ flash            # Flash NUCLEO via OpenOCD
python scripts/sensor_stream.py --dry-run        # Simulate UART sensor streaming (Sprint 18)
python scripts/board_experiment_recorder.py --dry-run  # Record on-board results (Sprint 19)

# Run all evaluations
python scripts/evaluate_all.py --exp_dir experiments/

# Memory profiling
python scripts/profile_memory.py --model ewc
```

## Repository Structure

```
cl-embedded/
├── CLAUDE.md                   # Context for Claude Code (read first)
├── README.md                   # This file
├── configs/                    # YAML hyperparameters (versioned)
│   ├── ewc_config.yaml         # M2 — Equipment Monitoring
│   ├── ewc_pump_config.yaml    # M2 — Pump Maintenance
│   ├── hdc_config.yaml         # M3 — Equipment Monitoring
│   ├── hdc_pump_config.yaml    # M3 — Pump Maintenance
│   ├── tinyol_config.yaml      # M1 — Pump Maintenance
│   ├── tinyol_monitoring_config.yaml
│   ├── unsupervised_config.yaml
│   ├── pump_by_id_config.yaml
│   ├── pump_by_temporal_window_config.yaml
│   ├── monitoring_by_location_config.yaml
│   ├── hdc_anomaly_detection_config.yaml       # Sprint 13
│   ├── tinyol_anomaly_detection_config.yaml    # Sprint 13
│   ├── unsupervised_anomaly_detection_config.yaml  # Sprint 13
│   ├── ewc_oneclass_config.yaml                # Sprint 14 — EWC OneClass Detector
│   ├── pump_normalizer.yaml    # Fixed Z-score statistics
│   ├── monitoring_normalizer.yaml
│   └── monitoring_normalizer_anomaly.yaml      # Fitted on normal samples only
├── data/                       # Raw + processed data (gitignored)
├── docs/
│   ├── models/                 # Detailed implementation specs (3 models + unsupervised)
│   ├── context/                # Project context, hardware, datasets
│   ├── sprints/                # Sprint-level task tracking
│   ├── roadmap.md              # Master roadmap index
│   ├── roadmap_phase1.md       # Phase 1 development roadmap (Sprints 1–9)
│   └── roadmap_phase2.md       # Phase 2 MCU portage + Sprints 17–19
├── skills/                     # Claude prompting guides
├── src/
│   ├── data/                   # Dataset loaders (pump + monitoring)
│   ├── models/
│   │   ├── ewc/                # M2 — EWC Online + MLP
│   │   ├── hdc/                # M3 — HDC (supervisé + one_class_mode)
│   │   ├── tinyol/             # M1 — TinyOL OtO head + TinyOLAnomalyDetector
│   │   └── unsupervised/       # K-Means, KNN, PCA, Mahalanobis, DBSCAN
│   ├── training/               # CL scenarios + run_anomaly_detection_scenario()
│   ├── evaluation/             # CL metrics + anomaly_metrics + memory profiler
│   └── utils/                  # Reproducibility, config loader
├── experiments/                # 160+ reproducible experiment outputs (exp_S19 series: on-board)
├── notebooks/                  # Exploration + visualization
│   └── cl_eval/                # Granular CL evaluation notebooks (Sprints 7–19)
├── tests/                      # Unit tests
└── scripts/                    # CLI entry points
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

> seed=42, CPU. RAM = tracemalloc peak (includes Python overhead; not representative of MCU deployment).

### Dataset 2 — Equipment Monitoring, by_equipment (3 domains: Pump → Turbine → Compressor)

| Method | AA | AF | BWT | RAM peak | Latency |
|--------|:--:|:--:|:---:|:--------:|:-------:|
| EWC Online (M2) | **0.9824** | 0.0010 | +0.0000 | 1.1 KB | 0.036 ms |
| HDC (M3) | 0.8698 | **0.0000** | +0.0019 | 14.2 KB | 0.048 ms |
| Mahalanobis (M6) | 0.9524 | 0.0010 | −0.0010 | 1.5 KB | **0.018 ms** |
| Fine-tuning naïf | 0.9811 | 0.0000 | +0.0010 | — | — |

> EWC achieves best accuracy with minimal forgetting. Mahalanobis is optimal for MCU deployment (80 B model weights, no backprop). HDC: AF = 0 by construction.

### Dataset 1 — Pump Maintenance, by_id (5 pumps)

| Method | AA | AF | BWT | RAM peak | Latency |
|--------|:--:|:--:|:---:|:--------:|:-------:|
| TinyOL (M1) | 0.5629 | 0.0071 | −0.0030 | 5.8 KB | **0.010 ms** |
| EWC Online (M2) | **0.5658** | **0.0099** | −0.0099 | 1.1 KB | 0.036 ms |
| Fine-tuning naïf | 0.5339 | 0.0595 | −0.0496 | — | — |

> Low AA on Dataset 1 is documented — inter-pump distributions are very similar (weak domain shift). See `docs/roadmap_phase1.md` for analysis.

Full experiment outputs: [`experiments/`](experiments/)

## Progress

| Component | Implemented | Tested | Experiments | RAM measured |
|-----------|:-----------:|:------:|:-----------:|:------------:|
| M2 EWC + MLP | ✅ | ✅ | ✅ exp_001, 013, 016, 025, 030, 036 | ✅ |
| M3 HDC | ✅ | ✅ | ✅ exp_002, 014, 017, 026, 031, 037 | ✅ |
| M1 TinyOL | ✅ | ✅ | ✅ exp_003, 011, 012, 018, 024, 032, 038 | ✅ |
| M6 Mahalanobis | ✅ | ✅ | ✅ exp_007, 015, 019, 027, 034, 040 | ✅ |
| M4 K-Means + KNN | ✅ | ✅ | ✅ exp_005, 020, 022, 028, 033, 039 | ✅ |
| M5 PCA | ✅ | ✅ | ✅ | ✅ |
| DBSCAN baseline | ✅ | ✅ | ✅ exp_008, 021, 023, 029, 035, 041 | ✅ |
| Anomaly detection (S13) | ✅ | ✅ | ✅ exp_086–093, exp_120–122 | ⬜ |
| EWC OneClass (S14) | ✅ | ✅ | ✅ exp_123–136 | ⬜ |
| M1 + UINT8 buffer | ⬜ | ⬜ | ⬜ exp_004 (planned) | ⬜ |
| ONNX export (S16) | ✅ | ⬜ | ✅ exp_160 (FP32→INT8 Δ AUROC≈0) | ✅ |
| Mahalanobis C firmware (S16) | ✅ | ✅ | ✅ 3 µs @ 180 MHz, 128 B SRAM | ✅ |
| UART pipeline + sensor sim (S16) | ✅ | ✅ | ✅ 10/10 trames, 0 CRC errors | — |
| NUCLEO périphériques (S17) | 🔄 | 🔄 | — O1 ✅ O2 ✅ O3–O4 ⬜ | — |
| Streaming pipeline capteurs (S18) | ⬜ | ⬜ | ⬜ exp_S18 (planifié) | ⬜ |
| Déploiement modèles sur carte (S19) | ⬜ | ⬜ | ⬜ exp_S19 (planifié) | ⬜ |
| Notebooks (S7–16) | 🟡 | — | — | — |

### Current Sprint

| Sprint | Status | Focus |
|--------|:------:|-------|
| Sprints 1–5 | ✅ | Infrastructure + M1/M2/M3 + unsupervised baselines |
| Sprint 6 | ✅ | Extended scenarios (pump_by_id, pump_temporal, monitoring_by_location) |
| Sprints 7–12 | ✅ | Notebooks CL + intégration CWRU/Pronostia + scénarios by-fault/severity |
| Sprint 13 | ✅ | Anomaly detection one-class — 4 modèles Monitoring + DBSCAN/KMeans v2 CWRU + DBSCAN Monitoring/Pronostia (exp_086–093, exp_120–122) |
| Sprint 14 | ✅ | Monitoring anomaly detection complet — DBSCAN + EWC one-class + accumulate + by_location (exp_123–136) |
| Sprint 15 | ✅ | Pronostia anomaly detection — 6 modèles × by_condition refit + accumulate (exp_137–142) |
| Sprint 16 | ✅ | Portage MCU Phase 2 — toolchain ARM + ONNX export + Mahalanobis C + UART pipeline + profiling HW (3 µs @ 180 MHz, 128 B SRAM) |
| Sprint 17 | 🔄 | NUCLEO-F439ZI périphériques — GPIO ✅ UART ✅ TIM PWM ⬜ Renode CI ⬜ |
| Sprint 18 | ⬜ | Pipeline données capteurs sur carte — streaming UART, dataset builder, auto-profiling |
| Sprint 19 | ⬜ | Déploiement modèles Phase 1 sur carte — EWC + TinyOL C, Unity tests, exp_S19 |

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

**STM32N6** (Cortex-M55, ~64 KB SRAM, 512 KB Flash, NeuralART NPU)

> NPU is **inference-only**. Backpropagation runs on Cortex-M55 in FP32 software.

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
