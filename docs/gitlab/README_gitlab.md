# CL-Embedded — Continual Learning for Resource-Constrained Embedded Systems

> M2 Research Internship — ISAE-SUPAERO (DISC) × ENAC (LII) × Edge Spectrum
> Author: Léonard Rivals | March–August 2026

## Overview

This repository implements and compares continual learning (CL) methods for deployment on
microcontrollers with severely limited resources (**active board: NUCLEO-F439ZI**, Cortex-M4
@ 180 MHz, 256 KB SRAM). The application domain is industrial predictive maintenance.

The implementations target **PC-first development** with explicit portability constraints (no
dynamic allocation, SGD-only optimizer, ReLU activations, fixed normalization statistics), then
are ported to C and validated on hardware. All three scientific gaps have been addressed.

> **Note**: The original target was STM32N6 (Cortex-M55, NPU) but is unavailable. All firmware
> development targets NUCLEO-F439ZI. Do not design for 64 KB.

## Models

### Supervised CL Methods (board-validated)

| ID | Method | CL Family | Datasets | Board RAM (.bss) |
|----|--------|-----------|----------|------------------|
| **M2** | EWC Online + MLP | Regularization-based | Monitoring, CWRU, CMAPSS, Paderborn | ~6.7 KB |
| **M2b** | EWC INT8 | Regularization + quantization | CMAPSS, Paderborn | ~1.7 KB |
| **M3** | HDC Hyperdimensional | Architecture-based | Monitoring, Pronostia, CWRU | ~14.0 KB |
| **M1** | TinyOL + OtO Head | Architecture-based | Pump, Monitoring | ~5.8 KB |

### Unsupervised / Anomaly Detection Baselines

| ID | Method | MCU-compatible | Datasets |
|----|--------|:--------------:|----------|
| **Mahalanobis** | Mahalanobis distance | ✅ | All — 80 B model, 3 µs @ 180 MHz |
| **KMeans** | K-Means clustering | ❌ | Monitoring, CWRU, Pronostia |
| **KNN** | KNN anomaly detection | ❌ | Monitoring, CWRU, Pronostia |
| **PCA** | PCA reconstruction error | ❌ | Monitoring, CWRU, Pronostia |
| **DBSCAN** | Density-based clustering | ❌ | Monitoring, CWRU, Pronostia |

All one-class models train on normal data only, evaluated with AUROC [T×T] matrices.

## Scientific Positioning — Triple Gap

| Gap | Claim | Status |
|-----|-------|--------|
| **Gap 1** | Validation on real industrial time-series data | ✅ CWRU, Pronostia, CMAPSS, Paderborn validated |
| **Gap 2** | CL demonstrated under 100 KB RAM with precise measurements | ✅ RAM 1 000 B (.bss), latency 3.7 µs P50 / 4.0 µs P99 on NUCLEO-F439ZI |
| **Gap 3** | INT8 quantization during incremental training | ✅ EWC INT8 + HDC INT8 implemented and board-validated |

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

# --- Firmware (NUCLEO-F439ZI) ---
make -C firmware/stm32f4_blink/ all     # Compile for Cortex-M4
make -C firmware/stm32f4_blink/ test    # Run Unity tests on host (TEST_MODE=1)
make -C firmware/stm32f4_blink/ flash   # Flash via OpenOCD ST-LINK
make -C firmware/stm32f4_blink/ size    # Flash/RAM size report

# Export model weights → C header
python scripts/export_weights_c.py --model ewc --config configs/board_ewc.yaml

# Board streaming & experiments
python scripts/sensor_stream.py --port /dev/ttyACM0 --dataset cwru

# --- Evaluation ---
python scripts/evaluate_all.py --exp_dir experiments/
python scripts/profile_memory.py --model ewc --dataset monitoring
```

## Repository Structure

```
cl-embedded/
├── README.md                   # This file
├── CONTRIBUTING.md             # Developer onboarding guide
├── configs/                    # YAML configs: base / board / dataset / anomaly / feature-subset
├── data/                       # Raw + processed data (gitignored)
├── docs/                       # Specs (M1–M4), hardware/dataset context, roadmaps, sprint notes
├── firmware/
│   └── stm32f4_blink/          # NUCLEO-F439ZI firmware (Cortex-M4)
├── src/
│   ├── data/                   # Dataset loaders
│   ├── models/                 # ewc/ hdc/ tinyol/ unsupervised/
│   ├── training/               # CL scenarios
│   ├── evaluation/             # metrics, memory_profiler, anomaly/online metrics, plots
│   └── utils/                  # Reproducibility, config loader
├── experiments/                # Experiment outputs (JSON results)
├── notebooks/                  # Exploration + visualization
├── tests/                      # Python unit tests
└── scripts/                    # CLI entry points (train, eval, export, board)
```

## Key Design Constraints

All implementations respect MCU portability requirements:

- **SGD optimizer only** (Adam prohibited — memory overhead)
- **ReLU activations only** (INT8-friendly for CMSIS-NN)
- **Fixed normalization statistics** (computed offline, stored in `configs/`)
- **No dynamic tensor allocation** in forward passes
- **Memory annotations** (`# MEM: X B @ FP32`) on every layer

## License & Contact

Research artifact produced during an M2 internship at ISAE-SUPAERO. See `CONTRIBUTING.md` for
how to set up the project and run the test suite.
