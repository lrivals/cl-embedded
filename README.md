# CL-Embedded — Continual Learning for Resource-Constrained Embedded Systems

> M2 Research Internship — ISAE-SUPAERO (DISC) × ENAC (LII) × Edge Spectrum  
> Author: Léonard Rivals | March–August 2026

## Overview

This repository implements and compares three continual learning (CL) methods designed for deployment on microcontrollers with severely limited resources (target: STM32N6, ~64 KB RAM). The application domain is industrial predictive maintenance.

The implementations target **PC-first development** with explicit design constraints ensuring portability to the MCU target (no dynamic allocation, SGD-only optimizer, ReLU activations, fixed normalization statistics).

## The Three Models

| ID | Method | CL Family | Dataset | Target RAM |
|----|--------|-----------|---------|-----------|
| **M2** | EWC Online + MLP | Regularization-based | Equipment Monitoring | ~9 KB |
| **M3** | Hyperdimensional Computing (HDC) | Architecture-based | Equipment Monitoring | ~12 KB |
| **M1** | TinyOL + OtO Head | Architecture-based | Pump Maintenance | ~7 KB |

All models operate within the 64 KB RAM budget of the STM32N6 target.

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

# Train M2 (EWC) on Dataset 2 — start here
python scripts/train_ewc.py --config configs/ewc_config.yaml

# Generate HDC base vectors (once, before training HDC)
python -m src.models.hdc.base_vectors

# Train M3 (HDC) on Dataset 2
python scripts/train_hdc.py --config configs/hdc_config.yaml

# Train M1 (TinyOL) on Dataset 1
python scripts/train_tinyol.py --config configs/tinyol_config.yaml

# Compare EWC vs HDC vs Fine-tuning (Sprint 2)
jupyter nbconvert --to notebook --execute notebooks/02_baseline_comparison.ipynb

# Compare all models
python scripts/evaluate_all.py --exp_dir experiments/

# Memory profiling
python scripts/profile_memory.py --model ewc
```

## Repository Structure

```
cl-embedded/
├── CLAUDE.md               # Context for Claude Code (read first)
├── README.md               # This file
├── configs/                # YAML hyperparameters (versioned)
├── data/                   # Raw + processed data (gitignored)
├── docs/
│   ├── models/             # Detailed implementation specs (3 models)
│   ├── context/            # Project context, hardware, datasets
│   └── roadmap.md          # Development roadmap
├── skills/                 # Claude prompting guides
├── src/
│   ├── data/               # Dataset loaders
│   ├── models/             # Model implementations
│   ├── training/           # CL training loops + baselines
│   ├── evaluation/         # CL metrics + memory profiler
│   └── utils/              # Quantization helpers, reproducibility
├── experiments/            # Reproducible experiment outputs
├── notebooks/              # Exploration + visualization
├── tests/                  # Unit tests
└── scripts/                # CLI entry points
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
| [`notebooks/01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb) | Exploration Dataset 2 (Equipment Monitoring) |
| [`notebooks/02_baseline_comparison.ipynb`](notebooks/02_baseline_comparison.ipynb) | **Comparison EWC vs HDC vs Fine-tuning** (Sprint 2) |

## Results (Sprint 2 — Dataset 2, seed=42, 3 domains)

| Metric | EWC Online | HDC | Fine-tuning naïf |
|--------|:----------:|:---:|:----------------:|
| AA | **0.9824** | 0.8698 | 0.9811 |
| AF | **0.0010** | **0.0000** | 0.0000 |
| BWT | +0.0000 | +0.0019 | +0.0010 |
| RAM peak inference | **1.1 KB** | 14.2 KB | — |
| Inference latency | **0.036 ms** | 0.048 ms | — |
| Budget 64 KB | ✅ 1.8% | ✅ 22.1% | — |

> Dataset 2 (Equipment Monitoring) — Pump → Turbine → Compressor.  
> HDC: AF = 0 by construction (additive accumulation, no catastrophic forgetting possible). Lower accuracy than EWC expected — HDC is less expressive but uses ~12× less RAM for weight updates.  
> Full results: `experiments/exp_002_hdc_dataset2/results/` · Analysis: `notebooks/02_baseline_comparison.ipynb`.

### M2 EWC — exp_001 (4 avril 2026)

| Metric | Value |
|--------|-------|
| AA | 0.9824 |
| AF | 0.0010 |
| BWT | +0.0000 |
| RAM peak inference | 1 171 B (1.1 KB) |
| RAM peak update | 6 837 B (6.7 KB) |
| Inference latency | 0.036 ms |
| n_params | 705 |

### M3 HDC — exp_002 (6 avril 2026)

| Metric | Value |
|--------|-------|
| AA | 0.8698 |
| AF | 0.0000 |
| BWT | +0.0019 |
| RAM peak inference (measured) | 14 504 B (14.2 KB) |
| RAM estimated FP32 | 14 344 B (14.0 KB) |
| RAM estimated INT8 | 6 152 B (6.0 KB) |
| Inference latency | 0.048 ms |
| n_params (prototype elements) | 2 048 |

## Progress Indicators

| Model | Implemented | Tested | Experiment | RAM measured |
|-------|:-----------:|:------:|:----------:|:------------:|
| M2 EWC + MLP | ✅ | ✅ | ✅ exp_001 | ✅ |
| M3 HDC | ✅ | ✅ | ✅ exp_002 | ✅ |
| M1 TinyOL | ⬜ | ⬜ | ⬜ | ⬜ |
| M1 + UINT8 buffer | ⬜ | ⬜ | ⬜ | ⬜ |

## Evaluation Metrics

For every CL experiment:

| Metric | Description |
|--------|-------------|
| `aa` | Average Accuracy across all tasks |
| `af` | Average Forgetting (0 = no forgetting) |
| `bwt` | Backward Transfer |
| `ram_peak_bytes` | Peak RAM measured via tracemalloc |
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
