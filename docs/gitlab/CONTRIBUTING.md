# Contributing — CL-Embedded

This guide helps a new contributor set up the project, understand its layout, and run the test
suite. It is the entry point for anyone picking up the work.

## 1. Environment setup

Requirements: Python ≥ 3.10, and for firmware work `arm-none-eabi-gcc` + OpenOCD.

```bash
git clone <gitlab-url> cl-embedded
cd cl-embedded
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Datasets are **not** committed (`data/` is gitignored). Place raw data under `data/raw/<dataset>/`
following the layout documented in `docs/context/datasets.md`.

## 2. Project layout (where things live)

| Area | Path | Notes |
|------|------|-------|
| Models | `src/models/{ewc,hdc,tinyol,unsupervised}/` | One package per CL family |
| Dataset loaders | `src/data/` | One loader per dataset (CWRU, Pronostia, CMAPSS, Paderborn, …) |
| Training loops | `src/training/` | CL scenarios + anomaly-detection scenarios |
| Evaluation | `src/evaluation/` | Metrics, memory profiler, plots — reuse, do not reimplement |
| Configs | `configs/` | All sizes/hyperparameters live here as YAML, never hardcoded |
| Firmware | `firmware/stm32f4_blink/` | NUCLEO-F439ZI C sources, headers, Unity tests |
| CLI scripts | `scripts/` | Train, evaluate, export weights, board streaming |
| Experiments | `experiments/` | JSON outputs — every number comes from a script run |

## 3. Coding conventions

- **Python**: type hints on public functions, NumPy-style docstrings, `black` (line length 100)
  and `ruff` for linting/formatting.
- **Configs over constants**: any size (layers, buffers, embeddings) must be a named value in a
  `configs/*.yaml` file. Defaults must fit in 256 KB SRAM (NUCLEO-F439ZI).
- **Reproducibility**: seeds are fixed via `src/utils/reproducibility.py` (`set_seed(42)`).
  Results are never hardcoded — they are produced by running a script.
- **Memory annotations**: keep `# MEM: X B @ FP32 / Y B @ INT8` comments on model layers.
- **Firmware**: buffer/layer sizes are `#define`s in `inc/` headers, never inlined in `.c`.
  Generated weight headers (`model_weights.h`, …) are produced by `scripts/export_weights_*.py`,
  never edited by hand.
- **UART protocol**: any change to the firmware UART protocol (`pipeline.c`) must update the host
  driver `scripts/sensor_stream.py` in the same change.

## 4. Running the tests

```bash
# Python unit tests
pytest tests/ -v

# Firmware unit tests (Unity on host)
make -C firmware/stm32f4_blink/ test

# Lint / format
black --check . && ruff check .
```

A change is considered complete only when the relevant Python tests and the firmware Unity suite
pass with no new regressions.

## 5. Hardware constraints (must respect)

1. RAM ≤ 256 KB (192 KB SRAM + 64 KB CCM on the NUCLEO-F439ZI).
2. No NPU — forward and backward passes run on the Cortex-M4 FPU in FP32.
3. Inference + update latency ≤ 100 ms (measured via DWT).
4. No full dataset in RAM — online learning or a bounded buffer only.

See `docs/context/hardware_constraints.md` for the detailed rationale.

## 6. Workflow

1. Branch from `main`.
2. Implement against a config; keep numbers reproducible from scripts.
3. Run the Python + firmware test suites.
4. Open a merge request describing what was validated (metrics, board results if applicable).
