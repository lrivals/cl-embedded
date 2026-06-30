# S2801 — Script benchmark unifié `benchmark_int8_fp32.py`

| Champ | Valeur |
|-------|--------|
| **Sprint** | 28 |
| **Priorité** | 🔴 Critique — bloquant pour S2807, S2808, S2810, S2811 |
| **Statut** | ✅ Implémenté (12 juin 2026) — voir « Bilan d'implémentation » ci-dessous |
| **Durée estimée** | 3h |
| **Dépendances** | `src/models/ewc/ewc_mlp_int8.py` ✅ · `src/utils/quantization.py` ✅ · S2803 (hdc_int8.py) · S2804 (tinyol_int8.py) · S2805 (mahalanobis_int8.py) |
| **Fichier cible** | `scripts/benchmark_int8_fp32.py` |
| **Références** | `scripts/compare_experiments.py` (pattern chargement configs) · `src/evaluation/memory_profiler.py` (tracemalloc) · `src/evaluation/metrics.py` (AUROC, F1) |

---

## Contexte

Le projet dispose de modèles INT8 Python dispersés mais pas d'outil unifié pour comparer FP32 vs INT8 de façon reproductible sur n'importe quel modèle × dataset. Ce script centralise cette comparaison et produit un JSON normalisé que `generate_int8_heatmaps.py` (S2810) peut ingérer directement.

---

## Interface CLI

```bash
python scripts/benchmark_int8_fp32.py \
    --model {ewc,hdc,tinyol,mahalanobis} \
    --config configs/ewc_int8_cwru.yaml \
    --output experiments/exp_S28_PC_ewc_hdc/results_ewc_cwru.json \
    [--n_samples 500]   # limiter le dataset pour tests rapides
```

---

## Format JSON de sortie

```json
{
  "model": "ewc",
  "dataset": "cwru",
  "config_path": "configs/ewc_int8_cwru.yaml",
  "timestamp": "2026-06-16T10:00:00",
  "fp32": {
    "metric_name": "auroc",
    "metric_value": 0.912,
    "ram_bytes": 9728,
    "latency_ms": 0.045
  },
  "int8": {
    "metric_name": "auroc",
    "metric_value": 0.899,
    "ram_bytes": 3600,
    "latency_ms": 0.051
  },
  "delta_metric": -0.013,
  "ram_ratio": 2.702,
  "gap3_metric_ok": true,
  "gap3_ram_ok": true
}
```

**Champs** :
- `metric_name` : `"auroc"` pour EWC/TinyOL/Mahalanobis anomaly, `"f1_macro"` pour HDC classification, `"rmse"` pour régression
- `delta_metric` = `int8.metric_value - fp32.metric_value` (négatif = dégradation)
- `gap3_metric_ok` = `abs(delta_metric) < 0.02` (critère Gap 3)
- `gap3_ram_ok` = `ram_ratio > 1.0`

---

## Logique interne

```python
# Pseudo-code structure
def run_benchmark(model_name, config_path, output_path, n_samples=None):
    config = load_yaml(config_path)
    dataset = load_dataset(config)  # utiliser loaders existants

    # Phase FP32
    model_fp32 = load_model_fp32(model_name, config)
    ram_fp32 = measure_ram(model_fp32)           # tracemalloc
    latency_fp32 = measure_latency(model_fp32, dataset)  # 100 runs
    metric_fp32 = evaluate(model_fp32, dataset)

    # Phase INT8
    model_int8 = load_model_int8(model_name, config)  # hdc_int8.py, etc.
    ram_int8 = model_int8.get_memory_footprint_int8()
    latency_int8 = measure_latency(model_int8, dataset)
    metric_int8 = evaluate_int8(model_int8, dataset)

    # Sortie JSON
    result = build_result_dict(...)
    save_json(result, output_path)
```

---

## Conventions de config YAML INT8

Les configs INT8 héritent de la config FP32 avec un champ supplémentaire :

```yaml
# configs/ewc_int8_cwru.yaml
extends: configs/board_ewc.yaml   # ou ewc_config.yaml
quantization: int8
dataset: cwru
```

---

## Vérification

```bash
# Test rapide sur mini-dataset synthétique
python scripts/benchmark_int8_fp32.py \
    --model ewc --config configs/ewc_int8_cwru.yaml \
    --output /tmp/test_benchmark.json --n_samples 50

# Vérifier JSON valide
python -c "import json; d=json.load(open('/tmp/test_benchmark.json')); assert d['ram_ratio'] > 1.0"

# Tests unitaires
pytest tests/test_int8_benchmark.py -v
```

---

## Bilan d'implémentation (12 juin 2026)

**Livré** : `scripts/benchmark_int8_fp32.py` (registre d'adaptateurs `MODEL_ADAPTERS`),
`configs/ewc_int8_monitoring.yaml`, `tests/test_int8_benchmark.py`, et le support
`extends:` (`load_config_extends`) ajouté à `src/utils/config_loader.py`.

**Périmètre réel** (décidé après constat d'état du dépôt) :

- **EWC** : entièrement câblé. FP32 = `EWCMlpClassifier`, INT8 = `EWCMlpInt8Classifier`,
  entraînés in-script (seed=42) en réutilisant `train_ewc.py` (`_get_tasks`, `train_ewc`,
  `_compute_auroc_after_training`). Métrique `auroc`, RAM via `estimate_ram_bytes(dtype)`.
- **HDC** : adaptateur best-effort. HDC est **nativement INT8** → métrique INT8 == FP32,
  seule la RAM diffère ; un avertissement explicite est émis. Nécessite une config HDC
  (sections `hdc`/`data.n_classes`/`n_features`).
- **TinyOL / Mahalanobis** : `build_int8` → `None` ⇒ `NotImplementedError` clair pointant
  **S2804 / S2805** (variants INT8 non encore implémentés).

**Écart au spec assumé** : le spec montrait `extends: configs/board_ewc.yaml`, mais
`board_ewc.yaml` est plat (sans sections `data/training`) et incompatible avec la boucle
`train_ewc`. La config d'exemple hérite donc de `ewc_config.yaml`
(`configs/ewc_int8_monitoring.yaml`, dataset monitoring binaire → AUROC propre, données
présentes localement). Les configs `ewc_int8_{cwru,pronostia,paderborn}.yaml` restent à
créer (S2802).

**Vérification** : run EWC réel `--n_samples 64` ⇒ JSON valide
(`ram_ratio=4.0`, `gap3_ram_ok=true`, `delta_metric≈+0.003`, `gap3_metric_ok=true`) ;
`pytest tests/test_int8_benchmark.py -v` ⇒ **6/6 PASS**.

> Note : `tests/test_int8_benchmark.py` couvre le schéma JSON, les flags Gap 3, le skip
> propre tinyol/mahalanobis et un smoke EWC réel. Le test « tous modèles × mini-dataset
> synthétique » de **S2809** reste à compléter une fois S2803/S2804/S2805 livrés.
