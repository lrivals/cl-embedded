# S1910 — Tests Python recorder : JSON output, champs obligatoires, format unifié

| Champ | Valeur |
|-------|--------|
| **Sprint** | 19 |
| **Priorité** | 🟡 Important |
| **Statut** | ✅ Implémenté — 13 passed, 11 skipped (board) |
| **Durée estimée** | 2h |
| **Dépendances** | S1907 (board_experiment_recorder.py ✅) |
| **Fichiers cibles** | `tests/test_board_recorder.py` |

---

## Contexte

`board_experiment_recorder.py` doit produire des fichiers JSON qui s'intègrent dans le pipeline d'analyse Phase 1 (notebooks, `evaluate_all.py`). Ces tests vérifient la robustesse du recorder en mode dry-run (sans board) et valident le format JSON.

---

## Objectif

Implémenter `tests/test_board_recorder.py` avec des tests `pytest` couvrant :
1. Dry-run produit un JSON valide
2. Tous les champs obligatoires sont présents
3. Les valeurs sont du bon type
4. Le format est compatible Phase 1

---

## Champs obligatoires Phase 1

Ces 6 métriques doivent **toujours** être présentes (définies dans `evaluation/metrics.py` et attendues par les notebooks) :

```python
REQUIRED_KEYS = {
    "acc_final",            # float [0, 1]
    "avg_forgetting",       # float [0, 1] (AF)
    "backward_transfer",    # float (BWT, peut être négatif)
    "ram_peak_bytes",       # int > 0
    "inference_latency_ms", # float > 0
    "n_params",             # int > 0
}
```

Champs supplémentaires board (doivent aussi être présents) :
```python
BOARD_KEYS = {
    "exp_id", "model", "dataset", "platform",
    "date", "n_tasks", "n_samples_total", "config_snapshot"
}
```

---

## Tests à implémenter

```python
"""tests/test_board_recorder.py — Tests pytest pour board_experiment_recorder.py"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

RECORDER = Path("scripts/board_experiment_recorder.py")
OUTPUT_DIR = Path("experiments/test_recorder_tmp")

REQUIRED_KEYS = {
    "acc_final", "avg_forgetting", "backward_transfer",
    "ram_peak_bytes", "inference_latency_ms", "n_params",
}
BOARD_KEYS = {
    "exp_id", "model", "dataset", "platform",
    "date", "n_tasks", "n_samples_total", "config_snapshot",
}


@pytest.fixture(autouse=True)
def cleanup(tmp_path):
    yield
    # nettoyage automatique via tmp_path


def run_recorder(model: str, dataset: str, output: Path) -> Path:
    result = subprocess.run(
        [sys.executable, str(RECORDER),
         "--model", model,
         "--dataset", dataset,
         "--dry-run",
         "--output", str(output)],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f"recorder failed:\n{result.stderr}"
    return output / "results.json"


class TestDryRunOutput:
    def test_json_file_created(self, tmp_path):
        json_path = run_recorder("mahalanobis", "cwru", tmp_path / "exp")
        assert json_path.exists(), "results.json not created"

    def test_json_is_valid(self, tmp_path):
        json_path = run_recorder("mahalanobis", "cwru", tmp_path / "exp")
        data = json.loads(json_path.read_text())
        assert isinstance(data, dict)

    def test_required_keys_present(self, tmp_path):
        json_path = run_recorder("mahalanobis", "cwru", tmp_path / "exp")
        data = json.loads(json_path.read_text())
        missing = REQUIRED_KEYS - data.keys()
        assert not missing, f"Missing required keys: {missing}"

    def test_board_keys_present(self, tmp_path):
        json_path = run_recorder("mahalanobis", "cwru", tmp_path / "exp")
        data = json.loads(json_path.read_text())
        missing = BOARD_KEYS - data.keys()
        assert not missing, f"Missing board keys: {missing}"

    def test_metric_types(self, tmp_path):
        json_path = run_recorder("mahalanobis", "cwru", tmp_path / "exp")
        data = json.loads(json_path.read_text())
        assert isinstance(data["acc_final"], float)
        assert isinstance(data["avg_forgetting"], float)
        assert isinstance(data["backward_transfer"], float)
        assert isinstance(data["ram_peak_bytes"], (int, float))
        assert isinstance(data["inference_latency_ms"], float)
        assert isinstance(data["n_params"], int)

    def test_metric_ranges(self, tmp_path):
        json_path = run_recorder("mahalanobis", "cwru", tmp_path / "exp")
        data = json.loads(json_path.read_text())
        assert 0.0 <= data["acc_final"] <= 1.0
        assert data["avg_forgetting"] >= 0.0
        assert data["ram_peak_bytes"] > 0
        assert data["inference_latency_ms"] > 0.0
        assert data["n_params"] > 0

    def test_n_params_mahalanobis(self, tmp_path):
        json_path = run_recorder("mahalanobis", "cwru", tmp_path / "exp")
        data = json.loads(json_path.read_text())
        assert data["n_params"] == 30  # mean(5) + precision(25)

    def test_n_params_ewc(self, tmp_path):
        json_path = run_recorder("ewc", "monitoring", tmp_path / "exp")
        data = json.loads(json_path.read_text())
        assert data["n_params"] == 1538

    def test_config_snapshot_exists(self, tmp_path):
        out = tmp_path / "exp"
        run_recorder("mahalanobis", "cwru", out)
        snapshot = out / "config_snapshot.yaml"
        assert snapshot.exists(), "config_snapshot.yaml not copied"

    def test_platform_field(self, tmp_path):
        json_path = run_recorder("mahalanobis", "cwru", tmp_path / "exp")
        data = json.loads(json_path.read_text())
        assert data["platform"] == "nucleo_f439zi"

    @pytest.mark.parametrize("model,dataset", [
        ("mahalanobis", "cwru"),
        ("ewc", "monitoring"),
        ("tinyol", "cwru"),
    ])
    def test_all_models_dry_run(self, tmp_path, model, dataset):
        json_path = run_recorder(model, dataset, tmp_path / f"exp_{model}")
        data = json.loads(json_path.read_text())
        assert REQUIRED_KEYS <= data.keys()
```

---

## Fichiers cibles

| Fichier | Action |
|---------|--------|
| `tests/test_board_recorder.py` | Créer avec le contenu ci-dessus |
| `scripts/board_experiment_recorder.py` | SUT — ne pas modifier |
| `configs/board_mahalanobis.yaml` | Doit exister (fixture implicite) |
| `configs/board_ewc.yaml` | Idem |
| `configs/board_tinyol.yaml` | Idem |

---

## Vérification

```bash
pytest tests/test_board_recorder.py -v
# Attendu : 11 tests PASSED (+ 3 paramétrisés = 14 total)
```

- [ ] Tous les tests PASS en dry-run sans board connecté
- [ ] `pytest tests/test_board_recorder.py -v --tb=short` → 0 FAILED
- [ ] Le test `test_config_snapshot_exists` passe → `config_snapshot.yaml` copié
