"""
test_feature_selection.py — Tests Sprint 35 (S3501/S3502/S3503).

Style léger : valide les **schémas** des configs générées et la logique d'énumération du
sweep, sans entraînement réel (rapide, déterministe). Le marqueur ``best_features`` permet
``pytest -m best_features``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.feature_conditions import (  # noqa: E402
    DATASETS,
    MODELS,
    NATIVE_FEATURE_NAMES,
)

pytestmark = pytest.mark.best_features


# ── S3502 : configs all_features ──────────────────────────────────────────────


@pytest.mark.parametrize("dataset", DATASETS)
def test_all_features_config_schema(dataset: str) -> None:
    path = ROOT / "configs" / "all_features" / f"{dataset}.yaml"
    assert path.exists(), f"{path} manquant (S3502)"
    cfg = yaml.safe_load(path.read_text())
    assert cfg["dataset"] == dataset
    assert cfg["condition"] == "all"
    assert cfg["n_features"] == len(cfg["feature_names"])
    # Cohérence avec les constantes des loaders.
    assert cfg["feature_names"] == NATIVE_FEATURE_NAMES[dataset]


# ── S3501 : schéma des configs best_features (si déjà générées) ───────────────


def _best_feature_files() -> list[Path]:
    d = ROOT / "configs" / "best_features"
    return sorted(d.glob("*.yaml")) if d.exists() else []


@pytest.mark.parametrize("path", _best_feature_files(), ids=lambda p: p.stem)
def test_best_features_config_schema(path: Path) -> None:
    cfg = yaml.safe_load(path.read_text())
    model, dataset = cfg["model"], cfg["dataset"]
    assert model in MODELS
    assert dataset in DATASETS
    n_total = cfg["n_features_total"]
    assert n_total == len(NATIVE_FEATURE_NAMES[dataset])

    k = cfg["n_features_selected"]
    idx = cfg["selected_indices"]
    assert k == len(idx) == len(cfg["selected_features"])
    assert 1 <= k <= n_total
    assert all(0 <= i < n_total for i in idx)
    assert len(set(idx)) == len(idx), "indices dupliqués"
    # Les indices doivent correspondre aux noms sélectionnés.
    assert [NATIVE_FEATURE_NAMES[dataset][i] for i in idx] == cfg["selected_features"]

    # k* = plus petit k à <parcimonie du F1 max (règle de parcimonie).
    f1_by_k = {int(kk): float(v) for kk, v in cfg["val_f1_by_k"].items()}
    assert set(f1_by_k) == set(range(1, n_total + 1))
    f1_max = max(f1_by_k.values())
    parcimonie = cfg.get("parcimonie", 0.01)
    assert f1_by_k[k] >= f1_max - parcimonie - 1e-9
    assert all(f1_by_k[kk] < f1_max - parcimonie for kk in range(1, k))


# ── S3504 : métrique F1 (classe faulty) — définition partagée PC↔board ────────


def test_compute_fault_f1_perfect() -> None:
    import numpy as np

    from src.evaluation.metrics import compute_fault_f1

    y = np.array([0, 1, 0, 1, 1])
    out = compute_fault_f1(y, y.copy())
    assert out["f1_faulty"] == 1.0
    assert out["f1_macro"] == 1.0
    assert out["precision_faulty"] == 1.0
    assert out["recall_faulty"] == 1.0


def test_compute_fault_f1_known_case() -> None:
    import numpy as np

    from src.evaluation.metrics import compute_fault_f1

    # TP=2, FP=1, FN=1 → precision=2/3, recall=2/3, F1_faulty=2/3.
    y_true = np.array([1, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0])
    out = compute_fault_f1(y_true, y_pred)
    assert out["precision_faulty"] == pytest.approx(2 / 3)
    assert out["recall_faulty"] == pytest.approx(2 / 3)
    assert out["f1_faulty"] == pytest.approx(2 / 3)
    # macro = moyenne F1(classe0=0.5, classe1=2/3).
    assert out["f1_macro"] == pytest.approx((0.5 + 2 / 3) / 2)


def test_compute_fault_f1_monoclass_no_raise() -> None:
    import numpy as np

    from src.evaluation.metrics import compute_fault_f1

    # Aucun positif prédit → zero_division=0, pas d'exception.
    out = compute_fault_f1(np.array([1, 1, 0]), np.array([0, 0, 0]))
    assert out["f1_faulty"] == 0.0


# ── S3513 : déterminisme de permutation_importance ────────────────────────────


def test_permutation_importance_deterministic_and_ranks_informative() -> None:
    """Seed fixe → scores identiques sur 2 appels ; feature informative > bruit (S3513).

    Cas connu : la colonne 0 décide le label (predict_fn = colonne 0 binarisée), la colonne 1
    est du bruit pur. La permutation de la colonne 0 doit dégrader l'accuracy (importance > 0),
    celle de la colonne 1 quasiment pas (importance ≈ 0).
    """
    import numpy as np

    from src.evaluation.feature_importance import permutation_importance

    rng = np.random.default_rng(0)
    n = 400
    informative = rng.random(n).astype(np.float32)  # ∈ [0,1] → décide le label
    noise = rng.random(n).astype(np.float32)
    X = np.stack([informative, noise], axis=1)
    y = (informative >= 0.5).astype(int)

    def predict_fn(arr: np.ndarray) -> np.ndarray:
        return arr[:, 0]  # score = feature informative (binarisé à threshold=0.5)

    names = ["informative", "noise"]
    out1 = permutation_importance(predict_fn, X, y, names, n_repeats=5, random_state=42)
    out2 = permutation_importance(predict_fn, X, y, names, n_repeats=5, random_state=42)

    # Déterminisme : même seed → mêmes scores.
    assert out1 == out2
    # La feature informative est la plus importante, le bruit ≈ 0.
    assert out1["informative"] > out1["noise"]
    assert out1["informative"] > 0.4  # permuter la colonne décisive casse l'accuracy
    assert abs(out1["noise"]) < 0.1


# ── S3503 : énumération du sweep ──────────────────────────────────────────────


def test_sweep_dry_run_enumerates_60(capsys) -> None:
    import scripts.run_feature_condition_sweep as sweep

    sys.argv = ["run_feature_condition_sweep.py", "--dry-run"]
    sweep.main()
    out = capsys.readouterr().out
    assert "60 cellules" in out
    # Une cellule représentative par condition.
    for token in ["exp_S35_PC_5feat_ewc_cwru", "exp_S35_PC_all_hdc_cmapss",
                  "exp_S35_PC_best_mahalanobis_paderborn"]:
        assert token in out


def test_resolve_indices_all_and_monitoring_fallback() -> None:
    from scripts.run_feature_condition_sweep import resolve_feature_indices

    # `all` → tous les indices natifs.
    idx, note = resolve_feature_indices("all", "ewc", "cwru")
    assert idx == list(range(9))
    assert "natives" in note

    # monitoring n'a pas de subset top-5 → 5feat retombe sur all (4 features).
    idx, note = resolve_feature_indices("5feat", "ewc", "monitoring")
    assert idx == list(range(4))
    assert "5feat" in note
