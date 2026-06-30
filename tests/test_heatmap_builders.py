"""
test_heatmap_builders.py — Tests Sprint 35 (S3510/S3513).

Valide les **builders importables** de ``scripts/generate_comparison_sprint23.py`` qui
alimentent les 12 heatmaps ``{F1, acc_final} × {5feat, all, best} × {board, pc}`` :

- ``_load_s35_conditions(root)`` ingère ``exp_S35_{PC,board}_*`` → structure complète
  ``[cond][ds][model][platform]`` (cellules mesurées = float, non mesurées = None « pending ») ;
- une matrice 5 datasets × 4 modèles assemblée par ``(metric, condition, platform)`` a la bonne
  forme et masque les cellules pending (pas de NaN affiché comme valeur) ;
- ``_apply_s3509_override`` remplace l'artefact HDC×monitoring board 0.1133 par la valeur mesurée.

On ne duplique PAS la logique du notebook : on teste la **source de données** des heatmaps.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.generate_comparison_sprint23 import (  # noqa: E402
    _DATASETS,
    _PLATFORMS,
    _S35_CONDITIONS,
    _S35_MODELS,
    _apply_s3509_override,
    _load_s35_conditions,
)

pytestmark = pytest.mark.best_features


def _write_exp(root: Path, name: str, payload: dict) -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps(payload))


@pytest.fixture()
def fake_experiments(tmp_path: Path) -> Path:
    """Quelques expériences S35 synthétiques (PC + board) + 1 bruit hors-périmètre."""
    _write_exp(tmp_path, "exp_S35_PC_5feat_ewc_cwru", {
        "exp_id": "exp_S35_PC_5feat_ewc_cwru", "condition": "5feat",
        "dataset": "cwru", "model": "ewc", "acc_final": 1.0, "f1_faulty": 0.97})
    _write_exp(tmp_path, "exp_S35_board_all_ewc_cmapss", {
        "exp_id": "exp_S35_board_all_ewc_cmapss", "condition": "all",
        "dataset": "cmapss", "model": "ewc", "online_accuracy": 0.9333,
        "f1_faulty": 0.6154, "n_features": 21})
    # board HDC×monitoring corrigé (S3509) — valeur mesurée, pas l'artefact.
    _write_exp(tmp_path, "exp_S35_board_all_hdc_monitoring", {
        "exp_id": "exp_S35_board_all_hdc_monitoring", "condition": "all",
        "dataset": "monitoring", "model": "hdc", "online_accuracy": 0.8667})
    # Bruit : ne doit pas être ingéré (modèle hors _S35_MODELS).
    _write_exp(tmp_path, "exp_S35_PC_5feat_ewc_int8_cwru", {
        "condition": "5feat", "dataset": "cwru", "model": "ewc_int8",
        "acc_final": 0.5})
    return tmp_path


def test_load_s35_conditions_full_structure(fake_experiments: Path) -> None:
    by_cond = _load_s35_conditions(fake_experiments)
    # Structure complète : 3 conditions × 5 datasets × 4 modèles × 2 plateformes.
    assert set(by_cond) == set(_S35_CONDITIONS)
    for cond in _S35_CONDITIONS:
        assert set(by_cond[cond]) == set(_DATASETS)
        for ds in _DATASETS:
            assert set(by_cond[cond][ds]) == set(_S35_MODELS)
            for m in _S35_MODELS:
                assert set(by_cond[cond][ds][m]) == set(_PLATFORMS)


def test_load_s35_conditions_measured_vs_pending(fake_experiments: Path) -> None:
    by_cond = _load_s35_conditions(fake_experiments)

    # Cellule mesurée (PC) → float, jamais NaN.
    pc = by_cond["5feat"]["cwru"]["ewc"]["pc"]
    assert pc["acc_final"] == 1.0
    assert pc["f1_faulty"] == pytest.approx(0.97)
    assert not np.isnan(pc["acc_final"])

    # Cellule mesurée (board) lit online_accuracy.
    board = by_cond["all"]["cmapss"]["ewc"]["nucleo_f439zi"]
    assert board["acc_final"] == pytest.approx(0.9333)
    assert board["f1_faulty"] == pytest.approx(0.6154)

    # Cellule non mesurée → None (« pending »), jamais 0/NaN factice.
    pending = by_cond["best"]["paderborn"]["mahalanobis"]["pc"]
    assert pending["acc_final"] is None
    assert pending["f1_faulty"] is None
    assert pending["note"] == "pending"

    # ewc_int8 (hors périmètre) n'a rien injecté.
    assert by_cond["5feat"]["cwru"]["ewc"]["pc"]["acc_final"] == 1.0  # ewc reste l'ewc fp32


def test_matrix_5x4_shape_and_masking(fake_experiments: Path) -> None:
    """Assemble une matrice datasets × modèles comme le builder de heatmap (S3510)."""
    by_cond = _load_s35_conditions(fake_experiments)

    def build_matrix(metric: str, condition: str, platform: str) -> np.ndarray:
        mat = np.full((len(_DATASETS), len(_S35_MODELS)), np.nan)
        for i, ds in enumerate(_DATASETS):
            for j, m in enumerate(_S35_MODELS):
                v = by_cond[condition][ds][m][platform][metric]
                if v is not None:
                    mat[i, j] = float(v)
        return mat

    mat = build_matrix("acc_final", "all", "nucleo_f439zi")
    assert mat.shape == (len(_DATASETS), len(_S35_MODELS)) == (5, 4)

    # cmapss×ewc board renseigné ; le reste pending → NaN (masqué, pas affiché).
    i_cmapss = _DATASETS.index("cmapss")
    j_ewc = _S35_MODELS.index("ewc")
    assert mat[i_cmapss, j_ewc] == pytest.approx(0.9333)
    # Une cellule jamais mesurée reste NaN (masque heatmap).
    assert np.isnan(mat[_DATASETS.index("paderborn"), _S35_MODELS.index("hdc")])


def test_apply_s3509_override_replaces_artifact(fake_experiments: Path) -> None:
    by_cond = _load_s35_conditions(fake_experiments)
    # Simule l'index `results` héritant de l'artefact 0.1133.
    results = {"monitoring": {"hdc": {"nucleo_f439zi": {"acc_final": 0.1133}}}}
    _apply_s3509_override(results, by_cond)
    corrected = results["monitoring"]["hdc"]["nucleo_f439zi"]["acc_final"]
    assert corrected != pytest.approx(0.1133)
    assert corrected == pytest.approx(0.8667)
