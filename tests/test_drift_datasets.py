"""Tests des datasets de drift (Sprint 43, S4305).

Couvre : contrat des loaders (formes cohérentes, ordre chronologique préservé), validité de
la ground-truth ``drift_points``, normalisation figée sur le segment 0, vérité-terrain exacte
du synthétique, absence de chiffre de résultat en dur dans le catalogue de figures S4304, et
déterminisme (idempotence) de la caractérisation.

Les datasets réels (gas/hydraulic/electricity) nécessitent ``data/raw/`` : les tests concernés
sont ``pytest.mark.skipif`` honnêtes si les données sont absentes. Le **synthétique** est généré
à la volée (numpy) et toujours testé.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from src.data import DRIFT_CONFIGS, DRIFT_LOADERS
from src.data.drift_dataset import DriftDataset, freeze_zscore
from src.utils.config_loader import load_config

ROOT = Path(__file__).resolve().parents[1]

# Datasets à ground-truth ponctuelle (drift_points non vides) vs structurelle/absente.
POINTWISE_GT = {"synthetic", "gas_sensor_drift", "hydraulic"}
NO_POINTWISE_GT = {"electricity"}


def _raw_available(dataset: str) -> bool:
    """Vrai si les données brutes requises par le loader sont présentes (sinon skip honnête)."""
    if dataset == "synthetic":
        return True  # généré à la volée, aucune donnée disque
    cfg = load_config(DRIFT_CONFIGS[dataset])
    raw_path = cfg.get("data", {}).get("raw_path")
    if raw_path is None:
        return False
    return (ROOT / raw_path).exists()


def _load(dataset: str) -> DriftDataset:
    return DRIFT_LOADERS[dataset](DRIFT_CONFIGS[dataset])


# ── Contrat des loaders ───────────────────────────────────────────────────────

@pytest.mark.parametrize("dataset", sorted(DRIFT_LOADERS))
def test_loader_contract(dataset: str) -> None:
    """Chaque loader retourne un DriftDataset aux formes cohérentes."""
    if not _raw_available(dataset):
        pytest.skip(f"data/raw absent pour {dataset}")
    d = _load(dataset)
    assert isinstance(d, DriftDataset)
    assert d.X.ndim == 2 and d.X.shape[0] > 0
    assert d.X.shape[1] == len(d.feature_names) == d.n_features
    assert d.drift_type is not None
    # Labels optionnels mais, si présents, alignés sur X.
    if d.y is not None:
        assert len(d.y) == d.n_samples


@pytest.mark.parametrize("dataset", sorted(DRIFT_LOADERS))
def test_chronological_order_preserved(dataset: str) -> None:
    """Deux chargements successifs donnent le même X (aucun shuffle global — drift ordonné)."""
    if not _raw_available(dataset):
        pytest.skip(f"data/raw absent pour {dataset}")
    a = _load(dataset).X
    b = _load(dataset).X
    assert a.shape == b.shape
    assert np.array_equal(a, b), "ordre non déterministe : un shuffle masquerait le drift"


# ── Validité de la ground-truth ───────────────────────────────────────────────

@pytest.mark.parametrize("dataset", sorted(DRIFT_LOADERS))
def test_drift_points_validity(dataset: str) -> None:
    """drift_points = indices valides, triés ; non vides pour la GT ponctuelle, None accepté sinon."""
    if not _raw_available(dataset):
        pytest.skip(f"data/raw absent pour {dataset}")
    d = _load(dataset)
    dp = d.drift_points
    if dataset in NO_POINTWISE_GT:
        assert dp is None or dp == [], "GT non ponctuelle → drift_points None/[] (honnête)"
        return
    assert dp, f"{dataset} devrait exposer des drift_points ponctuels"
    assert all(0 <= p < d.n_samples for p in dp), "indices de drift hors bornes"
    assert dp == sorted(dp), "drift_points doivent être triés (ordre temporel)"


# ── Normalisation figée sur le segment 0 ──────────────────────────────────────

def test_freeze_zscore_fits_on_reference_only() -> None:
    """freeze_zscore ajuste sur le segment 0 seul → segment 0 centré, segment décalé non recentré."""
    rng = np.random.default_rng(42)
    seg0 = rng.normal(loc=0.0, scale=1.0, size=(500, 3)).astype(np.float32)
    seg1 = rng.normal(loc=5.0, scale=1.0, size=(500, 3)).astype(np.float32)  # drift de moyenne
    X = np.vstack([seg0, seg1])

    X_norm, mean, std = freeze_zscore(X, (0, 500))

    # Le segment de référence est bien centré-réduit…
    assert np.allclose(X_norm[:500].mean(axis=0), 0.0, atol=1e-2)
    assert np.allclose(X_norm[:500].std(axis=0), 1.0, atol=1e-2)
    # …mais le segment postérieur reste décalé (le drift n'est PAS masqué par une renorm par segment).
    assert np.all(X_norm[500:].mean(axis=0) > 1.0), "la normalisation figée doit laisser voir le drift"


# ── Vérité-terrain exacte du synthétique ──────────────────────────────────────

def test_synthetic_drift_points_match_config() -> None:
    """Le synthétique retourne exactement les drift_points imposés en config (GT parfaite)."""
    cfg = load_config(DRIFT_CONFIGS["synthetic"])
    expected = [int(p) for p in cfg["data"]["drift_points"]]
    d = _load("synthetic")
    assert d.drift_points == expected
    # Les frontières internes des segments coïncident avec les drift_points.
    from src.data.drift_dataset import segments_to_drift_points

    assert segments_to_drift_points(d.segments) == expected


# ── 0 chiffre de résultat en dur dans le catalogue de figures S4304 ───────────

def test_no_hardcoded_results_drift() -> None:
    """Scan AST de drift_datasets.py : aucun flottant hors liste blanche de layout."""
    src = ROOT / "src/figures/catalogs/drift_datasets.py"
    # Constantes de mise en page autorisées (positions, alpha, largeurs, tailles) — AUCUN résultat.
    layout_whitelist: set[float] = {0.0, 0.005, 0.01, 0.8, 0.88, 1.0, 1.2, 1.6, 2.0, 220.0}
    tree = ast.parse(src.read_text(encoding="utf-8"))
    offending = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, float)
        and node.value not in layout_whitelist
    }
    assert not offending, (
        f"Littéraux flottants suspects dans drift_datasets.py : {sorted(offending)} — "
        "toute valeur tracée doit venir d'un JSON/loader, pas d'un littéral."
    )


# ── Déterminisme / idempotence de la caractérisation ──────────────────────────

def test_characterization_deterministic() -> None:
    """characterize() sur le synthétique deux fois (seed 42) → résultat identique (idempotent)."""
    import json

    from scripts.characterize_drift import characterize

    r1 = characterize("synthetic", DRIFT_CONFIGS["synthetic"])
    r2 = characterize("synthetic", DRIFT_CONFIGS["synthetic"])
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
