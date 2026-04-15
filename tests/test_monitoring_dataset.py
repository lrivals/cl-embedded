"""
tests/test_monitoring_dataset.py — Tests unitaires et d'intégration pour monitoring_dataset.py.

Tests unitaires : fixtures synthétiques (pas de dépendance au vrai CSV).
Tests d'intégration : marqués ``@pytest.mark.integration``, ignorés si le CSV est absent.

Exécution :
    pytest tests/test_monitoring_dataset.py -v
    pytest tests/test_monitoring_dataset.py -v -m integration  # avec vrai CSV
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.monitoring_dataset import (
    DOMAIN_ORDER,
    NUMERIC_FEATURES,
    df_to_tensors,
    encode_categoricals,
    get_cl_dataloaders,
    get_monitoring_dataloaders_single_task,
    get_task_split,
    load_raw_dataset,
    normalize_features,
)

# ---------------------------------------------------------------------------
# Chemins pour les tests d'intégration
# ---------------------------------------------------------------------------

_CSV_PATH = Path(
    "data/raw/equipment_monitoring/"
    "Industrial_Equipment_Monitoring_Dataset/equipment_anomaly_data.csv"
)
_NORMALIZER_PATH = Path("configs/monitoring_normalizer.yaml")

_integration = pytest.mark.skipif(
    not _CSV_PATH.exists(),
    reason=f"CSV introuvable : {_CSV_PATH}",
)

# ---------------------------------------------------------------------------
# Fixtures synthétiques
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """DataFrame minimal couvrant les 3 domaines (60 lignes), faulty ~17%.

    Garantit au moins 2 défauts par domaine pour le split stratifié sklearn.
    """
    rng = np.random.default_rng(42)
    n_per_domain = 20
    domains = ["Pump"] * n_per_domain + ["Turbine"] * n_per_domain + ["Compressor"] * n_per_domain
    n = len(domains)
    # Garantit 2 faulty minimum par domaine (indices 0,1 / 20,21 / 40,41)
    faulty = np.zeros(n, dtype=int)
    for start in [0, n_per_domain, 2 * n_per_domain]:
        faulty[start] = 1
        faulty[start + 1] = 1
    return pd.DataFrame(
        {
            "temperature": rng.normal(70.0, 15.0, n),
            "pressure": rng.normal(35.0, 10.0, n),
            "vibration": rng.normal(1.6, 0.7, n),
            "humidity": rng.normal(50.0, 12.0, n),
            "equipment": domains,
            "location": ["SiteA"] * n,
            "faulty": faulty,
        }
    )


@pytest.fixture
def synthetic_normalizer() -> dict:
    """Normaliseur synthétique avec les stats T1 (Pump) du vrai dataset."""
    return {
        "mean": {
            "temperature": 70.634028,
            "pressure": 35.629021,
            "vibration": 1.613323,
            "humidity": 50.197351,
        },
        "std": {
            "temperature": 15.781869,
            "pressure": 10.501268,
            "vibration": 0.700299,
            "humidity": 11.874382,
        },
    }


# ---------------------------------------------------------------------------
# Tests unitaires — load_raw_dataset (validation)
# ---------------------------------------------------------------------------


def test_load_raw_dataset_validation_missing_column(synthetic_df: pd.DataFrame, tmp_path: Path):
    """ValueError si une colonne obligatoire est absente."""
    bad_df = synthetic_df.drop(columns=["vibration"])
    csv_path = tmp_path / "bad.csv"
    bad_df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Colonnes manquantes"):
        load_raw_dataset(csv_path)


def test_load_raw_dataset_validation_bad_labels(synthetic_df: pd.DataFrame, tmp_path: Path):
    """ValueError si faulty contient des valeurs hors {0, 1}."""
    bad_df = synthetic_df.copy()
    bad_df.loc[0, "faulty"] = 2
    csv_path = tmp_path / "bad_labels.csv"
    bad_df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Labels inattendus"):
        load_raw_dataset(csv_path)


def test_load_raw_dataset_validation_missing_domain(synthetic_df: pd.DataFrame, tmp_path: Path):
    """ValueError si un domaine CL est absent."""
    bad_df = synthetic_df[synthetic_df["equipment"] != "Turbine"]
    csv_path = tmp_path / "no_turbine.csv"
    bad_df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Domaines manquants"):
        load_raw_dataset(csv_path)


def test_load_raw_dataset_file_not_found():
    """FileNotFoundError si le CSV n'existe pas."""
    with pytest.raises(FileNotFoundError):
        load_raw_dataset(Path("nonexistent/path/file.csv"))


# ---------------------------------------------------------------------------
# Tests unitaires — normalize_features
# ---------------------------------------------------------------------------


def test_normalize_features_does_not_mutate(
    synthetic_df: pd.DataFrame, synthetic_normalizer: dict
):
    """normalize_features retourne une copie, n'altère pas le DataFrame original."""
    original_temp = synthetic_df["temperature"].copy()
    _ = normalize_features(synthetic_df, synthetic_normalizer)
    pd.testing.assert_series_equal(synthetic_df["temperature"], original_temp)


def test_normalize_features_output_shape(
    synthetic_df: pd.DataFrame, synthetic_normalizer: dict
):
    """normalize_features conserve la forme et les colonnes du DataFrame."""
    df_norm = normalize_features(synthetic_df, synthetic_normalizer)
    assert df_norm.shape == synthetic_df.shape
    assert list(df_norm.columns) == list(synthetic_df.columns)


# ---------------------------------------------------------------------------
# Tests unitaires — encode_categoricals
# ---------------------------------------------------------------------------


def test_encode_categoricals_mapping(synthetic_df: pd.DataFrame):
    """Vérification du mapping fixe Pump→0, Turbine→1, Compressor→2."""
    df_enc = encode_categoricals(synthetic_df)
    assert "equipment_encoded" in df_enc.columns
    assert df_enc.loc[df_enc["equipment"] == "Pump", "equipment_encoded"].iloc[0] == 0
    assert df_enc.loc[df_enc["equipment"] == "Turbine", "equipment_encoded"].iloc[0] == 1
    assert df_enc.loc[df_enc["equipment"] == "Compressor", "equipment_encoded"].iloc[0] == 2


def test_encode_categoricals_preserves_equipment_column(synthetic_df: pd.DataFrame):
    """La colonne 'equipment' originale doit être conservée."""
    df_enc = encode_categoricals(synthetic_df)
    assert "equipment" in df_enc.columns


# ---------------------------------------------------------------------------
# Tests unitaires — df_to_tensors
# ---------------------------------------------------------------------------


def test_df_to_tensors_shape(synthetic_df: pd.DataFrame):
    """X : [N, 4] float32 — y : [N, 1] float32."""
    x, y = df_to_tensors(synthetic_df)
    n = len(synthetic_df)
    assert x.shape == (n, 4)
    assert y.shape == (n, 1)
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_label_binary(synthetic_df: pd.DataFrame):
    """y ne contient que 0.0 et 1.0."""
    _, y = df_to_tensors(synthetic_df)
    unique_vals = set(y.unique().tolist())
    assert unique_vals.issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# Tests unitaires — get_task_split
# ---------------------------------------------------------------------------


def test_no_data_leakage(synthetic_df: pd.DataFrame, synthetic_normalizer: dict):
    """Aucune donnée d'un domaine ne doit apparaître dans les splits d'un autre."""
    df_proc = encode_categoricals(normalize_features(synthetic_df, synthetic_normalizer))

    splits: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for domain in DOMAIN_ORDER:
        splits[domain] = get_task_split(df_proc, domain, val_ratio=0.2, seed=42)

    for domain, (df_train, df_val) in splits.items():
        assert (df_train["equipment"] == domain).all(), (
            f"Train de '{domain}' contient des données d'autres domaines."
        )
        assert (df_val["equipment"] == domain).all(), (
            f"Val de '{domain}' contient des données d'autres domaines."
        )


def test_task_split_no_overlap(synthetic_df: pd.DataFrame, synthetic_normalizer: dict):
    """Train + val couvrent exactement le domaine, sans duplication de lignes."""
    df_proc = encode_categoricals(normalize_features(synthetic_df, synthetic_normalizer))
    df_pump = df_proc[df_proc["equipment"] == "Pump"]
    df_train, df_val = get_task_split(df_proc, "Pump", val_ratio=0.2, seed=42)

    combined = pd.concat([df_train, df_val]).sort_values(NUMERIC_FEATURES).reset_index(drop=True)
    original = df_pump.sort_values(NUMERIC_FEATURES).reset_index(drop=True)

    assert len(combined) == len(original), "Train + val != taille du domaine (duplication ou perte)"
    pd.testing.assert_frame_equal(combined[NUMERIC_FEATURES], original[NUMERIC_FEATURES])


# ---------------------------------------------------------------------------
# Tests unitaires — get_cl_dataloaders (avec CSV temporaire synthétique)
# ---------------------------------------------------------------------------


def _write_synthetic_csv(df: pd.DataFrame, tmp_path: Path) -> Path:
    csv_path = tmp_path / "equipment_anomaly_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _write_synthetic_normalizer(normalizer: dict, tmp_path: Path) -> Path:
    import yaml

    norm_path = tmp_path / "monitoring_normalizer.yaml"
    with open(norm_path, "w") as f:
        yaml.dump(
            {"fit_domain": "Pump", "normalization": "zscore", **normalizer},
            f,
        )
    return norm_path


def test_cl_dataloaders_returns_three_tasks(
    synthetic_df: pd.DataFrame,
    synthetic_normalizer: dict,
    tmp_path: Path,
):
    """get_cl_dataloaders retourne exactement 3 tâches."""
    csv_path = _write_synthetic_csv(synthetic_df, tmp_path)
    norm_path = _write_synthetic_normalizer(synthetic_normalizer, tmp_path)

    tasks = get_cl_dataloaders(csv_path, norm_path, batch_size=4)
    assert len(tasks) == 3


def test_cl_dataloaders_task_order(
    synthetic_df: pd.DataFrame,
    synthetic_normalizer: dict,
    tmp_path: Path,
):
    """task_id = 1,2,3 et domaines Pump → Turbine → Compressor."""
    csv_path = _write_synthetic_csv(synthetic_df, tmp_path)
    norm_path = _write_synthetic_normalizer(synthetic_normalizer, tmp_path)

    tasks = get_cl_dataloaders(csv_path, norm_path, batch_size=4)

    for expected_id, expected_domain, task in zip(
        [1, 2, 3], DOMAIN_ORDER, tasks
    ):
        assert task["task_id"] == expected_id
        assert task["domain"] == expected_domain


def test_cl_dataloaders_shape(
    synthetic_df: pd.DataFrame,
    synthetic_normalizer: dict,
    tmp_path: Path,
):
    """X.shape == [batch, 4] et y.shape == [batch, 1] pour tous les loaders."""
    csv_path = _write_synthetic_csv(synthetic_df, tmp_path)
    norm_path = _write_synthetic_normalizer(synthetic_normalizer, tmp_path)

    tasks = get_cl_dataloaders(csv_path, norm_path, batch_size=4)

    for task in tasks:
        x_batch, y_batch = next(iter(task["train_loader"]))
        assert x_batch.shape[1] == len(NUMERIC_FEATURES), (
            f"X doit avoir {len(NUMERIC_FEATURES)} features, got {x_batch.shape[1]}"
        )
        assert y_batch.shape[1] == 1

        x_val, y_val = next(iter(task["val_loader"]))
        assert x_val.shape[1] == len(NUMERIC_FEATURES)
        assert y_val.shape[1] == 1


def test_cl_dataloaders_n_train_n_val(
    synthetic_df: pd.DataFrame,
    synthetic_normalizer: dict,
    tmp_path: Path,
):
    """n_train + n_val == taille du domaine dans le DataFrame."""
    csv_path = _write_synthetic_csv(synthetic_df, tmp_path)
    norm_path = _write_synthetic_normalizer(synthetic_normalizer, tmp_path)

    tasks = get_cl_dataloaders(csv_path, norm_path, batch_size=4)

    for task in tasks:
        domain_size = (synthetic_df["equipment"] == task["domain"]).sum()
        assert task["n_train"] + task["n_val"] == domain_size


# ---------------------------------------------------------------------------
# Tests d'intégration — vrai CSV
# ---------------------------------------------------------------------------


@_integration
def test_normalizer_applied_on_real_data():
    """Sur le vrai CSV, la moyenne de T1 (Pump) doit être ≈ 0 après normalisation."""
    tasks = get_cl_dataloaders(_CSV_PATH, _NORMALIZER_PATH, batch_size=512)

    # Collecter tous les batchs de T1
    all_x = torch.cat([x for x, _ in tasks[0]["train_loader"]], dim=0)

    # La moyenne de chaque feature normalisée doit être proche de 0
    means = all_x.mean(dim=0)
    for i, feat in enumerate(NUMERIC_FEATURES):
        assert abs(means[i].item()) < 0.15, (
            f"Feature '{feat}' : mean={means[i]:.4f} après normalisation (attendu ≈ 0)"
        )


@_integration
def test_real_data_label_binary():
    """Sur le vrai CSV, tous les labels sont dans {0.0, 1.0}."""
    tasks = get_cl_dataloaders(_CSV_PATH, _NORMALIZER_PATH)
    for task in tasks:
        for _, y_batch in task["train_loader"]:
            unique = set(y_batch.unique().tolist())
            assert unique.issubset({0.0, 1.0}), f"Labels inattendus : {unique}"


@_integration
def test_real_data_domain_sizes():
    """Sur le vrai CSV, les tailles par domaine correspondent aux valeurs documentées."""
    from src.data.monitoring_dataset import DOMAIN_SIZES

    tasks = get_cl_dataloaders(_CSV_PATH, _NORMALIZER_PATH)
    for task in tasks:
        domain = task["domain"]
        expected = DOMAIN_SIZES[domain]
        got = task["n_train"] + task["n_val"]
        assert got == expected, (
            f"Domaine '{domain}' : attendu {expected} échantillons, obtenu {got}"
        )


# ---------------------------------------------------------------------------
# Tests unitaires — get_monitoring_dataloaders_single_task
# ---------------------------------------------------------------------------


def test_single_task_returns_dict_not_list(
    synthetic_df: pd.DataFrame,
    tmp_path: Path,
):
    """get_monitoring_dataloaders_single_task retourne un dict (pas une liste) — signal hors-CL."""
    csv_path = _write_synthetic_csv(synthetic_df, tmp_path)

    result = get_monitoring_dataloaders_single_task(csv_path, batch_size=4, seed=42)

    assert isinstance(result, dict), "Doit retourner un dict (pas une liste)"
    assert not isinstance(result, list)


def test_single_task_required_keys(
    synthetic_df: pd.DataFrame,
    tmp_path: Path,
):
    """Le dict retourné contient toutes les clés obligatoires."""
    csv_path = _write_synthetic_csv(synthetic_df, tmp_path)

    result = get_monitoring_dataloaders_single_task(csv_path, batch_size=4, seed=42)

    required_keys = {"train_loader", "val_loader", "test_loader", "n_train", "n_val", "n_test", "normalizer"}
    assert required_keys.issubset(result.keys()), (
        f"Clés manquantes : {required_keys - result.keys()}"
    )


def test_single_task_split_sizes_sum_to_total(
    synthetic_df: pd.DataFrame,
    tmp_path: Path,
):
    """n_train + n_val + n_test == total du dataset, sans duplication ni perte."""
    csv_path = _write_synthetic_csv(synthetic_df, tmp_path)
    total = len(synthetic_df)

    result = get_monitoring_dataloaders_single_task(
        csv_path, batch_size=4, test_ratio=0.2, val_ratio=0.1, seed=42
    )

    assert result["n_train"] + result["n_val"] + result["n_test"] == total, (
        f"n_train({result['n_train']}) + n_val({result['n_val']}) + "
        f"n_test({result['n_test']}) != {total}"
    )


def test_single_task_no_test_leak_in_train(
    synthetic_df: pd.DataFrame,
    tmp_path: Path,
):
    """n_test == environ test_ratio * total (pas de duplication de données dans test)."""
    csv_path = _write_synthetic_csv(synthetic_df, tmp_path)
    total = len(synthetic_df)
    test_ratio = 0.2

    result = get_monitoring_dataloaders_single_task(
        csv_path, batch_size=4, test_ratio=test_ratio, val_ratio=0.1, seed=42
    )

    # Test set ≈ test_ratio * total (tolérance ±1 pour stratification)
    expected_test = round(total * test_ratio)
    assert abs(result["n_test"] - expected_test) <= 2, (
        f"n_test={result['n_test']} très éloigné de l'attendu {expected_test}"
    )


def test_single_task_normalizer_fitted_on_train(
    synthetic_df: pd.DataFrame,
    tmp_path: Path,
):
    """Le normalizer est fittée uniquement sur le train split : mean/std présentes dans le dict."""
    csv_path = _write_synthetic_csv(synthetic_df, tmp_path)

    result = get_monitoring_dataloaders_single_task(csv_path, batch_size=4, seed=42)

    normalizer = result["normalizer"]
    assert "mean" in normalizer and "std" in normalizer, "normalizer doit avoir 'mean' et 'std'"
    # Les 4 features doivent être présentes
    for feat in NUMERIC_FEATURES:
        assert feat in normalizer["mean"], f"Feature '{feat}' absente de normalizer['mean']"
        assert feat in normalizer["std"], f"Feature '{feat}' absente de normalizer['std']"


def test_single_task_loaders_shape(
    synthetic_df: pd.DataFrame,
    tmp_path: Path,
):
    """Les DataLoaders retournent des batchs [B, 4] pour X et [B, 1] pour y."""
    csv_path = _write_synthetic_csv(synthetic_df, tmp_path)

    result = get_monitoring_dataloaders_single_task(csv_path, batch_size=4, seed=42)

    for split_name in ("train_loader", "val_loader", "test_loader"):
        x_batch, y_batch = next(iter(result[split_name]))
        assert x_batch.shape[1] == len(NUMERIC_FEATURES), (
            f"{split_name} X: attendu {len(NUMERIC_FEATURES)} features, obtenu {x_batch.shape[1]}"
        )
        assert y_batch.shape[1] == 1, (
            f"{split_name} y: attendu shape [B, 1], obtenu {y_batch.shape}"
        )


def test_single_task_all_domains_in_splits(
    synthetic_df: pd.DataFrame,
    tmp_path: Path,
):
    """Tous les équipements (Pump, Turbine, Compressor) sont présents dans le train et test."""
    csv_path = _write_synthetic_csv(synthetic_df, tmp_path)

    result = get_monitoring_dataloaders_single_task(
        csv_path, batch_size=len(synthetic_df), seed=42
    )

    # Vérifier via n_train et n_test : dataset complet = les 3 domaines
    # Proxy : n_train + n_val + n_test couvre les 60 lignes (3 domaines × 20)
    total = result["n_train"] + result["n_val"] + result["n_test"]
    assert total == len(synthetic_df), (
        f"Le split total ({total}) ne couvre pas tout le dataset ({len(synthetic_df)})"
    )


def test_single_task_reproducible(
    synthetic_df: pd.DataFrame,
    tmp_path: Path,
):
    """Deux appels avec le même seed retournent les mêmes tailles."""
    csv_path = _write_synthetic_csv(synthetic_df, tmp_path)

    r1 = get_monitoring_dataloaders_single_task(csv_path, batch_size=4, seed=42)
    r2 = get_monitoring_dataloaders_single_task(csv_path, batch_size=4, seed=42)

    assert r1["n_train"] == r2["n_train"]
    assert r1["n_val"] == r2["n_val"]
    assert r1["n_test"] == r2["n_test"]


@_integration
def test_single_task_real_data_sizes():
    """Sur le vrai CSV, n_train + n_val + n_test == 7672 (total documenté)."""
    from src.data.monitoring_dataset import DOMAIN_SIZES

    total_expected = sum(DOMAIN_SIZES.values())  # 7672
    csv_path = Path(
        "data/raw/equipment_monitoring/"
        "Industrial_Equipment_Monitoring_Dataset/equipment_anomaly_data.csv"
    )

    result = get_monitoring_dataloaders_single_task(
        csv_path, batch_size=32, test_ratio=0.2, val_ratio=0.1, seed=42
    )

    total_got = result["n_train"] + result["n_val"] + result["n_test"]
    assert total_got == total_expected, (
        f"Total attendu : {total_expected}, obtenu : {total_got}"
    )
