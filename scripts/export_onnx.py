"""
scripts/export_onnx.py — Export ONNX + quantification PTQ INT8 pour EWC, TinyOL, Mahalanobis.

Pipeline :
    PyTorch .pt → torch.onnx.export → .onnx → onnxruntime quantize_dynamic → _int8.onnx
    Mahalanobis → graphe ONNX manuel (Sub/MatMul/Transpose/Sqrt)
    HDC → skip (accumulation additive non différentiable, documenté dans manifest)

Usage :
    # Export simple (rétrocompatibilité Sprint 4)
    python scripts/export_onnx.py --model ewc --config configs/ewc_config.yaml \\
        --output experiments/exp_160/ewc_backbone.onnx

    # Export modèle × dataset spécifique (S2403)
    python scripts/export_onnx.py --model ewc --dataset cwru \\
        --output_dir experiments/onnx_sprint24/

    # Export systématique complet (S2403b)
    python scripts/export_onnx.py --all --output_dir experiments/onnx_sprint24/

Références : S1002 — docs/sprints/sprint_phase2/S1002_onnx_export.md
             S2403 — docs/sprints/sprint_24/S2403_onnx_multi_dataset.md
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic

from src.models.ewc.ewc_mlp import EWCMlpClassifier
from src.models.tinyol.autoencoder import TinyOLAutoencoder
from src.utils.config_loader import load_config

# Opset ONNX compatible STM32Cube.AI NeuralART Turbo — TODO(dorra): confirmer opset optimal
OPSET_VERSION: int = 17

# ---------------------------------------------------------------------------
# Grille multi-dataset (S2403)
# ---------------------------------------------------------------------------

SUPPORTED_DATASETS = ["monitoring", "pump", "cwru", "pronostia", "cmapss", "paderborn"]
SUPPORTED_MODELS = ["ewc", "hdc", "tinyol", "mahalanobis"]

# (model, dataset) → (config_path, input_dim)
MODEL_DATASET_MAP: dict[tuple[str, str], tuple[str, int]] = {
    ("ewc", "monitoring"):   ("configs/ewc_config.yaml",                         4),
    ("ewc", "pump"):         ("configs/ewc_pump_config.yaml",                    25),
    ("ewc", "cwru"):         ("configs/cwru_by_fault_config.yaml",                9),
    ("ewc", "pronostia"):    ("configs/ewc_pronostia_by_condition_config.yaml",  13),
    ("ewc", "cmapss"):       ("configs/cmapss_config.yaml",                       5),
    ("ewc", "paderborn"):    ("configs/paderborn_config.yaml",                    5),
    ("hdc", "monitoring"):   ("configs/hdc_config.yaml",                          4),
    ("hdc", "cwru"):         ("configs/cwru_by_fault_config.yaml",                9),
    ("hdc", "pump"):         ("configs/hdc_pump_config.yaml",                    25),
    ("hdc", "cmapss"):       ("configs/cmapss_config.yaml",                       5),
    ("hdc", "paderborn"):    ("configs/paderborn_config.yaml",                    5),
    ("tinyol", "monitoring"):("configs/tinyol_monitoring_config.yaml",            4),
    ("tinyol", "pump"):      ("configs/tinyol_config.yaml",                      25),
    ("tinyol", "cwru"):      ("configs/cwru_by_fault_config.yaml",                9),
    ("mahalanobis", "monitoring"):("configs/ewc_config.yaml",                     4),
    ("mahalanobis", "pump"):      ("configs/ewc_pump_config.yaml",               25),
    ("mahalanobis", "cwru"):      ("configs/cwru_by_fault_config.yaml",           9),
    ("mahalanobis", "pronostia"): ("configs/pronostia_config.yaml",              13),
    ("mahalanobis", "cmapss"):    ("configs/cmapss_config.yaml",                  5),
    ("mahalanobis", "paderborn"): ("configs/paderborn_config.yaml",               5),
}

SKIP_COMBOS: dict[tuple[str, str], str] = {
    ("hdc", "pronostia"):    "pas d'expérience HDC correspondante sur Pronostia",
    ("tinyol", "cmapss"):    "pas de loader temporel approprié pour CMAPSS",
    ("tinyol", "paderborn"): "pas de loader temporel approprié pour Paderborn",
    ("tinyol", "pronostia"): "pas d'expérience TinyOL correspondante sur Pronostia",
}


# ---------------------------------------------------------------------------
# Wrapper pour exporter uniquement l'encodeur TinyOL
# ---------------------------------------------------------------------------

class _TinyOLEncoderWrapper(nn.Module):
    """Expose encode() comme forward() pour l'export ONNX du backbone TinyOL."""

    def __init__(self, autoencoder: TinyOLAutoencoder) -> None:
        super().__init__()
        self.enc1 = autoencoder.enc1
        self.enc2 = autoencoder.enc2
        self.enc3 = autoencoder.enc3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.enc1(x))  # MEM: 32 × 4 B = 128 B @ FP32 (batch=1)
        z = torch.relu(self.enc2(z))  # MEM: 16 × 4 B = 64 B @ FP32 (batch=1)
        z = self.enc3(z)              # MEM: 8 × 4 B = 32 B @ FP32 (batch=1)
        return z


# ---------------------------------------------------------------------------
# Fonctions d'export — modèles individuels (rétrocompatibilité Sprint 4)
# ---------------------------------------------------------------------------

def export_ewc_backbone(
    config_path: str,
    output_path: str,
    checkpoint_path: str | None = None,
) -> dict:
    """
    Exporte le backbone EWC-MLP vers ONNX et le quantifie en INT8.

    Parameters
    ----------
    config_path : str
        Chemin vers ewc_config.yaml.
    output_path : str
        Chemin de sortie pour le fichier .onnx FP32.
    checkpoint_path : str, optional
        Chemin vers un checkpoint PyTorch (.pt). Si absent, poids aléatoires.

    Returns
    -------
    dict : résumé de l'export (chemins, métriques de validation).
    """
    cfg = load_config(config_path)
    input_dim: int = cfg["model"]["input_dim"]
    hidden_dims: list[int] = cfg["model"]["hidden_dims"]

    model = EWCMlpClassifier(input_dim=input_dim, hidden_dims=hidden_dims, dropout=0.0)

    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")
        model.load_state(checkpoint_path)
        print(f"[EWC] Checkpoint chargé : {checkpoint_path}")
    else:
        warnings.warn(
            "Aucun checkpoint fourni — poids aléatoires. "
            "Le graphe ONNX est valide mais les sorties sont sans signification.",
            stacklevel=2,
        )

    return _export_and_quantize(
        model=model,
        dummy_input=torch.zeros(1, input_dim),
        output_path=output_path,
        model_name="ewc_backbone",
        input_dim=input_dim,
    )


def export_tinyol_encoder(
    config_path: str,
    output_path: str,
    checkpoint_path: str | None = None,
) -> dict:
    """
    Exporte l'encodeur TinyOL (backbone frozen) vers ONNX et le quantifie en INT8.

    Parameters
    ----------
    config_path : str
        Chemin vers tinyol_config.yaml.
    output_path : str
        Chemin de sortie pour le fichier .onnx FP32.
    checkpoint_path : str, optional
        Chemin vers un checkpoint PyTorch (.pt). Si absent, poids aléatoires.

    Returns
    -------
    dict : résumé de l'export.
    """
    cfg = load_config(config_path)
    input_dim: int = cfg["backbone"]["input_dim"]
    encoder_dims: tuple[int, ...] = tuple(cfg["backbone"]["encoder_dims"])
    decoder_dims: tuple[int, ...] = tuple(cfg["backbone"]["decoder_dims"])

    # Si un checkpoint est fourni, inférer l'architecture réelle depuis les poids.
    # Priorité au checkpoint sur le config (le config peut référencer un autre dataset).
    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        # enc1.weight shape = [encoder_dims[0], actual_input_dim]
        if "enc1.weight" in state_dict:
            actual_input_dim = state_dict["enc1.weight"].shape[1]
            if actual_input_dim != input_dim:
                print(
                    f"[TinyOL] input_dim inféré depuis checkpoint : {actual_input_dim} "
                    f"(config indique {input_dim}) — utilisation de la valeur checkpoint."
                )
                input_dim = actual_input_dim
        # dec3.weight shape = [actual_input_dim, decoder_dims[-1]]
        if "dec3.bias" in state_dict:
            actual_out_dim = state_dict["dec3.bias"].shape[0]
            if actual_out_dim != decoder_dims[-1]:
                decoder_dims = decoder_dims[:-1] + (actual_out_dim,)
    else:
        state_dict = None
        warnings.warn(
            "Aucun checkpoint fourni — poids aléatoires. "
            "Le graphe ONNX est valide mais les sorties sont sans signification.",
            stacklevel=2,
        )

    autoencoder = TinyOLAutoencoder(
        input_dim=input_dim,
        encoder_dims=encoder_dims,
        decoder_dims=decoder_dims,
    )

    if state_dict is not None:
        autoencoder.load_state_dict(state_dict)
        print(f"[TinyOL] Checkpoint chargé : {checkpoint_path}")

    encoder_wrapper = _TinyOLEncoderWrapper(autoencoder)

    return _export_and_quantize(
        model=encoder_wrapper,
        dummy_input=torch.zeros(1, input_dim),
        output_path=output_path,
        model_name="tinyol_encoder",
        input_dim=input_dim,
    )


# ---------------------------------------------------------------------------
# Fonctions d'export multi-dataset (S2403)
# ---------------------------------------------------------------------------

def export_ewc_for_dataset(dataset: str, output_path: str) -> dict:
    """
    Exporte EWC-MLP pour un dataset donné en lisant la config depuis MODEL_DATASET_MAP.

    Poids aléatoires (pas de checkpoint requis) — le graphe ONNX est structurellement valide.
    """
    key = ("ewc", dataset)
    if key not in MODEL_DATASET_MAP:
        raise ValueError(f"Combo (ewc, {dataset}) non supporté.")
    config_path, input_dim = MODEL_DATASET_MAP[key]

    cfg = load_config(config_path)
    hidden_dims: list[int] = cfg["model"]["hidden_dims"]

    model = EWCMlpClassifier(input_dim=input_dim, hidden_dims=hidden_dims, dropout=0.0)
    warnings.warn(
        f"[EWC/{dataset}] Poids aléatoires — graphe ONNX valide, sorties sans signification.",
        stacklevel=2,
    )

    result = _export_and_quantize(
        model=model,
        dummy_input=torch.zeros(1, input_dim),
        output_path=output_path,
        model_name=f"ewc_{dataset}",
        input_dim=input_dim,
    )
    result["dataset"] = dataset
    result["model_type"] = "ewc"
    return result


def export_tinyol_for_dataset(dataset: str, output_path: str) -> dict:
    """
    Exporte l'encodeur TinyOL pour un dataset donné depuis MODEL_DATASET_MAP.

    Poids aléatoires (pas de checkpoint requis).
    """
    key = ("tinyol", dataset)
    if key not in MODEL_DATASET_MAP:
        raise ValueError(f"Combo (tinyol, {dataset}) non supporté.")
    config_path, input_dim = MODEL_DATASET_MAP[key]

    cfg = load_config(config_path)
    encoder_dims: tuple[int, ...] = tuple(cfg["backbone"]["encoder_dims"])
    decoder_dims: tuple[int, ...] = tuple(cfg["backbone"]["decoder_dims"])

    autoencoder = TinyOLAutoencoder(
        input_dim=input_dim,
        encoder_dims=encoder_dims,
        decoder_dims=decoder_dims,
    )
    warnings.warn(
        f"[TinyOL/{dataset}] Poids aléatoires — graphe ONNX valide, sorties sans signification.",
        stacklevel=2,
    )

    encoder_wrapper = _TinyOLEncoderWrapper(autoencoder)

    result = _export_and_quantize(
        model=encoder_wrapper,
        dummy_input=torch.zeros(1, input_dim),
        output_path=output_path,
        model_name=f"tinyol_{dataset}",
        input_dim=input_dim,
    )
    result["dataset"] = dataset
    result["model_type"] = "tinyol"
    return result


def export_mahalanobis_onnx(dataset: str, output_path: str) -> dict:
    """
    Exporte la distance de Mahalanobis comme graphe ONNX manuel.

    Opérateurs utilisés : Sub, MatMul, Transpose, Sqrt — tous compatibles CMSIS-NN.
    μ et Σ⁻¹ sont des Initializer avec des valeurs aléatoires cohérentes avec input_dim.
    Fallback joblib si onnx.checker échoue.

    Returns
    -------
    dict avec status 'exported' ou 'fallback_joblib'.
    """
    key = ("mahalanobis", dataset)
    if key not in MODEL_DATASET_MAP:
        raise ValueError(f"Combo (mahalanobis, {dataset}) non supporté.")
    _, input_dim = MODEL_DATASET_MAP[key]

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    mu = rng.standard_normal(input_dim).astype(np.float32)
    # Σ⁻¹ symétrique définie positive ← I + ε·Aᵀ·A
    A = rng.standard_normal((input_dim, input_dim)).astype(np.float32) * 0.1
    sigma_inv = np.eye(input_dim, dtype=np.float32) + A.T @ A

    mu_init = numpy_helper.from_array(mu, name="mu")
    sigma_inv_init = numpy_helper.from_array(sigma_inv, name="sigma_inv")

    # Graphe :
    #   diff       = input - mu          [1, d]
    #   maha_vec   = diff @ sigma_inv    [1, d]
    #   diff_T     = Transpose(diff)     [d, 1]
    #   maha2      = maha_vec @ diff_T   [1, 1]  (distance² de Mahalanobis)
    #   output     = Sqrt(maha2)         [1, 1]
    nodes = [
        helper.make_node("Sub",       ["input", "mu"],           ["diff"]),
        helper.make_node("MatMul",    ["diff", "sigma_inv"],     ["maha_vec"]),
        helper.make_node("Transpose", ["diff"],                  ["diff_T"],   perm=[1, 0]),
        helper.make_node("MatMul",    ["maha_vec", "diff_T"],   ["maha2"]),
        helper.make_node("Sqrt",      ["maha2"],                 ["output"]),
    ]

    input_vi  = helper.make_tensor_value_info("input",  TensorProto.FLOAT, [1, input_dim])
    output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])

    graph = helper.make_graph(
        nodes, f"mahalanobis_{dataset}",
        [input_vi], [output_vi],
        [mu_init, sigma_inv_init],
    )
    model_proto = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET_VERSION)],
    )
    model_proto.ir_version = 8

    try:
        onnx.checker.check_model(model_proto)
        onnx.save(model_proto, str(out_path))
        print(f"[Mahalanobis/{dataset}] ONNX écrit : {out_path}")

        # Validation onnxruntime
        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        x_test = np.random.randn(1, input_dim).astype(np.float32)
        ort_out = sess.run(None, {"input": x_test})[0]
        validated = ort_out.shape == (1, 1)
        print(f"[Mahalanobis/{dataset}] onnxruntime → shape {ort_out.shape} | validated={validated}")

        n_params = input_dim + input_dim * input_dim  # mu + sigma_inv
        return {
            "model_type":    "mahalanobis",
            "dataset":       dataset,
            "filename":      out_path.name,
            "onnx_path":     str(out_path),
            "input_dim":     input_dim,
            "n_params":      n_params,
            "file_size_kb":  round(out_path.stat().st_size / 1024, 2),
            "onnx_opset":    OPSET_VERSION,
            "validated":     validated,
            "onnx_valid":    True,
            "status":        "exported",
            "operators":     ["Sub", "MatMul", "Transpose", "Sqrt"],
        }

    except Exception as exc:
        print(f"[Mahalanobis/{dataset}] Export ONNX échoué ({exc}) — fallback joblib")
        try:
            import joblib
            from src.models.unsupervised import MahalanobisDetector
            joblib_path = out_path.with_suffix(".joblib")
            dummy_model = MahalanobisDetector({})
            dummy_model.mu_ = mu
            dummy_model.sigma_inv_ = sigma_inv
            joblib.dump(dummy_model, str(joblib_path))
            print(f"[Mahalanobis/{dataset}] Sauvegardé en joblib : {joblib_path}")
        except Exception:
            pass
        return {
            "model_type": "mahalanobis",
            "dataset":    dataset,
            "status":     "fallback_joblib",
            "reason":     str(exc),
            "validated":  False,
        }


def export_hdc_for_dataset(dataset: str, _output_path: str) -> dict:
    """
    HDC n'est pas un réseau de neurones différentiable — export ONNX non applicable.

    HDC utilise une accumulation additive de prototypes (prototypical vectors).
    Opérations équivalentes : MatMul (encodage) + ArgMax (inférence), manuellement portables.
    """
    print(f"[HDC/{dataset}] Skip — accumulation additive non exportable en ONNX.")
    return {
        "model_type": "hdc",
        "dataset":    dataset,
        "status":     "skipped",
        "reason":     (
            "HDC utilise une accumulation additive de prototypes INT32 sans graphe "
            "computationnel différentiable. Export ONNX non applicable. "
            "Opérations équivalentes : MatMul (encodage) + ArgMax (inférence)."
        ),
        "pseudo_onnx_ops":    ["MatMul", "ArgMax"],
        "cmsis_nn_compatible": True,
    }


# ---------------------------------------------------------------------------
# Manifest JSON (S2403b)
# ---------------------------------------------------------------------------

def generate_manifest(entries: list[dict], output_dir: str) -> str:
    """
    Génère onnx_manifest.json dans output_dir.

    Parameters
    ----------
    entries : list[dict]
        Résultats de chaque export (exportés + skippés).
    output_dir : str
        Répertoire de sortie.

    Returns
    -------
    str : chemin du fichier manifest.
    """
    manifest_path = Path(output_dir) / "onnx_manifest.json"
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "sprint": 24,
        "onnx_opset": OPSET_VERSION,
        "models": [],
    }

    for e in entries:
        if e.get("status") in ("skipped",):
            manifest["models"].append({
                "model":   e.get("model_type", "?"),
                "dataset": e.get("dataset", "?"),
                "status":  "skipped",
                "reason":  e.get("reason", ""),
            })
        elif e.get("status") == "fallback_joblib":
            manifest["models"].append({
                "model":   e.get("model_type", "?"),
                "dataset": e.get("dataset", "?"),
                "status":  "fallback_joblib",
                "reason":  e.get("reason", ""),
            })
        else:
            manifest["models"].append({
                "filename":    e.get("filename") or Path(e.get("onnx_path", "")).name,
                "model":       e.get("model_type", e.get("model_name", "?")),
                "dataset":     e.get("dataset", "?"),
                "input_dim":   e.get("input_dim", 0),
                "n_params":    e.get("n_params", 0),
                "file_size_kb":e.get("file_size_kb") or round(
                    e.get("onnx_fp32_bytes", 0) / 1024, 2
                ),
                "onnx_opset":  e.get("opset_version", OPSET_VERSION),
                "validated":   e.get("onnx_valid", False) and e.get("ort_match_atol_1e5", False),
                "status":      "exported",
            })

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest ONNX → {manifest_path}")
    return str(manifest_path)


# ---------------------------------------------------------------------------
# Export commun : torch.onnx → checker → onnxruntime → quantize_dynamic
# ---------------------------------------------------------------------------

def _export_and_quantize(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: str,
    model_name: str,
    input_dim: int,
) -> dict:
    """
    Exporte un nn.Module vers ONNX, le valide, le compare avec OnnxRuntime
    et produit une version quantifiée INT8.

    Returns
    -------
    dict avec les clés :
        onnx_path, int8_path, n_params, onnx_valid, ort_match_atol,
        max_abs_diff, mean_abs_diff
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    int8_path = out_path.parent / (out_path.stem + "_int8.onnx")

    model.eval()

    # --- 1. Export torch → ONNX ---
    print(f"\n[{model_name}] Export ONNX → {out_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(out_path),
            opset_version=OPSET_VERSION,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=None,  # batch=1 fixe (MCU)
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"[{model_name}] Fichier ONNX écrit : {out_path} ({out_path.stat().st_size} B)")

    # --- 2. Vérification onnx.checker ---
    onnx_model = onnx.load(str(out_path))
    try:
        onnx.checker.check_model(onnx_model)
        onnx_valid = True
        print(f"[{model_name}] onnx.checker.check_model → OK")
    except onnx.checker.ValidationError as e:
        onnx_valid = False
        print(f"[{model_name}] onnx.checker.check_model → ERREUR : {e}")

    # --- 3. Vérification onnxruntime vs PyTorch ---
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    np.random.seed(42)
    test_inputs = np.random.randn(100, input_dim).astype(np.float32)
    abs_diffs: list[float] = []

    with torch.no_grad():
        for x_np in test_inputs:
            x_t = torch.from_numpy(x_np[None])  # [1, input_dim]
            pt_out = model(x_t).numpy()
            ort_out = sess.run(None, {input_name: x_np[None]})[0]
            abs_diffs.append(float(np.max(np.abs(pt_out - ort_out))))

    max_diff = float(np.max(abs_diffs))
    mean_diff = float(np.mean(abs_diffs))
    ort_match = max_diff < 1e-5
    status = "OK" if ort_match else "ATTENTION"
    print(
        f"[{model_name}] PyTorch vs ONNX Runtime (100 inputs) → {status} "
        f"| max |Δ| = {max_diff:.2e} (seuil 1e-5)"
    )

    # --- 4. Quantification PTQ INT8 (poids uniquement, sans calibration) ---
    print(f"[{model_name}] Quantification dynamique INT8 → {int8_path}")
    quantize_dynamic(
        model_input=str(out_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    print(f"[{model_name}] Fichier INT8 écrit : {int8_path} ({int8_path.stat().st_size} B)")

    n_params = sum(p.numel() for p in model.parameters())

    # --- 5. Extraction opérateurs ONNX ---
    onnx_model_reloaded = onnx.load(str(out_path))
    operators = sorted({n.op_type for n in onnx_model_reloaded.graph.node})
    _CMSIS_NN_OPS = {"Gemm", "Relu", "Sigmoid", "MatMul", "Add", "Mul", "Reshape", "Flatten"}
    cmsis_nn_compatible = set(operators).issubset(_CMSIS_NN_OPS)
    stm32cubeai_compatible = cmsis_nn_compatible

    return {
        "model_name":           model_name,
        "onnx_path":            str(out_path),
        "int8_path":            str(int8_path),
        "opset_version":        OPSET_VERSION,
        "input_dim":            input_dim,
        "n_params":             n_params,
        "onnx_fp32_bytes":      out_path.stat().st_size,
        "onnx_int8_bytes":      int8_path.stat().st_size,
        "onnx_valid":           onnx_valid,
        "ort_match_atol_1e5":   ort_match,
        "max_abs_diff":         max_diff,
        "mean_abs_diff":        mean_diff,
        "operators":            operators,
        "cmsis_nn_compatible":  cmsis_nn_compatible,
        "stm32cubeai_compatible": stm32cubeai_compatible,
    }


# ---------------------------------------------------------------------------
# STM32Cube.AI (optionnel — dépend de S1001)
# ---------------------------------------------------------------------------

def try_stm32ai_generate(onnx_path: str, output_dir: str) -> bool:
    """
    Tente de générer du code C INT8 via STM32Cube.AI CLI.

    Ne bloque pas si stm32ai n'est pas dans le PATH.
    TODO(dorra): confirmer la version STM32Cube.AI pour compatibilité STM32N6.
    """
    if shutil.which("stm32ai") is None:
        print("[STM32Cube.AI] stm32ai non trouvé dans le PATH — génération C ignorée.")
        print("  → Installez STM32Cube.AI CLI depuis https://www.st.com/stm32cubeai")
        return False

    stm32ai_out = Path(output_dir) / "stm32ai_output"
    stm32ai_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        "stm32ai", "generate",
        "-m", onnx_path,
        "--type", "onnx",
        "-o", str(stm32ai_out),
    ]
    print(f"[STM32Cube.AI] Commande : {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[STM32Cube.AI] Génération réussie → {stm32ai_out}")
        return True
    else:
        print(f"[STM32Cube.AI] Erreur (code {result.returncode}) :\n{result.stderr}")
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export PyTorch → ONNX + quantification PTQ INT8 (S1002 / S2403)"
    )
    parser.add_argument(
        "--model",
        choices=["ewc", "tinyol", "hdc", "mahalanobis", "all"],
        default=None,
        help=(
            "Modèle à exporter. Avec --dataset : export d'un combo spécifique. "
            "Sans --dataset : utilise --config (ancien comportement)."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        default=None,
        help="Dataset cible pour --model (ex. cwru, monitoring). "
             "Active le mode multi-dataset ; incompatible avec --config.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export systématique de tous les combos MODEL_DATASET_MAP (S2403b).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Chemin vers le fichier de config YAML (mode rétrocompat Sprint 4).",
    )
    parser.add_argument(
        "--ewc-config",
        default="configs/ewc_config.yaml",
        help="Config EWC (utilisé avec --model all sans --all). [défaut: configs/ewc_config.yaml]",
    )
    parser.add_argument(
        "--tinyol-config",
        default="configs/tinyol_config.yaml",
        help="Config TinyOL (utilisé avec --model all sans --all). [défaut: configs/tinyol_config.yaml]",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Chemin de sortie .onnx (pour --model ewc|tinyol avec --config).",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        default="experiments/onnx_sprint24/",
        help="Répertoire de sortie pour --all ou --dataset. [défaut: experiments/onnx_sprint24/]",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint PyTorch (.pt) à charger. Poids aléatoires si absent.",
    )
    parser.add_argument(
        "--stm32ai",
        action="store_true",
        help="Tenter la génération code C via STM32Cube.AI CLI (requiert S1001).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results: list[dict] = []
    out_dir = Path(args.output_dir)

    # ------------------------------------------------------------------
    # Mode --all : export systématique de tous les combos (S2403b)
    # ------------------------------------------------------------------
    if args.all:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 60}")
        print(f"  Export ONNX systématique — {len(MODEL_DATASET_MAP)} combos (S2403b)")
        print(f"  Sortie : {out_dir}")
        print(f"{'=' * 60}")

        for (model_id, dataset_id) in sorted(MODEL_DATASET_MAP.keys()):
            print(f"\n--- {model_id} × {dataset_id} ---")
            output_path = str(out_dir / f"{model_id}_{dataset_id}.onnx")

            if model_id == "ewc":
                r = export_ewc_for_dataset(dataset_id, output_path)
            elif model_id == "tinyol":
                r = export_tinyol_for_dataset(dataset_id, output_path)
            elif model_id == "mahalanobis":
                r = export_mahalanobis_onnx(dataset_id, output_path)
            else:  # hdc
                r = export_hdc_for_dataset(dataset_id, output_path)

            results.append(r)

        # Combos skippés
        for (model_id, dataset_id), reason in SKIP_COMBOS.items():
            results.append({
                "model_type": model_id,
                "dataset":    dataset_id,
                "status":     "skipped",
                "reason":     reason,
            })

        generate_manifest(results, str(out_dir))
        _print_all_summary(results)
        return

    # ------------------------------------------------------------------
    # Mode --model + --dataset : export d'un combo spécifique (S2403a)
    # ------------------------------------------------------------------
    if args.model is not None and args.dataset is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"{args.model}_{args.dataset}.onnx")

        key = (args.model, args.dataset)
        if key in SKIP_COMBOS:
            print(f"[{args.model}/{args.dataset}] Skip : {SKIP_COMBOS[key]}")
            return
        if key not in MODEL_DATASET_MAP and args.model != "hdc":
            print(f"Erreur : combo ({args.model}, {args.dataset}) non supporté.", file=sys.stderr)
            sys.exit(1)

        if args.model == "ewc":
            r = export_ewc_for_dataset(args.dataset, output_path)
        elif args.model == "tinyol":
            r = export_tinyol_for_dataset(args.dataset, output_path)
        elif args.model == "mahalanobis":
            r = export_mahalanobis_onnx(args.dataset, output_path)
        else:
            r = export_hdc_for_dataset(args.dataset, output_path)

        results.append(r)
        generate_manifest(results, str(out_dir))
        return

    # ------------------------------------------------------------------
    # Mode rétrocompatibilité Sprint 4 (--model ewc|tinyol|all + --config)
    # ------------------------------------------------------------------
    if args.model is None:
        print(
            "Erreur : --model requis (ou utiliser --all pour le mode multi-dataset).",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.model == "ewc":
        if args.config is None:
            print("Erreur : --config requis avec --model ewc", file=sys.stderr)
            sys.exit(1)
        out = args.output or "experiments/exp_160/ewc_backbone.onnx"
        summary = export_ewc_backbone(args.config, out, args.checkpoint)
        results.append(summary)
        if args.stm32ai:
            try_stm32ai_generate(out, str(Path(out).parent))

    elif args.model == "tinyol":
        if args.config is None:
            print("Erreur : --config requis avec --model tinyol", file=sys.stderr)
            sys.exit(1)
        out = args.output or "experiments/exp_160/tinyol_encoder.onnx"
        summary = export_tinyol_encoder(args.config, out, args.checkpoint)
        results.append(summary)
        if args.stm32ai:
            try_stm32ai_generate(out, str(Path(out).parent))

    else:  # all (legacy Sprint 4 — EWC + TinyOL seulement)
        legacy_out_dir = Path(args.output_dir)

        ewc_out = str(legacy_out_dir / "ewc_backbone.onnx")
        results.append(export_ewc_backbone(args.ewc_config, ewc_out, args.checkpoint))
        if args.stm32ai:
            try_stm32ai_generate(ewc_out, str(legacy_out_dir))

        tinyol_out = str(legacy_out_dir / "tinyol_encoder.onnx")
        results.append(export_tinyol_encoder(args.tinyol_config, tinyol_out, args.checkpoint))
        if args.stm32ai:
            try_stm32ai_generate(tinyol_out, str(legacy_out_dir))

    # --- Résumé JSON (export_summary.json — format interne) ---
    out_dir_for_summary = Path(results[0]["onnx_path"]).parent
    summary_path = out_dir_for_summary / "export_summary.json"
    summary_data = {
        "_exported_at": datetime.now().isoformat(),
        "_opset_version": OPSET_VERSION,
        "models": results,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nRésumé export → {summary_path}")

    # --- onnx_validation_report.json ---
    validation_report = {
        "_exported_at": datetime.now().isoformat(),
        "onnx_opset": OPSET_VERSION,
        "models": [
            {
                "model": r["model_name"],
                "onnx_path": r["onnx_path"],
                "int8_path": r["int8_path"],
                "onnx_opset": r["opset_version"],
                "operators": r.get("operators", []),
                "cmsis_nn_compatible": r.get("cmsis_nn_compatible", False),
                "stm32cubeai_compatible": r.get("stm32cubeai_compatible", False),
                "max_deviation_ort_vs_pytorch": r["max_abs_diff"],
                "ort_match_atol_1e5": r["ort_match_atol_1e5"],
                "n_params": r["n_params"],
            }
            for r in results
        ],
        "hdc": {
            "model": "hdc_classifier",
            "onnx_exportable": False,
            "reason": (
                "HDC utilise une accumulation additive de prototypes INT32 sans graphe "
                "computationnel différentiable. L'export ONNX n'est pas applicable. "
                "Opérations équivalentes : MatMul (encodage) + ArgMax (inférence)."
            ),
            "pseudo_onnx_ops": ["MatMul", "ArgMax"],
            "cmsis_nn_compatible": True,
        },
    }
    validation_report_path = out_dir_for_summary / "onnx_validation_report.json"
    with open(validation_report_path, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)
    print(f"Rapport validation → {validation_report_path}")

    # --- Affichage synthèse ---
    print("\n" + "=" * 60)
    print("SYNTHÈSE EXPORT ONNX (S1002)")
    print("=" * 60)
    all_ok = True
    for r in results:
        ok = r["onnx_valid"] and r["ort_match_atol_1e5"]
        all_ok &= ok
        status = "✓" if ok else "✗"
        fp32_kb = r["onnx_fp32_bytes"] / 1024
        int8_kb = r["onnx_int8_bytes"] / 1024
        print(
            f"  {status} {r['model_name']:20s} | params={r['n_params']:>5d} "
            f"| FP32={fp32_kb:.1f}Ko | INT8={int8_kb:.1f}Ko "
            f"| max|Δ|={r['max_abs_diff']:.1e}"
        )
    print("=" * 60)
    if not all_ok:
        print("ATTENTION : certains exports ont des problèmes de validation.")
        sys.exit(1)


def _print_all_summary(results: list[dict]) -> None:
    """Affiche un tableau synthèse pour le mode --all."""
    exported = [r for r in results if r.get("status") not in ("skipped", "fallback_joblib")]
    skipped  = [r for r in results if r.get("status") == "skipped"]
    fallback = [r for r in results if r.get("status") == "fallback_joblib"]

    print("\n" + "=" * 72)
    print("  SYNTHÈSE EXPORT ONNX SYSTÉMATIQUE (S2403b)")
    print("=" * 72)
    print(f"  Exportés  : {len(exported)}")
    print(f"  Skippés   : {len(skipped)}")
    print(f"  Fallback  : {len(fallback)}")
    print("-" * 72)
    for r in sorted(exported, key=lambda x: (x.get("model_type",""), x.get("dataset",""))):
        validated = r.get("validated", r.get("onnx_valid", False))
        ok = "✓" if validated else "✗"
        kb = r.get("file_size_kb") or round(r.get("onnx_fp32_bytes", 0) / 1024, 2)
        print(
            f"  {ok} {r.get('model_type','?'):12s} × {r.get('dataset','?'):12s} "
            f"| dim={r.get('input_dim',0):>3d} | {kb:.1f} Ko"
        )
    for r in sorted(skipped, key=lambda x: (x.get("model_type",""), x.get("dataset",""))):
        print(f"  ~ {r.get('model_type','?'):12s} × {r.get('dataset','?'):12s} → skip")
    print("=" * 72)


if __name__ == "__main__":
    main()
