#!/usr/bin/env python3
"""run_s39_board.py — Validation board NUCLEO-F439ZI du kernel INT8 v2 (S3915/S3916/S3919).

Flashe la tête EWC INT8 **v2** (acc int32 + scales par-canal + activations calibrées) sur la
carte réelle et mesure ce que le PC ne peut pas : latence DWT, ``.bss`` cible, F1 board, CRC,
et **parité gelée board↔PC**. Le v2 est sélectionné **à la compilation** (``-DEWC_INT8_V2``)
car le nibble protocole est saturé : le chemin 0x40 (``FRAME_FLAGS_INT8_MODE``) route vers le
v2 au lieu du v1 (mirroir ``-DMAHA_INT8``, S2912). Le wire format UART est **inchangé**
(``sensor_stream.py`` intact).

Schémas (variante de build) :
    - legacy_c          : kernel **v1** (aucun flag) — baseline A/B (S3916)
    - per_channel_int8  : v2 défaut         (``-DEWC_INT8_V2``)
    - q15               : v2 16-bit         (``-DEWC_INT8_V2 -DEWC_INT8_Q15``)
    - mixed             : v2 poids int8/act int16 (``-DEWC_INT8_V2 -DEWC_INT8_MIXED``)

Parité (S3918/S3919) : le côté PC est l'**émulateur bit-exact** exécutant le **même schéma**
(``forward_quant``), jamais le QAT S28. Le checkpoint FP32 est **réutilisé tel quel** depuis
``exp_S39_matched/checkpoints/`` (produit par ``run_s39_matched_compare.py``) → mêmes poids des
deux côtés → parité gelée bit-exacte attendue. À défaut de checkpoint apparié, on l'entraîne et
on le dumpe (mêmes hyperparamètres board que l'ablation).

Honnêteté : **aucun JSON écrit tant que la carte n'a pas streamé** (règle « pas de résultat
inventé »).

Usage :
    python scripts/run_s39_board.py --scheme per_channel_int8 --dataset pronostia
    python scripts/run_s39_board.py --scheme q15 --dataset cmapss --condition 5feat
    python scripts/run_s39_board.py --scheme per_channel_int8 --dataset pronostia --ab
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.sensor_stream as ss  # noqa: E402
from scripts.run_feature_condition_board import _bss_bytes, train_maha_board  # noqa: E402
from scripts.run_s39_int8_ablation import train_ewc_head  # noqa: E402
from scripts.run_s39_matched_compare import SCHEME_TO_QUANTCONFIG  # noqa: E402
from src.evaluation.feature_conditions import load_condition_arrays  # noqa: E402
from src.utils.int8_c_emulation import (  # noqa: E402
    EWCHeadWeights,
    calibrate_activations,
    forward_quant,
    predict,
)

FW_DIR = Path("firmware/stm32f4_blink")
EXPERIMENTS = Path("experiments")
MATCHED_DIR = EXPERIMENTS / "exp_S39_matched"
OUT_DIR = EXPERIMENTS / "exp_S39_board"
GAP2_LATENCY_US = 100_000  # 100 ms (Gap 2)

# Schéma → flags de build (EXTRA_CFLAGS). legacy_c = kernel v1 (aucun flag v2).
SCHEME_BUILD_FLAGS: dict[str, str] = {
    "legacy_c": "",
    "per_channel_int8": "-DEWC_INT8_V2",
    "q15": "-DEWC_INT8_V2 -DEWC_INT8_Q15",
    "mixed": "-DEWC_INT8_V2 -DEWC_INT8_MIXED",
}
ALL_SCHEMES = list(SCHEME_BUILD_FLAGS)


def _run(cmd: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _resolve_checkpoint(dataset: str, condition: str, X: np.ndarray, y: np.ndarray,
                        seed: int) -> Path:
    """Réutilise le checkpoint apparié (parité exacte) ou l'entraîne + dumpe si absent."""
    matched = MATCHED_DIR / "checkpoints" / f"ewc_{dataset}_{condition}.pt"
    if matched.exists():
        print(f"  [ckpt] réutilise l'apparié {matched} (poids identiques PC↔board)")
        return matched
    import torch

    print(f"  [ckpt] apparié absent → entraîne la tête board ({dataset}/{condition})")
    model = train_ewc_head(X, y, seed=seed)
    matched.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, matched)
    return matched


def build_and_flash(scheme: str, dataset: str, condition: str, k: int, X: np.ndarray,
                    y: np.ndarray, ckpt: Path, exp_dir: Path) -> int:
    """Export (FP32 + v2 calibré) → build (flags du schéma) → flash. Retourne .bss (B)."""
    maha_ckpt = train_maha_board(X, exp_dir)  # cohérence dims du build (non streamé)
    export = [sys.executable, "scripts/export_weights_c.py",
              "--mahal", str(maha_ckpt), "--ewc-head", str(ckpt),
              "--dataset", dataset, "--condition", condition, "--model", "ewc"]
    if scheme != "legacy_c":
        # v2 : exporte poids int8 par-canal + scales + EWC_V2_ACT_MAX (calibrés sur la
        # condition → même act_max que l'émulateur → parité par construction).
        export += ["--int8-v2", str(ckpt)]
    if _run(export).returncode != 0:
        raise RuntimeError("export_weights_c échec")

    make_dims = [f"EWC_IN={k}", f"MAHA_DIM={k}", f"TINYOL_IN={k}", f"HDC_N_FEATURES={k}"]
    if k > 16:
        make_dims.append(f"PROTO_MAX_N={k}")
    subprocess.run(["make", "-C", str(FW_DIR), "clean"], capture_output=True)
    make_cmd = ["make", "-C", str(FW_DIR), *make_dims]
    extra = SCHEME_BUILD_FLAGS[scheme]
    if extra:
        make_cmd.append(f"EXTRA_CFLAGS={extra}")
    make_cmd.append("all")
    if _run(make_cmd).returncode != 0:
        raise RuntimeError("make échec")
    bss = _bss_bytes()
    if _run(["make", "-C", str(FW_DIR), "flash"]).returncode != 0:
        raise RuntimeError("flash échec")
    return bss


def _load_head_weights(ckpt: Path) -> EWCHeadWeights:
    """Charge les poids FP32 (``EWCHeadWeights``) du checkpoint apparié."""
    import torch

    sd = torch.load(ckpt, map_location="cpu").get("model_state_dict")
    return EWCHeadWeights.from_state_dict(sd)


def run_scheme(scheme: str, dataset: str, condition: str, args, seed: int) -> dict:
    """Build/flash/stream d'un schéma + parité gelée board↔émulateur PC."""
    print(f"\n{'='*70}\n=== BOARD v2  scheme={scheme}  dataset={dataset}  "
          f"condition={condition}  ===\n{'='*70}")
    X, y, idx, names = load_condition_arrays(dataset, condition, "ewc", seed=seed)
    k = len(idx)
    ckpt = _resolve_checkpoint(dataset, condition, X, y, seed)

    exp_dir = OUT_DIR / f"build_{scheme}_{dataset}_{condition}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    bss = build_and_flash(scheme, dataset, condition, k, X, y, ckpt, exp_dir)

    n_samples = min(args.n_samples, len(X)) if args.n_samples > 0 else len(X)
    # Stream gelé (sans --update) : le chemin 0x40 exécute le v2 (ou v1 si legacy_c).
    results = ss._stream_uart(
        args.port, args.baud, X, y,
        n_samples=n_samples, n_tasks=args.n_tasks,
        rate_hz=args.rate_hz, request_update=False, verbose=args.verbose,
        protocol_version=3, model_flags=ss.FRAME_FLAGS_INT8_MODE,
    )
    if not results:
        raise RuntimeError("aucune réponse board (stream vide)")
    stats = ss._compute_stats(results)

    # Parité gelée : ré-émule le schéma sur les features réellement streamées.
    # act_max calibré sur X complet == calibration board (EWC_V2_ACT_MAX) == émulateur.
    feats = np.array([r["features"] for r in results], dtype=np.float64)
    board_pred = np.array([int(r["pred"]) for r in results])
    w = _load_head_weights(ckpt)
    act_max = calibrate_activations(w, X)
    logits_pc = forward_quant(w, feats, SCHEME_TO_QUANTCONFIG[scheme], act_max=act_max)
    pc_pred = predict(logits_pc)
    n_cmp = min(len(board_pred), len(pc_pred))
    parity_rate = float((board_pred[:n_cmp] == pc_pred[:n_cmp]).mean()) if n_cmp else None
    n_mismatch = int((board_pred[:n_cmp] != pc_pred[:n_cmp]).sum())
    lat = stats.get("latency_p50_us")

    # F1 émulateur apparié (référence attendue, cf. S3918).
    matched_json = MATCHED_DIR / f"matched_ewc_{dataset}_{scheme}.json"
    f1_emulator = None
    if matched_json.exists():
        f1_emulator = json.loads(matched_json.read_text()).get("f1_int8_pc")

    result = {
        "exp_id": f"results_{scheme}_{dataset}",
        "platform": "nucleo_f439zi", "model": "ewc", "kernel": (
            "int8_v1" if scheme == "legacy_c" else "int8_v2"),
        "scheme": scheme, "dataset": dataset, "condition": condition, "n_features": k,
        "feature_names": names, "date": datetime.now().isoformat(timespec="seconds"),
        "stream_mode": "frozen (sans --update)",
        "n_streamed": len(results), "n_compared": n_cmp,
        "f1_faulty": stats.get("f1_faulty"), "f1_macro": stats.get("f1_macro"),
        "metric_value": stats.get("f1_faulty"),
        "f1_emulator_matched": f1_emulator,
        "online_accuracy": stats.get("accuracy"),
        "latency_us_p50": lat, "latency_us_p99": stats.get("latency_p99_us"),
        "latency_us_mean": stats.get("latency_mean_us"),
        "bss_bytes": bss, "crc_errors": stats.get("crc_errors"),
        "gap2_latency_compliant": (lat is not None and lat < GAP2_LATENCY_US),
        # Parité gelée board↔émulateur (schéma exact). Attendue ≈ 1.000 (inférence déterministe).
        "parity_class": "frozen",
        "parity_rate": parity_rate, "parity_mismatch_count": n_mismatch,
    }
    out = OUT_DIR / f"results_{scheme}_{dataset}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"  {scheme}: .bss={bss}B lat_p50={lat}µs F1={result['f1_faulty']} "
          f"(émul={f1_emulator}) parité={parity_rate} (mismatch={n_mismatch}) "
          f"crc={stats.get('crc_errors')} → {out}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Validation board kernel INT8 v2 (S3915)")
    p.add_argument("--scheme", choices=ALL_SCHEMES, default="per_channel_int8")
    p.add_argument("--dataset", required=True)
    p.add_argument("--condition", default="5feat")
    p.add_argument("--ab", action="store_true",
                   help="Aussi flasher le v1 (legacy_c) et comparer F1 v1 vs v2 (S3916).")
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--n-samples", type=int, default=300, dest="n_samples")
    p.add_argument("--n-tasks", type=int, default=3, dest="n_tasks")
    p.add_argument("--rate-hz", type=float, default=50.0, dest="rate_hz")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    schemes = [args.scheme]
    if args.ab and "legacy_c" not in schemes:
        schemes = ["legacy_c", args.scheme]  # A/B : v1 baseline puis v2 (S3916)

    rows = []
    for s in schemes:
        try:
            rows.append(run_scheme(s, args.dataset, args.condition, args, args.seed))
        except Exception as exc:  # noqa: BLE001 — cellule robuste
            print(f"  [FAIL {s}/{args.dataset}] {type(exc).__name__}: {exc}")

    if len(rows) >= 2:  # bilan A/B (S3916)
        v1 = next((r for r in rows if r["scheme"] == "legacy_c"), None)
        v2 = next((r for r in rows if r["scheme"] != "legacy_c"), None)
        if v1 and v2:
            print(f"\n[S3916] A/B {args.dataset} : v1(legacy) F1={v1['f1_faulty']} "
                  f"→ v2({v2['scheme']}) F1={v2['f1_faulty']} "
                  f"(récupération F1 sur board réelle)")


if __name__ == "__main__":
    main()
