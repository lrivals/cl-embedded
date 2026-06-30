"""
board_pair_recorder.py — Capture board d'une PAIRE Mahalanobis + supervisé (Sprint 30, S3010).

Généralisation board du DUAL_MODE (Sprint 27) au benchmark fixe « paire » :
le firmware PAIR_MODE (0x90/0xA0/0xB0, cf. pipeline.c) co-exécute le détecteur
Mahalanobis et un modèle supervisé en une seule trame UART (réponse 22 B).

Ce script mesure, sur la **carte réelle** (aucun chiffre inventé) :
  - latence **Mahalanobis seul**   (flags=0x00 → chemin Mahalanobis)
  - latence **supervisé seul**     (flags du modèle, ex. EWC 0x10)
  - latence **combinée (paire)**   (flags PAIR, ex. 0x90)
  - `.bss` du firmware via arm-none-eabi-size
  - métriques en ligne renvoyées par la paire (AUROC Maha + F1 supervisé)

Usage :
    python scripts/board_pair_recorder.py --pair maha-ewc --dataset cwru \\
        --port /dev/ttyACM0 --n-samples 300 --update \\
        --output experiments/exp_S30_board_maha_ewc

    # Sans board (vérifie l'orchestration, latences = N/A)
    python scripts/board_pair_recorder.py --pair maha-ewc --dataset cwru --dry-run \\
        --output experiments/exp_S30_board_maha_ewc
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ELF = _REPO_ROOT / "firmware" / "stm32f4_blink" / "build" / "stm32f4_blink.elf"
_INC = _REPO_ROOT / "firmware" / "stm32f4_blink" / "inc"

# Paire → (modèle supervisé sensor_stream, FLAGS supervisé seul, FLAGS PAIR, FLAGS TRIPLE).
# triple_flag absent ⇒ paire non portée board en TRIPLE_MODE (Sprint 31 : maha-ewc, maha-hdc).
_PAIRS: dict[str, dict] = {
    "maha-ewc":    {"sup": "ewc",    "sup_flag": 0x10, "pair_flag": 0x90, "triple_flag": 0xD0, "n_params": 30 + 1538 + 5},
    "maha-hdc":    {"sup": "hdc",    "sup_flag": 0x20, "pair_flag": 0xA0, "triple_flag": 0xE0, "n_params": 30 + 7000 + 5},
    "maha-tinyol": {"sup": "tinyol", "sup_flag": 0x80, "pair_flag": 0xB0, "n_params": 30 + 881},
}


def _read_maha_threshold() -> float | None:
    """Lit MAHA_THRESHOLD_INIT depuis inc/model_weights.h (= g_detector.threshold board)."""
    header = _INC / "model_weights.h"
    if not header.exists():
        return None
    m = re.search(r"MAHA_THRESHOLD_INIT\s*=\s*([-\d.eE]+)f?", header.read_text())
    return float(m.group(1)) if m else None


def _meta_forward_np(weights: dict, feats: np.ndarray) -> float:
    """meta_forward (numpy FP32) — référence de parité pour le verdict board."""
    feats = np.asarray(feats, dtype=np.float32)
    if weights["kind"] == "logreg":
        w = np.asarray(weights["w"], dtype=np.float32)
        z = float(np.dot(w, feats)) + float(weights["b"])
    else:  # mlp 1 couche cachée
        w1 = np.asarray(weights["w1"], dtype=np.float32)
        b1 = np.asarray(weights["b1"], dtype=np.float32)
        w2 = np.asarray(weights["w2"], dtype=np.float32).reshape(-1)
        hidden = np.maximum(w1 @ feats + b1, 0.0)
        z = float(np.dot(w2, hidden)) + float(weights["b2"])
    return float(1.0 / (1.0 + np.exp(-z)))


def _meta_parity(res_triple: list[dict], weights: dict, threshold: float) -> dict:
    """Reconstruit le vecteur de features méta du board et vérifie verdict board == numpy.

    Le board envoie (score_maha, p_sup, pred_maha, pred_sup, pred_meta, prob_meta). On reconstruit
    feats = [p_maha, p_sup, disagreement, conf_sup] *exactement* comme pipeline.c, puis on compare.
    """
    n_ok, n_tot, max_dprob = 0, 0, 0.0
    for r in res_triple:
        if "prob_meta" not in r:
            continue
        p_maha = 1.0 / (1.0 + np.exp(-(r["score_maha"] - threshold)))
        p_sup = r["p_sup"]
        feats = np.array(
            [p_maha, p_sup, 1.0 if r["pred_maha"] != r["pred_sup"] else 0.0, abs(p_sup - 0.5) * 2.0],
            dtype=np.float32,
        )
        prob_pc = _meta_forward_np(weights, feats)
        pred_pc = 1 if prob_pc > 0.5 else 0
        n_tot += 1
        if pred_pc == r["pred_meta"]:
            n_ok += 1
        max_dprob = max(max_dprob, abs(prob_pc - r["prob_meta"]))
    return {
        "n": n_tot,
        "parity_rate": (n_ok / n_tot) if n_tot else None,
        "max_prob_delta": round(max_dprob, 6) if n_tot else None,
    }


def _load_stream_module():
    spec = importlib.util.spec_from_file_location(
        "sensor_stream", Path(__file__).parent / "sensor_stream.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_bss_bytes() -> int | None:
    """Lit la taille .bss du firmware via arm-none-eabi-size (aucune valeur inventée)."""
    if not _ELF.exists():
        return None
    try:
        out = subprocess.check_output(
            ["arm-none-eabi-size", str(_ELF)], text=True, stderr=subprocess.STDOUT
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    # Format : text  data  bss  dec  hex  filename  (2e ligne = valeurs)
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    cols = re.split(r"\s+", lines[1].strip())
    try:
        return int(cols[2])  # colonne bss
    except (IndexError, ValueError):
        return None


def _latency_stats(results: list[dict]) -> dict:
    lats = [r["latency_us"] for r in results] if results else []
    if not lats:
        return {"n": 0, "p50_us": None, "p99_us": None, "mean_us": None}
    return {
        "n":       len(lats),
        "p50_us":  round(float(np.percentile(lats, 50)), 2),
        "p99_us":  round(float(np.percentile(lats, 99)), 2),
        "mean_us": round(float(np.mean(lats)), 2),
    }


def _run_stream(mod, port: str, baud: int, X, y, n_samples: int, n_tasks: int,
                model_flags: int, update: bool, dry_run: bool, verbose: bool) -> list[dict]:
    if dry_run:
        return mod._stream_dry_run(X, y, n_samples, n_tasks, update, verbose,
                                   protocol_version=3, model_flags=model_flags)
    return mod._stream_uart(port, baud, X, y, n_samples, n_tasks,
                            0.0, update, verbose,
                            protocol_version=3, model_flags=model_flags)


def main() -> None:
    p = argparse.ArgumentParser(description="Capture board d'une paire Maha+supervisé (S3010)")
    p.add_argument("--pair", choices=list(_PAIRS), required=True)
    p.add_argument("--dataset", choices=["cwru", "monitoring", "pronostia", "cmapss", "paderborn"],
                   default="cwru")
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--n-samples", type=int, default=300)
    p.add_argument("--n-tasks", type=int, default=3)
    p.add_argument("--update", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--triple", action="store_true",
                   help="Sprint 31 — ajoute le run TRIPLE_MODE (PAIR + méta) + parité board↔PC")
    p.add_argument("--meta", type=Path, default=None,
                   help="meta_weights.json (défaut : experiments/exp_S31_PC_{pair}_{dataset}/meta_weights.json)")
    p.add_argument("--platform", default="nucleo_f439zi")
    p.add_argument("--output", required=True)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    cfg = _PAIRS[args.pair]
    mod = _load_stream_module()

    if args.triple and "triple_flag" not in cfg:
        p.error(f"--triple non supporté pour la paire {args.pair} (board : maha-ewc, maha-hdc seulement)")

    print(f"Chargement dataset '{args.dataset}'...")
    X, y = mod._load_dataset(args.dataset)
    print(f"  {len(X)} samples, {X.shape[1]} features")

    t0 = time.time()
    # 1) Mahalanobis seul (flags=0x00 → chemin Mahalanobis du pipeline)
    print(f"\n[1/3] Mahalanobis seul ({args.n_samples} samples)…")
    res_maha = _run_stream(mod, args.port, args.baud, X, y, args.n_samples,
                           args.n_tasks, 0x00, args.update, args.dry_run, args.verbose)
    # 2) Supervisé seul
    print(f"[2/3] {cfg['sup']} seul…")
    res_sup = _run_stream(mod, args.port, args.baud, X, y, args.n_samples,
                          args.n_tasks, cfg["sup_flag"], args.update, args.dry_run, args.verbose)
    # 3) Paire combinée (PAIR_MODE)
    print(f"[3/3] Paire {args.pair} (PAIR_MODE 0x{cfg['pair_flag']:02X})…")
    res_pair = _run_stream(mod, args.port, args.baud, X, y, args.n_samples,
                           args.n_tasks, cfg["pair_flag"], args.update, args.dry_run, args.verbose)

    # 4) TRIPLE_MODE (PAIR + méta) — Sprint 31, optionnel
    res_triple: list[dict] = []
    parity: dict | None = None
    meta_weights: dict | None = None
    if args.triple:
        print(f"[4/4] Triple {args.pair} (TRIPLE_MODE 0x{cfg['triple_flag']:02X})…")
        res_triple = _run_stream(mod, args.port, args.baud, X, y, args.n_samples,
                                 args.n_tasks, cfg["triple_flag"], args.update,
                                 args.dry_run, args.verbose)
        # Parité board↔PC du verdict méta (pas en dry-run : pas de prob_meta réelle)
        meta_path = args.meta or (
            _REPO_ROOT / "experiments"
            / f"exp_S31_PC_{args.pair.replace('-', '_')}_{args.dataset}" / "meta_weights.json"
        )
        thr = _read_maha_threshold()
        if not args.dry_run and Path(meta_path).exists() and thr is not None:
            meta_weights = json.loads(Path(meta_path).read_text())
            parity = _meta_parity(res_triple, meta_weights, thr)
        else:
            print(f"  [parité] ignorée (dry-run, meta={meta_path} absent, ou seuil introuvable)")

    collection_time_s = time.time() - t0

    # Métriques en ligne de la paire (dernière trame ; dry-run ⇒ champs absents)
    last = res_pair[-1] if res_pair else {}
    auroc_maha = last.get("auroc_maha")
    f1_sup     = last.get("f1_sup")

    lat_maha = _latency_stats(res_maha)
    lat_sup  = _latency_stats(res_sup)
    lat_pair = _latency_stats(res_pair)
    lat_triple = _latency_stats(res_triple) if res_triple else None
    bss = _read_bss_bytes()

    # Métrique de référence Gap 2 : la latence triple si présente, sinon la paire.
    lat_gap2 = lat_triple if lat_triple else lat_pair

    out: dict = {
        "exp_id":   Path(args.output).name,
        "pair":     args.pair,
        "dataset":  args.dataset,
        "platform": args.platform,
        "mode":     "dry-run" if args.dry_run else "uart",
        "date":     datetime.now().strftime("%Y-%m-%d"),
        "n_params": cfg["n_params"],
        "frame_response_bytes": 27 if args.triple else 22,
        "latency_maha_us":     lat_maha,
        "latency_supervised_us": lat_sup,
        "latency_pair_us":     lat_pair,
        "online_metrics": {"auroc_maha": auroc_maha, "f1_supervised": f1_sup},
        "bss_bytes": bss,
        "collection_time_s": round(collection_time_s, 2),
        # Gap 2 — latence combinée << 100 ms
        "gap2_latency_compliant": (lat_gap2["mean_us"] is not None
                                   and lat_gap2["mean_us"] / 1000.0 < 100.0),
        "gap2_ram_compliant": (bss is not None and bss < 262144),
    }
    if args.triple:
        last_t = res_triple[-1] if res_triple else {}
        out["latency_triple_us"] = lat_triple
        out["meta"] = {
            "kind": meta_weights["kind"] if meta_weights else None,
            "f1_meta_online": last_t.get("f1_sup"),   # F1 en ligne du verdict méta (slot f1_sup)
            "parity": parity,                         # verdict board == numpy
        }

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(out, indent=2))

    snapshot = {
        "pair": args.pair, "dataset": args.dataset, "platform": args.platform,
        "port": args.port, "baud": args.baud, "n_samples": args.n_samples,
        "n_tasks": args.n_tasks, "update": args.update, "dry_run": args.dry_run,
        "triple": args.triple,
        "sup_flag": cfg["sup_flag"], "pair_flag": cfg["pair_flag"],
        "triple_flag": cfg.get("triple_flag"),
        "firmware_elf": str(_ELF),
    }
    import yaml
    (out_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False))

    print("\n--- Résultats paire board ---")
    print(f"  latence Maha seul   : {lat_maha}")
    print(f"  latence {cfg['sup']:<10}: {lat_sup}")
    print(f"  latence combinée    : {lat_pair}")
    if args.triple:
        print(f"  latence triple      : {lat_triple}")
        print(f"  parité méta board↔PC: {parity}")
    print(f"  AUROC Maha={auroc_maha}  F1 {cfg['sup']}={f1_sup}")
    print(f"  .bss={bss} B  ({None if bss is None else round(bss/2621.44, 1)}% de 256 Ko)")
    print(f"\nSauvegardé : {out_dir/'results.json'}")


if __name__ == "__main__":
    main()
