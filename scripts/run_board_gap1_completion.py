#!/usr/bin/env python3
"""run_board_gap1_completion.py — Complète le heatmap Gap 1 sur board réelle (S33).

Comble les cellules `pending` du heatmap accuracy cross-dataset (Gap 1) et produit
les latences par modèle de détection de faute, sur la NUCLEO-F439ZI réelle.

Deux campagnes, **un seul flash** (le firmware embarque tous les modes) :

1. **Accuracy (HW-only)** — `online_accuracy` board pour les 5 combos manquants :
     TinyOL×{cwru, paderborn}, HDC×{cwru, paderborn, monitoring}
   HDC (projection embarquée) et TinyOL (init en ligne) sont HW-only : pas de
   checkpoint, parité N/A par construction (cf. S3205). Streaming **avec** `--update`
   (online learning, comme le heatmap board des sprints précédents).
   → `experiments/exp_S33_board_gap1/results_{model}_{dataset}.json`

2. **Latence par modèle** — inférence vs inférence+update, pour les 4 modèles de
   détection (Mahalanobis, EWC, HDC, TinyOL) sur un dataset 5-feat représentatif
   (cwru). Deux streams par modèle : sans `--update` (inférence) et avec `--update`
   (inf+update). Remplace les valeurs hardcodées de `plot_latency()`.
   → `experiments/exp_S33_board_latency/latency_{model}_{inf|update}.json`

Aucun mode UART nouveau : on réutilise les nibbles existants
(`mahalanobis`=0x00, `ewc`=0x10, `hdc`=0x20, `tinyol`=0x80). `sensor_stream.py`
reste la source de vérité du protocole.

Usage :
    python scripts/run_board_gap1_completion.py --port /dev/ttyACM0
    python scripts/run_board_gap1_completion.py --skip-flash          # firmware déjà flashé
    python scripts/run_board_gap1_completion.py --dry-run             # valide la boucle sans board
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

FW_DIR = Path("firmware/stm32f4_blink")
EXPERIMENTS = Path("experiments")
GAP1_DIR = EXPERIMENTS / "exp_S33_board_gap1"
LATENCY_DIR = EXPERIMENTS / "exp_S33_board_latency"

GAP2_LATENCY_US = 100_000  # 100 ms (Gap 2)

# Cellules accuracy manquantes du heatmap board (model → datasets)
ACCURACY_COMBOS = [
    ("tinyol", "cwru"),
    ("tinyol", "paderborn"),
    ("hdc", "cwru"),
    ("hdc", "paderborn"),
    ("hdc", "monitoring"),   # monitoring zéro-paddé à 5 feat (sensor_sim._load_monitoring)
]

# Modèles de détection de faute pour le plot de latence (inf vs inf+update)
LATENCY_MODELS = ["mahalanobis", "ewc", "hdc", "tinyol"]
LATENCY_DATASET = "cwru"   # dataset 5-feat représentatif (latence ≈ dataset-agnostique)

N_SAMPLES = 150
RATE_HZ = 50.0


def _run(cmd: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _bss_bytes() -> int | None:
    """Lit .bss du binaire courant via arm-none-eabi-size (None si indisponible)."""
    try:
        out = subprocess.run(
            ["arm-none-eabi-size", str(FW_DIR / "build/stm32f4_blink.elf")],
            capture_output=True, text=True,
        )
        line = out.stdout.strip().splitlines()[-1].split()
        return int(line[2])  # text data bss ...
    except Exception:
        return None


def _stream(dataset: str, model: str, *, update: bool, port: str,
            out_json: Path, dry_run: bool) -> dict:
    """Lance sensor_stream.py et renvoie le dict stats (accuracy, latency_p50_us…)."""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "scripts/sensor_stream.py",
        "--dataset", dataset,
        "--model", model,
        "--n-samples", str(N_SAMPLES),
        "--rate-hz", str(RATE_HZ),
        "--protocol-version", "3",
        "--dump-samples",
        "--port", port,
        "--output", str(out_json),
    ]
    if update:
        cmd.append("--update")
    if dry_run:
        cmd.append("--dry-run")
    proc = _run(cmd)
    if proc.returncode != 0:
        print(f"    [!] sensor_stream a échoué (rc={proc.returncode})")
        print(proc.stderr[-800:])
        return {}
    if not out_json.exists():
        print(f"    [!] sortie absente : {out_json}")
        return {}
    return json.loads(out_json.read_text())


def run_accuracy(port: str, dry_run: bool) -> list[dict]:
    """Campagne 1 — online_accuracy board des 5 combos manquants."""
    print("\n=== Campagne 1 : accuracy board (HW-only) ===")
    bss = _bss_bytes()
    summary: list[dict] = []
    for model, dataset in ACCURACY_COMBOS:
        print(f"\n[{model} × {dataset}]")
        stream_json = GAP1_DIR / f"stream_{model}_{dataset}.json"
        stats = _stream(dataset, model, update=True, port=port,
                        out_json=stream_json, dry_run=dry_run)
        if not stats:
            continue
        lat_p50 = stats.get("latency_p50_us")
        result = {
            "exp_id": f"exp_S33_board_gap1_{model}_{dataset}",
            "model": model,
            "dataset": dataset,
            "platform": "nucleo_f439zi",
            "date": datetime.now().isoformat(timespec="seconds"),
            "parity_class": "hw_only",
            "parity_note": "N/A par construction (HDC projection embarquée / TinyOL init en ligne)",
            "online_accuracy": stats.get("accuracy"),
            "latency_us_p50": lat_p50,
            "latency_us_p99": stats.get("latency_p99_us"),
            "bss_bytes": bss,
            "ram_response_bytes": stats.get("ram_mean_bytes"),
            "crc_errors": stats.get("crc_errors"),
            "n_samples": stats.get("n_samples"),
            "gap2_latency_compliant": (lat_p50 is not None and lat_p50 < GAP2_LATENCY_US),
            "note_feature": ("monitoring zéro-paddé 4→5 feat (5ᵉ synthétique nulle)"
                             if dataset == "monitoring" else None),
        }
        out = GAP1_DIR / f"results_{model}_{dataset}.json"
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"    acc={result['online_accuracy']} lat_p50={lat_p50}µs → {out}")
        summary.append(result)
    return summary


def run_latency(port: str, dry_run: bool) -> list[dict]:
    """Campagne 2 — latence inférence vs inf+update par modèle de détection."""
    print("\n=== Campagne 2 : latence par modèle (inf vs inf+update) ===")
    summary: list[dict] = []
    for model in LATENCY_MODELS:
        row = {"model": model, "dataset": LATENCY_DATASET, "platform": "nucleo_f439zi"}
        for phase, update in [("inf", False), ("update", True)]:
            print(f"\n[{model} — {phase}]")
            stream_json = LATENCY_DIR / f"stream_{model}_{phase}.json"
            stats = _stream(LATENCY_DATASET, model, update=update, port=port,
                            out_json=stream_json, dry_run=dry_run)
            lat_p50 = stats.get("latency_p50_us") if stats else None
            key = "latency_inf_us" if phase == "inf" else "latency_inf_update_us"
            row[key] = lat_p50
            out = LATENCY_DIR / f"latency_{model}_{phase}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({
                "model": model, "phase": phase, "dataset": LATENCY_DATASET,
                "platform": "nucleo_f439zi",
                "latency_us_p50": lat_p50,
                "latency_us_p99": stats.get("latency_p99_us") if stats else None,
                "date": datetime.now().isoformat(timespec="seconds"),
            }, indent=2, ensure_ascii=False))
            print(f"    lat_p50={lat_p50}µs → {out}")
        summary.append(row)
    (LATENCY_DIR / "latency_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--skip-flash", action="store_true",
                   help="Ne pas (re)compiler/flasher — firmware déjà sur board")
    p.add_argument("--dry-run", action="store_true",
                   help="Valide la boucle sans board (valeurs simulées sensor_stream)")
    p.add_argument("--only", choices=["accuracy", "latency"], default=None,
                   help="Ne lancer qu'une des deux campagnes")
    args = p.parse_args()

    if not args.skip_flash and not args.dry_run:
        print("=== Build + flash firmware (une fois) ===")
        proc = _run(["make", "-C", str(FW_DIR), "all"])
        if proc.returncode != 0:
            print(proc.stdout[-1200:]); print(proc.stderr[-1200:])
            raise SystemExit("build firmware échoué")
        proc = _run(["make", "-C", str(FW_DIR), "flash"])
        if proc.returncode != 0:
            print(proc.stderr[-1200:])
            raise SystemExit("flash firmware échoué")

    acc_summary, lat_summary = [], []
    if args.only != "latency":
        acc_summary = run_accuracy(args.port, args.dry_run)
    if args.only != "accuracy":
        lat_summary = run_latency(args.port, args.dry_run)

    print("\n=== Récapitulatif ===")
    for r in acc_summary:
        print(f"  acc  {r['model']:11s} × {r['dataset']:10s} : {r['online_accuracy']}")
    for r in lat_summary:
        print(f"  lat  {r['model']:11s} : inf={r.get('latency_inf_us')}µs "
              f"inf+update={r.get('latency_inf_update_us')}µs")
    print("\nProchaine étape : python scripts/generate_comparison_sprint23.py "
          "puis régénérer les figures (notebook + generate_presentation_plots.py).")


if __name__ == "__main__":
    main()
