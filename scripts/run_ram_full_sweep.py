#!/usr/bin/env python3
"""run_ram_full_sweep.py — Orchestrateur unique de la RAM totale par cellule (S4902).

Produit, pour chaque cellule (modèle × dataset × condition × encodage × plateforme), un
JSON conforme au **schéma S4901** :

    RAM totale = .data + .bss + pic de pile   (board, référence matérielle)
    RAM proxy  = tracemalloc peak             (PC, allocateur Python/PyTorch)

Board et PC sont **non fusionnables** (CR 16 juillet 2026 §1) : colonnes séparées via le
champ ``platform``, jamais additionnées.

**Réutilisation stricte** (aucune primitive réécrite) :
  • pic de pile board  → ``measure_stack_watermark.py`` (OpenOCD Tcl RPC : ``OpenOCD``,
    ``read_symbols``, ``scan_stack_peak`` — stack painting position-dépendant) ;
  • stream board       → ``scripts/sensor_stream.py`` (subprocess, protocole V3 inchangé —
    le pic de pile n'est PAS dans le snapshot UART, il est lu par OpenOCD après coup) ;
  • split .bss/modèle  → ``scripts/ram_breakdown.py`` (vérifié vs ``nm``) ;
  • RAM PC             → ``feature_conditions.train_and_evaluate`` + ``tracemalloc`` direct
    (peak d'un forward représentatif).

Le pic de pile est **quasi identique entre modèles/phases** (~4,3–4,7 Ko) : le compilateur
réserve une seule trame pour ``pipeline_run()`` (max de ses branches, dominée par
``hv[HDC_DIM]`` = 4 Ko). Cette nuance est consignée dans chaque JSON board.

Usage :
    # PC (aucune carte) — dry-run puis réel
    python scripts/run_ram_full_sweep.py --platform pc --datasets monitoring --models ewc --dry-run
    python scripts/run_ram_full_sweep.py --platform pc

    # Board (carte requise ; lancer d'abord le serveur OpenOCD Tcl RPC 6666)
    openocd -f interface/stlink.cfg -f target/stm32f4x.cfg &
    python scripts/run_ram_full_sweep.py --platform board
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

FW_DIR = ROOT / "firmware" / "stm32f4_blink"
ELF = FW_DIR / "build" / "stm32f4_blink.elf"
OUT_DEFAULT = ROOT / "experiments" / "exp_S49_ram"

ALL_MODELS = ["ewc", "hdc", "tinyol", "mahalanobis"]
ALL_DATASETS = ["monitoring", "pronostia"]
ALL_ENCODINGS = ["fp32", "int8"]
PHASES = ["idle", "inference", "update"]

# Flags sensor_stream --model par encodage. maha int8 = build dédié -DMAHA_INT8 (hors
# périmètre mesure RAM S49) → N/A honnête.
FP32_FLAG = {"ewc": "ewc", "hdc": "hdc", "tinyol": "tinyol", "mahalanobis": "mahalanobis"}
INT8_FLAG = {"ewc": "ewc-int8", "hdc": "hdc-int8", "tinyol": "tinyol-int8"}

GAP2_LATENCY_US = 100_000  # 100 ms (Gap 2), référence pour annotation

# Protocole de référence consigné dans conditions{} (mêmes groupes + MAJ/éval, CR §1).
UPDATE_PROTOCOL = "SGD embarqué 1 échantillon (sensor_stream --update, protocole V3)"
EVAL_PROTOCOL = "stream chronologique par tâche CL, seed 42, condition figée"


# ── Schéma S4901 ─────────────────────────────────────────────────────────────

def _cell_path(out_dir: Path, model: str, dataset: str, condition: str,
               encoding: str, platform: str) -> Path:
    return out_dir / f"{model}_{dataset}_{condition}_{encoding}_{platform}.json"


def _blank_cell(model: str, dataset: str, condition: str, encoding: str,
                platform: str) -> dict:
    """Cellule au schéma S4901, `null`/`pending` (aucun chiffre inventé)."""
    return {
        "model": model, "dataset": dataset, "condition": condition,
        "encoding": encoding, "platform": platform,
        "data_bytes": None, "bss_bytes": None,
        "stack_peak_inference_bytes": None, "stack_peak_update_bytes": None,
        "total_ram_bytes": None,
        "stack_history": [],
        "conditions": {
            "same_groups": True,
            "update_protocol": UPDATE_PROTOCOL,
            "eval_protocol": EVAL_PROTOCOL,
        },
        "status": "pending",
    }


# ── Sous-process helper ──────────────────────────────────────────────────────

def _run(cmd: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# ── Dimensions par cellule (source unique feature_conditions) ────────────────

def _dims(dataset: str, condition: str) -> int:
    from src.evaluation.feature_conditions import resolve_feature_indices
    # 5feat/all → k identique pour tous les modèles (indices partagés).
    return len(resolve_feature_indices(condition, "ewc", dataset)[0])


# ══════════════════════════════════════════════════════════════════════════════
# PC — proxy tracemalloc (aucune carte)
# ══════════════════════════════════════════════════════════════════════════════

def _truncate_tasks(tasks: list[dict], n: int = 300) -> list[dict]:
    """Tronque X/y de chaque tâche à `n` lignes (RAM dominée par le modèle, pas les données)."""
    out = []
    for t in tasks:
        tt = dict(t)
        for key in ("X_train", "y_train", "X_val", "y_val"):
            if key in tt and tt[key] is not None:
                tt[key] = tt[key][:n]
        out.append(tt)
    return out


def run_pc_cell(model: str, dataset: str, condition: str, encoding: str,
                seed: int, dry_run: bool) -> dict:
    cell = _blank_cell(model, dataset, condition, encoding, "pc")

    if encoding == "int8":
        cell["status"] = "na"
        cell["na_reason"] = ("PC = proxy tracemalloc FP32 ; le gain INT8 (poids ÷4) est "
                             "analytique — la RAM INT8 mesurée est côté board.")
        return cell

    if dry_run:
        return cell  # pending/null, schéma valide

    from src.evaluation.feature_conditions import (
        load_condition_arrays,
        load_native_task_arrays,
        resolve_feature_indices,
        train_and_evaluate,
    )

    idx = resolve_feature_indices(condition, model, dataset)[0]
    tasks = _truncate_tasks(load_native_task_arrays(dataset, seed=seed))
    res = train_and_evaluate(model, tasks, idx, seed=seed)
    predict = res["predict_fn"]

    # Pic RAM d'un forward représentatif (tracemalloc = même primitive que memory_profiler).
    X, _y, _i, _n = load_condition_arrays(dataset, condition, model, seed=seed)
    X_repr = np.ascontiguousarray(X[:64])
    tracemalloc.start()
    predict(X_repr)
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    cell.update({
        "data_bytes": None, "bss_bytes": None,   # n/a en Python
        "stack_peak_inference_bytes": int(peak),
        "stack_peak_update_bytes": None,          # forward-only proxy PC
        "total_ram_bytes": int(peak),
        "stack_history": [{"phase": "inference", "stack_peak_bytes": int(peak)}],
        "n_params": int(res["n_params"]),
        "model_build_peak_bytes": int(res["ram_peak_bytes"]),
        "status": "done",
    })
    cell["conditions"]["pc_method"] = "tracemalloc peak d'un forward (feature_conditions)"
    return cell


# ══════════════════════════════════════════════════════════════════════════════
# BOARD — mesure matérielle (.data + .bss + pic de pile via OpenOCD)
# ══════════════════════════════════════════════════════════════════════════════

def _elf_sections() -> dict[str, int]:
    from measure_stack_watermark import read_symbols
    syms = read_symbols(ELF, ["_sdata", "_edata", "_sbss", "_ebss", "_estack"])
    return {
        "data_bytes": syms["_edata"] - syms["_sdata"],
        "bss_bytes": syms["_ebss"] - syms["_sbss"],
        "_ebss": syms["_ebss"], "_estack": syms["_estack"],
    }


def _openocd_reachable(host: str, tcl_port: int) -> bool:
    import socket
    try:
        s = socket.create_connection((host, tcl_port), timeout=2.0)
        s.close()
        return True
    except OSError:
        return False


def _board_reset(host: str, tcl_port: int) -> None:
    from measure_stack_watermark import OpenOCD
    ocd = OpenOCD(host, tcl_port)
    try:
        ocd.cmd("reset run")  # re-peinture de la pile au boot (startup.s)
    finally:
        ocd.close()
    time.sleep(0.5)


def _board_scan(host: str, tcl_port: int, ebss: int, estack: int) -> int:
    from measure_stack_watermark import OpenOCD, scan_stack_peak
    ocd = OpenOCD(host, tcl_port)
    try:
        ocd.cmd("halt")
        peak = scan_stack_peak(ocd, ebss, estack)
    finally:
        ocd.cmd("resume")
        ocd.close()
    return int(peak)


def _stream(dataset: str, flag: str, condition: str, args, update: bool) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / "sensor_stream.py"),
           "--dataset", dataset, "--model", flag, "--condition", condition,
           "--port", args.port, "--n-samples", str(args.n_samples),
           "--rate-hz", str(args.rate_hz), "--protocol-version", "3"]
    if update:
        cmd.append("--update")
    _run(cmd, timeout=600)


def _build_flash(k: int, host: str, tcl_port: int, extra_cflags: str = "") -> bool:
    """make clean && make <dims> all, puis flash via le serveur OpenOCD déjà lancé.

    Le flash passe par le Tcl RPC (``program … verify reset``) plutôt que ``make flash`` :
    ``make flash`` démarre sa PROPRE instance openocd et entre en conflit avec le serveur
    persistant qui possède déjà la sonde ST-LINK (accès unique).
    """
    make_dims = [f"EWC_IN={k}", f"MAHA_DIM={k}", f"TINYOL_IN={k}", f"HDC_N_FEATURES={k}"]
    if k > 16:
        make_dims.append(f"PROTO_MAX_N={k}")
    cmd_all = ["make", "-C", str(FW_DIR), *make_dims, "all"]
    if extra_cflags:
        cmd_all.append(f"EXTRA_CFLAGS={extra_cflags}")
    subprocess.run(["make", "-C", str(FW_DIR), "clean"], capture_output=True)
    if _run(cmd_all).returncode != 0:
        print("  [FAIL build]")
        return False
    from measure_stack_watermark import OpenOCD
    try:
        ocd = OpenOCD(host, tcl_port)
        ocd.cmd(f"program {ELF} verify reset")
        ocd.close()
    except OSError as exc:
        print(f"  [FAIL flash via RPC] {exc}")
        return False
    time.sleep(0.5)
    return True


def _bss_model_bytes(model: str) -> int | None:
    """Contribution .bss par modèle (ram_breakdown, informatif) ; None si indispo."""
    try:
        import ram_breakdown as rb
        bd = rb.monitoring_breakdown(ELF)
        key = {"ewc": "EWC", "hdc": "HDC", "tinyol": "TinyOL", "mahalanobis": "Mahalanobis"}[model]
        return int(bd[key]["bss_real_nm"])
    except Exception:  # noqa: BLE001 — informatif, tolérant à la dérive de dims
        return None


def run_board_dataset(dataset: str, condition: str, models: list[str],
                      encodings: list[str], args) -> list[dict]:
    """Une build/flash par (dataset, condition) → cellules fp32 + int8(ewc/hdc/tinyol).

    Le pic de pile et .bss/.data sont des propriétés du build (partagés) ; seules les
    phases inference/update varient (marginalement) par flag modèle streamé.
    """
    k = _dims(dataset, condition)
    cells: list[dict] = []

    # Cellules non applicables (maha × int8 = build dédié -DMAHA_INT8, hors périmètre RAM).
    if "int8" in encodings and "mahalanobis" in models:
        na = _blank_cell("mahalanobis", dataset, condition, "int8", "board")
        na["status"] = "na"
        na["na_reason"] = ("Maha INT8 = build compilation dédié -DMAHA_INT8 + export ; hors "
                           "périmètre mesure RAM S49 (gain = poids ÷4, cf. Sprint 29).")
        cells.append(na)

    if args.dry_run:
        for enc in encodings:
            flags = FP32_FLAG if enc == "fp32" else INT8_FLAG
            for m in models:
                if m not in flags:
                    continue
                cells.append(_blank_cell(m, dataset, condition, enc, "board"))
        return cells

    if not _build_flash(k, args.host, args.tcl_port):
        for enc in encodings:
            flags = FP32_FLAG if enc == "fp32" else INT8_FLAG
            for m in models:
                if m not in flags:
                    continue
                c = _blank_cell(m, dataset, condition, enc, "board")
                c["status"] = "na"
                c["na_reason"] = f"build/flash échoué (k={k})"
                cells.append(c)
        return cells

    sec = _elf_sections()

    # Phase idle : propriété du build (aucun stream) — mesurée une fois.
    _board_reset(args.host, args.tcl_port)
    idle_peak = _board_scan(args.host, args.tcl_port, sec["_ebss"], sec["_estack"])
    print(f"  [idle] pic pile = {idle_peak} B  (.bss={sec['bss_bytes']} .data={sec['data_bytes']})")

    for enc in encodings:
        flags = FP32_FLAG if enc == "fp32" else INT8_FLAG
        for m in models:
            if m not in flags:
                continue  # maha×int8 déjà traité en N/A
            flag = flags[m]

            # Phase inference (forward gelé).
            _board_reset(args.host, args.tcl_port)
            _stream(dataset, flag, condition, args, update=False)
            inf_peak = _board_scan(args.host, args.tcl_port, sec["_ebss"], sec["_estack"])

            # Phase update (inférence + MAJ CL embarquée).
            _board_reset(args.host, args.tcl_port)
            _stream(dataset, flag, condition, args, update=True)
            upd_peak = _board_scan(args.host, args.tcl_port, sec["_ebss"], sec["_estack"])

            total = sec["data_bytes"] + sec["bss_bytes"] + max(inf_peak, upd_peak)
            cell = _blank_cell(m, dataset, condition, enc, "board")
            cell.update({
                "data_bytes": sec["data_bytes"], "bss_bytes": sec["bss_bytes"],
                "bss_model_bytes": _bss_model_bytes(m),
                "stack_peak_inference_bytes": inf_peak,
                "stack_peak_update_bytes": upd_peak,
                "total_ram_bytes": total,
                "stack_history": [
                    {"phase": "idle", "stack_peak_bytes": idle_peak},
                    {"phase": "inference", "stack_peak_bytes": inf_peak},
                    {"phase": "update", "stack_peak_bytes": upd_peak},
                ],
                "n_features": k, "stream_flag": flag,
                "gap2_total_pct_256ko": round(100.0 * total / (256 * 1024), 2),
                "shared_frame_note": ("pic de pile ~identique entre modèles : trame unique "
                                      "pipeline_run() (max des branches, hv[HDC_DIM]=4Ko)"),
                "weights_note": "poids par défaut (RAM/pile invariantes aux valeurs de poids)",
                "status": "done",
            })
            cells.append(cell)
            print(f"  [{m:11s} {enc}] flag={flag} pile inf={inf_peak} upd={upd_peak} "
                  f"→ total={total} B ({cell['gap2_total_pct_256ko']}%)")
    return cells


# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(ALL_MODELS),
                    help="liste csv (défaut : ewc,hdc,tinyol,mahalanobis)")
    ap.add_argument("--datasets", default=",".join(ALL_DATASETS),
                    help="liste csv (défaut : monitoring,pronostia)")
    ap.add_argument("--encodings", default=",".join(ALL_ENCODINGS),
                    help="liste csv (défaut : fp32,int8)")
    ap.add_argument("--condition", default="5feat", choices=["5feat", "all", "best"])
    ap.add_argument("--platform", required=True, choices=["board", "pc"])
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--rate-hz", type=float, default=0.0)  # 0 = vitesse max
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--tcl-port", type=int, default=6666)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="écrit les JSON pending/null (schéma valide) sans carte")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    encodings = [e.strip() for e in args.encodings.split(",") if e.strip()]
    args.out.mkdir(parents=True, exist_ok=True)

    def _write(cell: dict) -> None:
        path = _cell_path(args.out, cell["model"], cell["dataset"], cell["condition"],
                          cell["encoding"], cell["platform"])
        if args.skip_existing and path.exists():
            prev = json.loads(path.read_text())
            if prev.get("status") == "done":
                print(f"  = skip (done) {path.name}")
                return
        path.write_text(json.dumps(cell, indent=2, ensure_ascii=False))
        print(f"  → {path.name}  [{cell['status']}]")

    if args.platform == "pc":
        for dataset in datasets:
            for enc in encodings:
                for m in models:
                    try:
                        cell = run_pc_cell(m, dataset, args.condition, enc, args.seed, args.dry_run)
                    except Exception as exc:  # noqa: BLE001 — cellule robuste
                        cell = _blank_cell(m, dataset, args.condition, enc, "pc")
                        cell["status"] = "na"
                        cell["na_reason"] = f"{type(exc).__name__}: {exc}"
                    _write(cell)
        return 0

    # platform == board
    if not ELF.exists() and not args.dry_run:
        print(f"⚠ ELF absent ({ELF}) — build/flash tentés par cellule.")
    if not args.dry_run and not _openocd_reachable(args.host, args.tcl_port):
        print(f"⚠ OpenOCD injoignable ({args.host}:{args.tcl_port}). Lance :\n"
              "  openocd -f interface/stlink.cfg -f target/stm32f4x.cfg\n"
              "Cellules board laissées 'pending'.")
        for dataset in datasets:
            for enc in encodings:
                flags = FP32_FLAG if enc == "fp32" else INT8_FLAG
                for m in models:
                    cell = _blank_cell(m, dataset, args.condition, enc, "board")
                    if enc == "int8" and m not in flags:
                        cell["status"] = "na"
                        cell["na_reason"] = "Maha INT8 = build dédié -DMAHA_INT8 (hors périmètre)."
                    _write(cell)
        return 1

    for dataset in datasets:
        for cell in run_board_dataset(dataset, args.condition, models, encodings, args):
            _write(cell)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
