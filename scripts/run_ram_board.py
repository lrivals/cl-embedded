#!/usr/bin/env python3
"""run_ram_board.py — Pic de RAM réel par modèle sur NUCLEO-F439ZI (Monitoring).

Driver optionnel (carte requise). Pour chaque modèle du dataset Monitoring :
  1. reset de la carte (re-peinture de la pile au boot, cf. startup.s) ;
  2. stream d'une charge Monitoring en mode entraînement (chemin le plus gourmand
     en pile) via scripts/sensor_stream.py ;
  3. halt + scan du high-water mark via scripts/measure_stack_watermark.py ;
  4. écriture d'un JSON par modèle dans experiments/exp_S39_ram/.

Le notebook ram_explained.ipynb ingère ces JSON s'ils existent ; sinon il marque
le pic carte « à mesurer » (règle « aucun chiffre inventé »).

Pré-requis : NUCLEO branchée (/dev/ttyACM0), openocd, firmware flashé (`make flash`),
et un serveur OpenOCD lancé à part :
    openocd -f interface/stlink.cfg -f target/stm32f4x.cfg

Exemple :
    python scripts/run_ram_board.py --port /dev/ttyACM0 --n-samples 400
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ELF = ROOT / "firmware" / "stm32f4_blink" / "build" / "stm32f4_blink.elf"
OUT_DIR = ROOT / "experiments" / "exp_S39_ram"

# Modèles Monitoring et leur flag sensor_stream (--model).
MODELS = {
    "EWC": "ewc",
    "HDC": "hdc",
    "Mahalanobis": "mahalanobis",
    "TinyOL": "tinyol",
}

# Étiquette lisible par flag sensor_stream (pour les modèles hors set Monitoring, ex. INT8).
FLAG_LABELS = {
    "ewc": "EWC", "hdc": "HDC", "mahalanobis": "Mahalanobis", "tinyol": "TinyOL",
    "ewc-int8": "EWC-INT8", "hdc-int8": "HDC-INT8", "tinyol-int8": "TinyOL-INT8",
    "maha-q15": "Mahalanobis-Q15",
}

sys.path.insert(0, str(ROOT / "scripts"))


def board_available(port: str) -> bool:
    return Path(port).exists() and shutil.which("openocd") is not None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--tcl-port", type=int, default=6666)
    ap.add_argument("--n-samples", type=int, default=400)
    ap.add_argument("--rate-hz", type=int, default=50)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    # Extensions (défauts = comportement historique Monitoring 5-feat, 4 modèles).
    ap.add_argument("--dataset", default="monitoring",
                    help="dataset streamé (sensor_stream --dataset), défaut monitoring")
    ap.add_argument("--condition", default=None, choices=[None, "5feat", "all", "best"],
                    help="condition de features S35 (sensor_stream --condition)")
    ap.add_argument("--models", nargs="+", default=None,
                    help="liste de flags sensor_stream (ex. ewc-int8 hdc). "
                         "Défaut : ewc hdc mahalanobis tinyol")
    ap.add_argument("--suffix", default="",
                    help="suffixe des JSON de sortie (ex. _int8, _all_k21) pour ne pas "
                         "écraser les 4 références")
    args = ap.parse_args()

    # Résolution de la liste de modèles à streamer.
    if args.models:
        models = {FLAG_LABELS.get(f, f): f for f in args.models}
    else:
        models = dict(MODELS)

    if not board_available(args.port):
        print(f"⚠ Carte indisponible ({args.port} / openocd) — mesure carte "
              "sautée. Le notebook retombera sur l'analytique (« à mesurer »).")
        return 0
    if not ELF.exists():
        raise SystemExit(f"ELF introuvable : {ELF} (lancer `make all` puis `make flash`)")

    # Import tardif : ne dépend d'OpenOCD que si la carte est là.
    from measure_stack_watermark import OpenOCD, read_symbols, scan_stack_peak

    syms = read_symbols(ELF, ["_sdata", "_edata", "_sbss", "_ebss", "_estack"])
    data_bytes = syms["_edata"] - syms["_sdata"]
    bss_bytes = syms["_ebss"] - syms["_sbss"]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Borne supérieure statique de pile du build courant (.su), contre-vérification.
    try:
        import stack_usage_report as sur
        frames = sur.parse_su_files(sur.DEFAULT_BUILD)
        static_bound = sur.compute_bound(frames)["bound_bytes"]
    except Exception as exc:  # noqa: BLE001 — borne optionnelle
        print(f"⚠ borne statique indisponible ({exc}) — champ omis.")
        static_bound = None

    for name, flag in models.items():
        print(f"\n=== {name} ({flag}) ===")
        ocd = OpenOCD(args.host, args.tcl_port)
        try:
            ocd.cmd("reset run")   # re-peinture de la pile au boot
        finally:
            ocd.close()
        time.sleep(0.5)

        # Charge en mode entraînement (pile maximale). --condition ajouté si demandé.
        cmd = [sys.executable, str(ROOT / "scripts" / "sensor_stream.py"),
               "--dataset", args.dataset, "--model", flag, "--update",
               "--port", args.port, "--n-samples", str(args.n_samples),
               "--rate-hz", str(args.rate_hz), "--protocol-version", "3"]
        if args.condition:
            cmd += ["--condition", args.condition]
        subprocess.run(cmd, check=True)

        ocd = OpenOCD(args.host, args.tcl_port)
        try:
            ocd.cmd("halt")
            stack_peak = scan_stack_peak(ocd, syms["_ebss"], syms["_estack"])
        finally:
            ocd.cmd("resume")
            ocd.close()

        ram_peak = data_bytes + bss_bytes + stack_peak
        import json
        rec = {
            "label": name, "model_flag": flag, "dataset": args.dataset,
            "condition": args.condition,
            "data_bytes": data_bytes, "bss_bytes": bss_bytes,
            "stack_peak_bytes": stack_peak, "ram_peak_bytes": ram_peak,
            "static_bound_bytes": static_bound,
            "bound_ge_measured": (static_bound >= stack_peak)
                                 if static_bound is not None else None,
        }
        out = args.out_dir / f"ram_{flag}{args.suffix}.json"
        out.write_text(json.dumps(rec, indent=2))
        bound_str = (f" | borne stat = {static_bound} B "
                     f"({'✅≥' if rec['bound_ge_measured'] else '❌<'} pic)"
                     if static_bound is not None else "")
        print(f"  pic pile = {stack_peak} B  |  RAM pic = {ram_peak} B{bound_str}  → {out}")

    print(f"\n✅ Mesures carte écrites dans {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
