#!/usr/bin/env python3
"""stack_usage_report.py — Borne supérieure STATIQUE du pic de pile firmware.

Complément « ceinture + bretelles » de ``scripts/measure_stack_watermark.py`` : ce
dernier mesure le pic de pile RÉEL atteint par la carte (high-water mark, stack
painting) ; ici on calcule une borne supérieure **statique**, indépendante du run,
à partir des cadres de pile par fonction émis par GCC (``-fstack-usage`` → un
fichier ``build/<unité>.su`` par unité de compilation).

Objectif : vérifier ``borne_statique ≥ pic_mesuré``. Si l'inégalité tient, le pic
mesuré ne peut pas avoir « raté » un chemin d'appel plus profond → renforce
l'argument Gap 2 (RAM < 100 Ko avec chiffres sûrs).

Méthode — chaîne pire-cas DÉCLARÉE (``WORST_CASE_CHAIN``), auditable :
le firmware NUCLEO-F439ZI n'utilise pas de récursion et son dispatch modèle est
fait par ``switch``/``if`` dans ``pipeline_run`` (appels DIRECTS, pas de pointeurs
de fonction) → le graphe d'appels de la boucle chaude est court et connu. On
somme, par niveau de la chaîne, le plus gros cadre effectivement présent dans les
``.su``. C'est une CONTRE-VÉRIFICATION, pas une preuve formelle : un walker de
call-graph automatique n'est pas fiable en présence de pointeurs de fonction (non
utilisés ici) — d'où le choix d'une chaîne déclarée. Tout gros cadre HORS chaîne
est signalé (garde-fou anti-dérive : si un futur commit câble une telle fonction
dans la boucle, il faut ré-auditer la chaîne).

CLI :
    python scripts/stack_usage_report.py
    python scripts/stack_usage_report.py --measured experiments/exp_S39_ram/ram_ewc.json
    python scripts/stack_usage_report.py --json experiments/exp_S39_ram/stack_bound.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
FW = ROOT / "firmware" / "stm32f4_blink"
DEFAULT_BUILD = FW / "build"

# ── Chaîne pire-cas déclarée (auditable) ──────────────────────────────────────
# Chaque niveau = ensemble de fonctions qui peuvent être empilées à ce niveau ;
# on prend le plus gros cadre présent. La boucle chaude est :
#   main → pipeline_run (détient hv[HDC_DIM] = 4 Ko) → un noyau modèle (feuille).
# Les noyaux SGD ne rappellent PAS leur forward (vérifié) → une seule feuille.
WORST_CASE_CHAIN: list[tuple[str, list[str]]] = [
    ("entrée", ["main"]),
    ("boucle pipeline", ["pipeline_run"]),
    (
        "noyau modèle (feuille la plus profonde appelée par pipeline_run)",
        [
            # HDC
            "hdc_encode", "hdc_predict", "hdc_update", "hdc_update_with_sample",
            "hdc_binarize", "hdc_int8_encode", "hdc_int8_predict", "hdc_int8_update",
            # EWC (binaire / multiclasse / régression / INT8)
            "ewc_forward", "ewc_sgd_step", "ewc_consolidate",
            "ewc_mc_forward", "ewc_mc_sgd_step",
            "ewc_reg_forward", "ewc_reg_sgd_step",
            "ewc_int8_forward", "ewc_int8_update", "ewc_int8_v2_forward",
            # Mahalanobis (fp32 / q15 / int8)
            "maha_score", "maha_update", "maha_predict",
            "maha_q15_score", "maha_int8_score",
            # TinyOL
            "tinyol_encode", "tinyol_decode", "tinyol_int8_encode",
            # méta + gate de dérive + ring buffer
            "meta_predict", "drift_update", "ring_buffer_push",
        ],
    ),
]

_SU_LINE = re.compile(r"^(?P<loc>[^\t]+)\t(?P<size>\d+)\t(?P<qual>\w+)")


# ── Parsing des .su ───────────────────────────────────────────────────────────
def parse_su_files(build_dir: Path) -> list[dict[str, Any]]:
    """Parse tous les build/*.su → liste de {func, file, frame, qualifier}.

    Format d'une ligne .su : ``fichier:ligne:col:fonction<TAB>taille<TAB>qualifieur``
    (le nom de fonction est le dernier champ ':' de la localisation).
    """
    frames: list[dict[str, Any]] = []
    su_files = sorted(build_dir.glob("*.su"))
    if not su_files:
        raise SystemExit(
            f"Aucun .su dans {build_dir} — recompiler avec -fstack-usage :\n"
            "  cd firmware/stm32f4_blink && make clean && make all"
        )
    for su in su_files:
        for line in su.read_text().splitlines():
            m = _SU_LINE.match(line)
            if not m:
                continue
            loc = m.group("loc")
            # loc = "src/hdc.c:12:6:hdc_encode" → func = dernier segment
            func = loc.rsplit(":", 1)[-1]
            file_part = loc.rsplit(":", 1)[0] if ":" in loc else loc
            frames.append({
                "func": func,
                "file": file_part,
                "frame": int(m.group("size")),
                "qualifier": m.group("qual"),  # static / dynamic / bounded
            })
    return frames


def _frame_of(frames: list[dict[str, Any]], names: list[str]) -> dict[str, Any] | None:
    """Retourne l'entrée de plus gros cadre parmi ``names`` (ou None)."""
    cands = [f for f in frames if f["func"] in names]
    return max(cands, key=lambda f: f["frame"]) if cands else None


def compute_bound(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Somme la chaîne pire-cas déclarée → borne supérieure statique de pile."""
    by_name: dict[str, int] = {}
    for f in frames:
        by_name[f["func"]] = max(by_name.get(f["func"], 0), f["frame"])

    chain: list[dict[str, Any]] = []
    total = 0
    for level, names in WORST_CASE_CHAIN:
        picked = _frame_of(frames, names)
        if picked is None:
            chain.append({"level": level, "func": None, "frame": 0})
            continue
        total += picked["frame"]
        chain.append({"level": level, "func": picked["func"],
                      "frame": picked["frame"]})

    chain_funcs = {c["func"] for c in chain if c["func"]}
    # Gros cadres HORS chaîne : garde-fou (dead code GC'd absent des .su n'y est pas ;
    # mais un .su peut lister une fonction non liée → on la signale si frame > seuil).
    off_chain = sorted(
        (f for f in frames
         if f["func"] not in chain_funcs and f["frame"] >= total),
        key=lambda f: f["frame"], reverse=True,
    )
    dynamic = sorted(
        (f for f in frames if f["qualifier"] != "static"),
        key=lambda f: f["frame"], reverse=True,
    )
    return {
        "bound_bytes": total,
        "chain": chain,
        "off_chain_large": off_chain,
        "dynamic_or_bounded": dynamic,
    }


# ── Restitution ───────────────────────────────────────────────────────────────
def _fmt(n: int) -> str:
    return f"{n:,} B".replace(",", " ")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD,
                    help="répertoire contenant les .su (défaut firmware build/)")
    ap.add_argument("--top", type=int, default=12,
                    help="nombre de plus gros cadres à afficher")
    ap.add_argument("--measured", type=Path, default=None,
                    help="JSON de measure_stack_watermark.py à contre-vérifier")
    ap.add_argument("--json", type=Path, default=None,
                    help="écrire le rapport (JSON) dans ce fichier")
    args = ap.parse_args()

    frames = parse_su_files(args.build_dir)
    result = compute_bound(frames)

    print(f"Cadres de pile (.su) — {len(frames)} fonctions, "
          f"{args.build_dir}\n")
    top = sorted(frames, key=lambda f: f["frame"], reverse=True)[: args.top]
    print(f"{'fonction':<28}{'cadre':>10}  {'qualifieur':<10}fichier")
    print("-" * 78)
    for f in top:
        print(f"{f['func']:<28}{_fmt(f['frame']):>10}  "
              f"{f['qualifier']:<10}{f['file']}")

    print("\n── Chaîne pire-cas déclarée → borne supérieure statique ──────────")
    for c in result["chain"]:
        name = c["func"] or "(aucun cadre trouvé)"
        print(f"  {c['level']:<52}{name:<26}{_fmt(c['frame']):>10}")
    print(f"  {'':<52}{'BORNE STATIQUE':<26}{_fmt(result['bound_bytes']):>10}")

    if result["dynamic_or_bounded"]:
        print("\n⚠ Cadres non 'static' (dynamic/bounded) — la borne peut être "
              "sous-estimée pour ces fonctions :")
        for f in result["dynamic_or_bounded"]:
            print(f"    {f['func']} : {_fmt(f['frame'])} [{f['qualifier']}]")
    if result["off_chain_large"]:
        print("\n⚠ Gros cadres HORS chaîne déclarée (à ré-auditer s'ils deviennent "
              "atteignables depuis pipeline_run) :")
        for f in result["off_chain_large"]:
            print(f"    {f['func']} : {_fmt(f['frame'])}  ({f['file']})")

    verdict = None
    measured_peak = None
    if args.measured is not None:
        rec = json.loads(args.measured.read_text())
        measured_peak = rec.get("stack_peak_bytes")
        print("\n── Contre-vérification vs pic mesuré ─────────────────────────")
        print(f"  pic de pile mesuré   : {_fmt(measured_peak)}  "
              f"({args.measured})")
        print(f"  borne statique       : {_fmt(result['bound_bytes'])}")
        if measured_peak is None:
            print("  ⚠ pas de stack_peak_bytes dans le JSON (mesure non faite)")
            verdict = None
        else:
            ok = result["bound_bytes"] >= measured_peak
            verdict = ok
            marge = result["bound_bytes"] - measured_peak
            print(f"  verdict              : "
                  f"{'✅ borne ≥ pic' if ok else '❌ BORNE < PIC (incohérent !)'}"
                  f"  (marge {marge:+d} B)")

    if args.json is not None:
        out = {
            "build_dir": str(args.build_dir),
            "bound_bytes": result["bound_bytes"],
            "chain": result["chain"],
            "top_frames": top,
            "dynamic_or_bounded": result["dynamic_or_bounded"],
            "off_chain_large": result["off_chain_large"],
            "measured_stack_peak_bytes": measured_peak,
            "bound_ge_measured": verdict,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\n→ rapport écrit : {args.json}")

    # Exit non nul si la contre-vérification échoue (borne < pic mesuré).
    return 1 if verdict is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
