"""
run_s53_build_isolation.py — S5306 / B3–B8 : isolation à variable unique de l'effet « build ».

LA QUESTION

Sur le build stock (dimensions par défaut), le courant suit la cadence (S5304 :
+20,73 ± 0,21 µA/Hz, r² 0,9996) et l'énergie par inférence se chiffre. Sur le build aux
dimensions du Sprint 38 (k=4), cette dépendance DISPARAÎT — et avec elle toute grandeur
qui s'en déduit, dont `energy_uj_per_update` (S5306). B3 a écarté les poids comme cause
(dimension tenue à k=4, poids exportés contre repli Xavier : les deux variantes plates).
B8 met la DIMENSION à l'épreuve, dans une séance unique et à poids non chargés des deux
côtés, en ne faisant varier qu'elle.

POURQUOI LA PAIRE EST PROPRE : `sensor_stream.py --dataset monitoring` envoie 4 features
quelle que soit la dimension compilée (`sensor_sim._load_monitoring`). Les deux binaires
reçoivent donc la MÊME trame de 27 octets, à la même cadence, avec la même tête non
chargée. Longueur de trame et contenu des poids sont écartés par CONSTRUCTION.

CE QUE CE PILOTE N'INVENTE PAS : la mesure est le balayage de cadence de S5304, appelé tel
quel (`run_s53_rate_sweep.measure` / `write_outputs`) ; la règle de décision vit dans
`src/evaluation/build_isolation.py`. Le pilote n'apporte que le pilotage du binaire (make,
flash, empreinte) et le manifeste anti-mislabel — la leçon du drapeau TinyOL du Sprint 52 :
une cellule mesurée sur un binaire qui n'est pas le sien est écrite sous un nom qui ment.

CE QU'IL NE FAIT PAS : le flash se fait `JP5` en place puis le cavalier bascule à la main
sur la sonde — deux phases séparées (`--prepare` puis `--measure`), comme S5306.

Usage :
    python scripts/run_s53_build_isolation.py --prepare --variant paire2_k4
    # basculer JP5 sur la sonde
    python scripts/run_s53_build_isolation.py --measure --variant paire2_k4 \
        --board-port /dev/ttyACM0
    python scripts/run_s53_build_isolation.py --summary
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rs = _load("run_s53_rate_sweep", ROOT / "scripts" / "run_s53_rate_sweep.py")
from src.evaluation import build_isolation as bi   # noqa: E402

FW_DIR = ROOT / "firmware" / "stm32f4_blink"
DEFAULT_OUT = ROOT / "experiments" / "exp_S53_build_isolation"
MANIFEST_NAME = "firmware_state.json"


# ── Phase 1 : compiler, empreindre, flasher ─────────────────────────────────

def _elf_metrics() -> dict:
    """`.text`, `.bss` et empreinte SHA-256 du binaire réellement produit.

    L'empreinte est ce qui permet, après coup, d'affirmer que deux cellules ont bien été
    mesurées sur deux binaires DIFFÉRENTS — et lesquels. `.text` dit ce qui a changé
    (les tables de poids pèsent quelques kilo-octets), `.bss` que la RAM, elle, n'a pas
    bougé : sans cela, un écart de pente pourrait s'expliquer par une autre variable.
    """
    elf = FW_DIR / "build" / "stm32f4_blink.elf"
    sizes = subprocess.run(["arm-none-eabi-size", str(elf)],
                           capture_output=True, text=True, check=True).stdout
    text, data, bss = (int(v) for v in sizes.splitlines()[1].split()[:3])
    # Empreinte des en-têtes de poids : c'est ce qui atteste que les deux arms d'une paire
    # ont bien reçu les MÊMES tables (et, pour B8, que le repli Xavier joue des deux côtés
    # parce que leur `NATIVE_DIM` ne correspond à aucune des deux dimensions compilées).
    # Sans elle, « poids identiques des deux côtés » resterait une intention : les en-têtes
    # sont régénérés par d'autres pilotes de la campagne (`--prepare` de S5306), et une
    # régénération entre deux arms suffirait à confondre poids et dimension — l'erreur
    # exacte qui a rendu la première isolation non concluante (S5306, 2026-09-07 matin).
    entetes = {
        nom: hashlib.sha256((FW_DIR / "inc" / nom).read_bytes()).hexdigest()[:16]
        for nom in ("model_weights.h", "model_weights_ewc.h")
        if (FW_DIR / "inc" / nom).is_file()
    }
    return {
        "text_bytes": text,
        "data_bytes": data,
        "bss_bytes": bss,
        "elf_sha256": hashlib.sha256(elf.read_bytes()).hexdigest(),
        "weight_headers_sha256": entetes,
    }


def prepare(args) -> dict:
    """Compile la variante, la flashe, et écrit le manifeste de ce qui tourne."""
    variant = bi.VARIANTS[args.variant]
    dims = bi.make_dims(int(variant["ewc_in"]))
    extra = variant["extra_cflags"]

    cmd_all = ["make", "-C", str(FW_DIR), *dims, "all"]
    if extra:
        cmd_all.insert(3, f'EXTRA_CFLAGS={extra}')
    subprocess.run(["make", "-C", str(FW_DIR), "clean"], check=True,
                   capture_output=True)
    subprocess.run(cmd_all, check=True, capture_output=True)
    metrics = _elf_metrics()
    subprocess.run(["make", "-C", str(FW_DIR), *dims, "flash"], check=True,
                   capture_output=True)

    manifest = {
        "variant": args.variant,
        "description": variant["description"],
        "make_dims": " ".join(dims),
        "extra_cflags": extra,
        "ewc_in": int(variant["ewc_in"]),
        "weights_state": variant["weights_state"],
        "dataset": bi.DATASET,
        "cell": bi.CELL,
        "paire": variant["paire"],
        **metrics,
        "flashed": datetime.now(timezone.utc).isoformat(),
    }
    out_dir = args.out / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[isolation] variante « {args.variant} » flashée : {manifest['make_dims']}")
    print(f"[isolation] .text={metrics['text_bytes']} B  .bss={metrics['bss_bytes']} B  "
          f"sha256={metrics['elf_sha256'][:12]}…")
    print(f"[isolation] manifeste → {out_dir / MANIFEST_NAME}")
    print("\nBasculer le cavalier JP5 sur l'alimentation sonde, puis lancer --measure.")
    return manifest


# ── Garde anti-mislabel ─────────────────────────────────────────────────────

def check_manifest(args) -> dict:
    """La variante déclarée au banc est-elle celle qui a été flashée ?"""
    path = args.out / args.variant / MANIFEST_NAME
    if not path.is_file():
        raise SystemExit(
            f"manifeste de flash absent : {path}\n"
            f"Lancer d'abord « --prepare --variant {args.variant} ». Aucune mesure n'est "
            f"écrite sans savoir quel binaire tourne."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["variant"] != args.variant:
        raise SystemExit(
            f"variante déclarée « {args.variant} » ≠ variante flashée "
            f"« {manifest['variant']} » ({manifest['flashed']}). Rien n'est mesuré."
        )
    return manifest


# ── Phase 2 : mesurer la cellule (balayage S5304 appelé tel quel) ────────────

def measure(args) -> dict:
    manifest = check_manifest(args)
    args.dataset = bi.DATASET
    args.only = bi.CELL
    cells = rs.parse_cells(args.only, argparse.ArgumentParser())
    mesurees = rs.measure(args, cells)
    for cell in mesurees.values():
        cell["build_variant"] = args.variant
        cell["firmware_state"] = manifest
        cell["session_id"] = args.session_id
    out_dir = args.out / args.variant
    rs.write_outputs(mesurees, out_dir)
    print(f"[isolation] cellule {bi.CELL} de « {args.variant} » → {out_dir}")
    return mesurees


# ── Phase 3 : verdicts appariés ─────────────────────────────────────────────

def _read_cell(out_dir: Path, variant: str) -> dict | None:
    """Cellule d'une variante, enrichie de son manifeste de flash s'il vit à côté.

    Les deux variantes de B3 ont été mesurées avant ce pilote : leur `firmware_state.json`
    est un fichier voisin, pas un bloc de la cellule. Il est relu ici — en LECTURE SEULE,
    aucune mesure existante n'est réécrite — pour que le tableau des variantes porte les
    mêmes colonnes (`.text`, `.bss`, empreinte) quelle que soit la séance d'origine.
    """
    path = out_dir / variant / f"{bi.CELL}.json"
    if not path.is_file():
        return None
    cell = json.loads(path.read_text(encoding="utf-8"))
    if "firmware_state" not in cell:
        voisin = out_dir / variant / MANIFEST_NAME
        if voisin.is_file():
            cell["firmware_state"] = json.loads(voisin.read_text(encoding="utf-8"))
    return cell


def build_summary(out_dir: Path) -> dict:
    """Verdicts de chaque paire — lecture seule, tout est calculé, rien n'est saisi."""
    cellules = {name: _read_cell(out_dir, name) for name in bi.VARIANTS}
    paires = {}
    empreintes = {name: (cellules[name] or {}).get("firmware_state", {})
                  .get("weight_headers_sha256")
                  for name in bi.VARIANTS if cellules.get(name)}
    for nom, spec in bi.PAIRS.items():
        a, b = spec["cells"]
        cell_a, cell_b = cellules.get(a), cellules.get(b)
        sessions = [c.get("session_id") for c in (cell_a, cell_b) if c is not None]
        meme_seance = (None if len(sessions) < 2 or any(s is None for s in sessions)
                       else sessions[0] == sessions[1])
        paires[nom] = bi.paired_verdict(cell_a, cell_b, a, b,
                                        varied=spec["varied"], same_session=meme_seance)
        paires[nom]["held_constant"] = spec["held"]
        paires[nom]["not_held"] = spec.get("not_held")
        # Les tables de poids étaient-elles littéralement les mêmes des deux côtés ?
        # `None` quand l'information manque (cellules d'avant ce pilote) : jamais un
        # « oui » par défaut.
        h_a, h_b = empreintes.get(a), empreintes.get(b)
        paires[nom]["identical_weight_headers"] = (
            None if not h_a or not h_b else h_a == h_b)

    summary = {
        "question": (
            "La disparition de la dépendance à la cadence sur le build aux dimensions du "
            "Sprint 38 suit-elle la DIMENSION compilée, une fois les poids écartés (B3) ?"
        ),
        "method": bi.METHOD,
        "method_note": (
            "la différence de pentes est ici un DIAGNOSTIC de dimension, jamais une "
            "énergie : la règle A4 réserve la conversion en µJ aux paires dont chaque "
            "régression est linéaire, et une cellule plate est le phénomène cherché."
        ),
        "reference_stock_build": {
            "source": "exp_S53_rate_sweep/ewc_fp32.json (S5304, build stock)",
        },
        "variants": {
            name: {
                **{k: v for k, v in bi.VARIANTS[name].items()},
                "measured": cellules.get(name) is not None,
                "text_bytes": (cellules[name].get("firmware_state", {}).get("text_bytes")
                               if cellules.get(name) else None),
                "bss_bytes": (cellules[name].get("firmware_state", {}).get("bss_bytes")
                              if cellules.get(name) else None),
                "elf_sha256": (cellules[name].get("firmware_state", {}).get("elf_sha256")
                               if cellules.get(name) else None),
                "slope_ua_per_hz": (cellules[name].get("slope_ua_per_hz")
                                    if cellules.get(name) else None),
                "slope_std_ua_per_hz": (cellules[name].get("slope_std_ua_per_hz")
                                        if cellules.get(name) else None),
                "r2": cellules[name].get("r2") if cellules.get(name) else None,
                "energy_uj_per_inference": (cellules[name].get("energy_uj_per_inference")
                                            if cellules.get(name) else None),
                "latency_p50_us": (cellules[name].get("dwt_latency_us_p50")
                                   if cellules.get(name) else None),
                "crc_errors": (cellules[name].get("protocol_check", {}).get("crc_errors")
                               if cellules.get(name) else None),
            } for name in bi.VARIANTS
        },
        "pairs": paires,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    # La rédaction de B3 (conception, conclusion, non-conclusions) n'est pas recalculable :
    # elle est reportée telle quelle plutôt que perdue à la première réécriture.
    ancien_path = out_dir / "summary.json"
    if ancien_path.is_file():
        ancien = json.loads(ancien_path.read_text(encoding="utf-8"))
        narratif = {k: ancien[k] for k in ("experiment", "design", "conclusion",
                                           "not_concluded", "delta_na_reason")
                    if k in ancien}
        if narratif:
            summary["b3_narrative"] = narratif

    stock = ROOT / "experiments" / "exp_S53_rate_sweep" / f"{bi.CELL}.json"
    if stock.is_file():
        cell = json.loads(stock.read_text(encoding="utf-8"))
        summary["reference_stock_build"].update({
            "slope_ua_per_hz": cell.get("slope_ua_per_hz"),
            "slope_std_ua_per_hz": cell.get("slope_std_ua_per_hz"),
            "r2": cell.get("r2"),
            "presence": bi.slope_presence(cell),
        })
    return summary


def refit(out_dir: Path) -> dict:
    """Recalcule les grandeurs dérivées de chaque variante, sans carte ni sonde."""
    recalculees = {}
    for name in bi.VARIANTS:
        variant_dir = out_dir / name
        if (variant_dir / f"{bi.CELL}.json").is_file():
            recalculees[name] = rs.refit(variant_dir)
    return recalculees


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--prepare", action="store_true",
                        help="compile + flashe la variante (JP5 en place)")
    parser.add_argument("--measure", action="store_true",
                        help="mesure la cellule de la variante flashée (JP5 sur la sonde)")
    parser.add_argument("--summary", action="store_true",
                        help="recalcule les verdicts appariés depuis les cellules écrites")
    parser.add_argument("--refit", action="store_true",
                        help="recalcule les grandeurs dérivées sans carte ni sonde")
    parser.add_argument("--variant", choices=sorted(bi.VARIANTS),
                        help="variante de binaire (cf. src/evaluation/build_isolation.py)")
    parser.add_argument("--session-id", default=None,
                        help="identifiant de séance : deux cellules qui le partagent sont "
                             "appariées, sinon le verdict porte sa réserve inter-séance")
    parser.add_argument("--board-port", default=None)
    parser.add_argument("--port", default=None, help="port de la SONDE (auto)")
    parser.add_argument("--rates", type=float, nargs="+", default=list(rs.DEFAULT_RATES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--window", type=float, default=10.0)
    parser.add_argument("--settle", type=float, default=2.0)
    parser.add_argument("--n-check", type=int, default=300)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--achieved-at-each-rate", action="store_true")
    parser.add_argument("--no-shuffle-order", dest="shuffle_order", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--hw-profile", type=Path,
                        default=ROOT / "configs" / "hw_profile_f439zi.yaml")
    args = parser.parse_args(argv)

    modes = [args.prepare, args.measure, args.summary, args.refit]
    if sum(bool(m) for m in modes) != 1:
        parser.error("choisir exactement un mode : --prepare, --measure, --summary ou --refit")
    if (args.prepare or args.measure) and not args.variant:
        parser.error("--variant est requis pour --prepare et --measure")
    if args.measure and not args.board_port:
        parser.error("--board-port est requis pour --measure")

    if args.prepare:
        prepare(args)
    if args.measure:
        measure(args)
    if args.refit:
        for name, cells in refit(args.out).items():
            for cell_name, cell in cells.items():
                print(f"[isolation] {name}/{cell_name} réajusté : pente="
                      f"{cell.get('slope_ua_per_hz', float('nan')):+.3f} ± "
                      f"{cell.get('slope_std_ua_per_hz', float('nan')):.3f} µA/Hz, "
                      f"r²={cell.get('r2', float('nan')):.3f}")

    if args.summary or args.measure or args.refit:
        summary = build_summary(args.out)
        (args.out / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        for nom, paire in summary["pairs"].items():
            print(f"[isolation] paire {nom} : {paire['verdict']} — {paire['rationale']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
