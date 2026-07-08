#!/usr/bin/env python3
"""generate_figures.py — régénère les figures des catalogues `src/figures/` (S4201).

Usage :
    python scripts/generate_figures.py --list
    python scripts/generate_figures.py --catalog quantization            # préfixe → tous les sous-catalogues
    python scripts/generate_figures.py --catalog quantization/pedagogy
    python scripts/generate_figures.py --all --style manuscript --out docs/figures

Idempotent : seed fixé, mêmes sources → mêmes PNG. La provenance (expériences
JSON, checkpoints) de chaque catalogue est affichée en fin d'exécution.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.figures.catalogs  # noqa: F401,E402  — auto-enregistrement des catalogues
from src.figures import registry  # noqa: E402
from src.figures.loaders import consume_session_sources  # noqa: E402
from src.figures.style import DEFAULT_OUT_ROOT, apply_style  # noqa: E402
from src.utils.reproducibility import set_seed  # noqa: E402


def resolve_catalogs(name: str) -> list[str]:
    """Nom exact ou préfixe (``quantization`` → tous les ``quantization/*``)."""
    matched = registry.match_catalogs(name)
    if not matched:
        raise SystemExit(
            f"Aucun catalogue ne correspond à {name!r} — disponibles : "
            f"{registry.list_catalogs()}"
        )
    return matched


def run_catalogs(names: list[str], out_root: Path) -> list[Path]:
    """Exécute les catalogues, affiche PNG produits + provenance, retourne les chemins."""
    all_paths: list[Path] = []
    consume_session_sources()  # remet la provenance de session à zéro
    for name in names:
        print(f"\n=== Catalogue {name} ===")
        paths = registry.get_catalog(name)(out_root)
        sources = consume_session_sources()
        all_paths.extend(paths)
        print(f"[figures] {len(paths)} figure(s) — sources :")
        for src in sources:
            print(f"  · {src}")
    return all_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", type=str, default=None,
                        help="Catalogue à régénérer (préfixe accepté : `quantization`)")
    parser.add_argument("--all", action="store_true", help="Régénère tous les catalogues")
    parser.add_argument("--list", action="store_true", help="Liste les catalogues enregistrés")
    parser.add_argument("--style", choices=["slide", "manuscript"], default="slide",
                        help="Preset de style (défaut : slide)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_ROOT,
                        help="Racine de sortie (défaut : docs/figures/)")
    args = parser.parse_args()

    if args.list:
        for name in registry.list_catalogs():
            print(name)
        return

    if not args.all and args.catalog is None:
        parser.error("préciser --catalog <nom>, --all ou --list")

    names = registry.list_catalogs() if args.all else resolve_catalogs(args.catalog)

    set_seed(42)
    apply_style(args.style)
    paths = run_catalogs(names, args.out)
    print(f"\n[figures] Total : {len(paths)} PNG sous {args.out}")


if __name__ == "__main__":
    main()
