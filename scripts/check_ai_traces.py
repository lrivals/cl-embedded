"""check_ai_traces.py — Scanner de traces d'IA générative.

Parcourt un arbre de fichiers et signale toute trace interdite (mentions
d'assistant IA, footers de co-auteur, outillage interne) définie dans
``configs/gitlab_release.yaml``. Sert à deux endroits :

  1. **Gate dur** appelé par ``prepare_gitlab_release.py`` sur l'export GitLab :
     l'export échoue si une trace subsiste.
  2. **Garde-fou CI / pré-commit** sur le repo source pour repérer tôt les
     nouvelles traces (mode ``--source`` qui ignore les zones connues légitimes).

Exit code 0 si propre, 1 si au moins une trace est détectée.

Usage :
    python scripts/check_ai_traces.py [TREE] [--config PATH] [--source] [--quiet]

Exemples :
    python scripts/check_ai_traces.py ../cl-embedded-gitlab
    python scripts/check_ai_traces.py . --source        # check du repo source
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

# Extensions binaires / volumineuses ignorées par le scan de contenu.
_SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".ico", ".svg",
    ".pkl", ".pt", ".pth", ".onnx", ".tflite", ".bin", ".elf", ".map",
    ".zip", ".gz", ".npy", ".npz", ".parquet", ".woff", ".woff2", ".ttf",
}
# Répertoires jamais scannés (caches, vendored, données).
_SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".mypy_cache",
              ".pytest_cache", ".ruff_cache", "data"}


@dataclass
class Finding:
    path: str
    line_no: int
    pattern: str
    excerpt: str


def load_config(config_path: Path) -> dict:
    """Charge la config YAML de release."""
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _iter_text_files(root: Path) -> Iterable[Path]:
    """Itère les fichiers texte candidats sous ``root``."""
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.suffix.lower() in _SKIP_SUFFIXES:
            continue
        yield path


def _compile_patterns(patterns: list[str]) -> list[tuple[str, re.Pattern]]:
    return [(p, re.compile(p, re.IGNORECASE)) for p in patterns]


def _allowed(rel_path: str, pattern: str, allowlist: list[dict]) -> bool:
    """Retourne True si (chemin, pattern) est explicitement toléré."""
    for entry in allowlist or []:
        glob = entry.get("files", "")
        tolerated = entry.get("patterns", [])
        if Path(rel_path).match(glob) and pattern in tolerated:
            return True
    return False


def scan_tree(root: Path, config: dict, source_mode: bool = False) -> list[Finding]:
    """Scanne ``root`` et retourne la liste des traces trouvées.

    Parameters
    ----------
    root : Path
        Racine de l'arbre à scanner.
    config : dict
        Config de release (patterns, allowlist, exclude_paths).
    source_mode : bool
        Si True, les ``exclude_paths`` (zones internes connues : CLAUDE.md,
        skills/, graphify-out/, .claude/) sont ignorées du scan plutôt que
        signalées — utile pour le repo source où ces fichiers existent
        légitimement. En mode export (False), ces chemins ne devraient plus
        exister et un fichier résiduel est signalé.
    """
    findings: list[Finding] = []
    compiled = _compile_patterns(config.get("forbidden_patterns", []))
    allowlist = config.get("allowlist", [])
    exclude_prefixes = [p.rstrip("/") for p in config.get("exclude_paths", [])]

    for path in _iter_text_files(root):
        rel = path.relative_to(root).as_posix()

        in_excluded = any(rel == pre or rel.startswith(pre + "/") for pre in exclude_prefixes)
        if in_excluded:
            if source_mode:
                continue  # zone interne connue, tolérée côté source
            # En mode export, un fichier exclu encore présent est une anomalie.
            findings.append(Finding(rel, 0, "<exclude_path résiduel>", rel))
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        for line_no, line in enumerate(text.splitlines(), start=1):
            for raw, rx in compiled:
                if rx.search(line) and not _allowed(rel, raw, allowlist):
                    findings.append(Finding(rel, line_no, raw, line.strip()[:120]))

    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tree", nargs="?", default=".",
                        help="Répertoire à scanner (défaut : repo courant).")
    parser.add_argument("--config", default=None,
                        help="Chemin de gitlab_release.yaml (défaut : configs/gitlab_release.yaml).")
    parser.add_argument("--source", action="store_true",
                        help="Mode source : ignore les zones internes connues (CLAUDE.md, skills/, …).")
    parser.add_argument("--quiet", action="store_true", help="N'affiche que le verdict final.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    config_path = Path(args.config) if args.config else repo_root / "configs" / "gitlab_release.yaml"
    config = load_config(config_path)

    root = Path(args.tree).resolve()
    findings = scan_tree(root, config, source_mode=args.source)

    if findings:
        if not args.quiet:
            print(f"[check_ai_traces] {len(findings)} trace(s) détectée(s) dans {root} :")
            for f in findings:
                loc = f"{f.path}:{f.line_no}" if f.line_no else f.path
                print(f"  - {loc}  [{f.pattern}]  {f.excerpt}")
        print(f"[check_ai_traces] ÉCHEC — {len(findings)} trace(s).")
        return 1

    print(f"[check_ai_traces] OK — aucune trace dans {root}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
