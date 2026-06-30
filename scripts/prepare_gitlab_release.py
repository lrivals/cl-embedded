"""prepare_gitlab_release.py — Transformation "repo source → version GitLab".

Produit, dans un **dépôt git séparé** (hors du repo courant), une copie propre et
professionnelle du projet, débarrassée de toute trace d'outillage interne / IA
générative, prête à être poussée vers le GitLab ISAE-SUPAERO.

Le repo source n'est **jamais** modifié ni poussé : seule la version exportée l'est,
et toujours via cette étape de transformation.

Étapes :
  1. Énumère les fichiers **suivis** par git (``git ls-files``) — pas de data, pas
     de fichiers ignorés/non versionnés.
  2. Retire les ``exclude_paths`` (CLAUDE.md, skills/, graphify-out/, .claude/, …).
  3. Copie le reste vers ``output_dir``.
  4. Applique les ``rewrite_rules`` (suppression de lignes/sections mentionnant
     l'outillage interne dans les docs conservées).
  5. Dépose les **docs neutres** (README/CONTRIBUTING générés).
  6. **Gate dur** : lance ``check_ai_traces.py`` sur l'export ; abandonne si trace.
  7. Initialise/Met à jour le dépôt git séparé et commit un snapshot propre
     (message neutre, sans footer). ``--push`` optionnel.

Usage :
    python scripts/prepare_gitlab_release.py [--dry-run] [--output-dir DIR]
        [--run-tests] [--push] [--config PATH]

Voir docs/gitlab_publication.md pour le workflow complet.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_ai_traces import load_config, scan_tree  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── git helpers ────────────────────────────────────────────────────────────────

def _git(args: list[str], cwd: Path) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, check=True,
                         capture_output=True, text=True)
    return out.stdout


def tracked_files(repo: Path) -> list[str]:
    """Liste les chemins suivis par git, relatifs à ``repo``."""
    return [ln for ln in _git(["ls-files"], repo).splitlines() if ln]


# ── filtrage / réécriture ───────────────────────────────────────────────────────

def is_excluded(rel: str, exclude_paths: list[str]) -> bool:
    for pre in exclude_paths:
        pre = pre.rstrip("/")
        if rel == pre or rel.startswith(pre + "/"):
            return True
    return False


def apply_rewrites(text: str, rel: str, rules: list[dict]) -> str:
    """Applique les règles de réécriture (replace/bloc/ligne) à un fichier conservé.

    Le matching des chemins utilise ``fnmatch`` (``*`` traverse les ``/``) : ``"*"``
    cible tous les fichiers, ``"*.md"`` tout markdown à n'importe quelle profondeur.
    """
    for rule in rules or []:
        if not fnmatch.fnmatch(rel, rule.get("files", "")):
            continue
        # (1) Substitutions regex → texte (neutralise sans supprimer la ligne).
        for sub in rule.get("replace", []) or []:
            text = re.sub(sub["pattern"], sub["repl"], text, flags=re.IGNORECASE)
        # (2) Suppression de blocs (sections markdown délimitées).
        for block in rule.get("drop_blocks", []) or []:
            start_rx = re.compile(block["start"], re.MULTILINE)
            end_rx = re.compile(block["end"], re.MULTILINE)
            text = _drop_block(text, start_rx, end_rx)
        # (3) Suppression de lignes entières.
        drops = [re.compile(p) for p in rule.get("drop_line_patterns", []) or []]
        if drops:
            kept = [ln for ln in text.splitlines()
                    if not any(rx.search(ln) for rx in drops)]
            text = "\n".join(kept) + ("\n" if text.endswith("\n") else "")
    return text


def _drop_block(text: str, start_rx: re.Pattern, end_rx: re.Pattern) -> str:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    skipping = False
    for ln in lines:
        if not skipping and start_rx.search(ln):
            skipping = True
            continue
        if skipping:
            if end_rx.search(ln):
                skipping = False
                out.append(ln)  # la borne de fin est conservée
            continue
        out.append(ln)
    return "".join(out)


# ── plan / exécution ────────────────────────────────────────────────────────────

def build_plan(repo: Path, config: dict) -> dict:
    """Calcule le plan de transformation sans rien écrire."""
    exclude_paths = config.get("exclude_paths", [])
    files = tracked_files(repo)
    kept, excluded = [], []
    for rel in files:
        (excluded if is_excluded(rel, exclude_paths) else kept).append(rel)
    neutral = config.get("neutral_docs", [])
    return {"kept": kept, "excluded": excluded, "neutral_docs": neutral}


def _clean_output_tree(out_dir: Path) -> None:
    """Vide le contenu de l'export en préservant le ``.git`` du dépôt séparé."""
    for child in out_dir.iterdir():
        if child.name == ".git":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def export(repo: Path, out_dir: Path, config: dict) -> dict:
    """Réalise l'export sanitisé dans ``out_dir`` et retourne le plan."""
    plan = build_plan(repo, config)
    rules = config.get("rewrite_rules", [])

    out_dir.mkdir(parents=True, exist_ok=True)
    _clean_output_tree(out_dir)

    text_suffixes = {".md", ".py", ".txt", ".yaml", ".yml", ".c", ".h", ".s",
                     ".cfg", ".toml", ".ini", ".json", ".rst", ".ipynb"}
    for rel in plan["kept"]:
        src = repo / rel
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix.lower() in text_suffixes:
            try:
                text = src.read_text(encoding="utf-8")
                dst.write_text(apply_rewrites(text, rel, rules), encoding="utf-8")
                continue
            except UnicodeDecodeError:
                pass
        shutil.copy2(src, dst)

    # Docs neutres (écrasent README.md/CONTRIBUTING.md de l'export).
    for entry in plan["neutral_docs"]:
        src = repo / entry["source"]
        dst = out_dir / entry["dest"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    return plan


def commit_snapshot(out_dir: Path, config: dict, push: bool) -> None:
    """Initialise si besoin le dépôt séparé et commit un snapshot propre."""
    rel = config.get("release", {})
    if not (out_dir / ".git").exists():
        _git(["init", "-q"], out_dir)
        _git(["checkout", "-q", "-B", "main"], out_dir)
    _git(["config", "user.name", rel.get("git_user_name", "CL-Embedded")], out_dir)
    _git(["config", "user.email", rel.get("git_user_email", "cl-embedded@isae-supaero.fr")], out_dir)
    _git(["add", "-A"], out_dir)
    status = _git(["status", "--porcelain"], out_dir)
    if not status.strip():
        print("[prepare_gitlab_release] Aucun changement à committer (export identique).")
        return
    msg = f"{rel.get('commit_message', 'Release snapshot')} {date.today().isoformat()}"
    _git(["commit", "-q", "-m", msg], out_dir)
    print(f"[prepare_gitlab_release] Commit créé dans {out_dir} : « {msg} »")

    if push:
        remote = rel.get("remote_name", "gitlab")
        remotes = _git(["remote"], out_dir).split()
        if remote not in remotes:
            print(f"[prepare_gitlab_release] ⚠ remote '{remote}' absent. Configurez d'abord :\n"
                  f"    git -C {out_dir} remote add {remote} <URL_GITLAB>")
            return
        _git(["push", remote, "main"], out_dir)
        print(f"[prepare_gitlab_release] Poussé vers {remote}/main.")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None,
                        help="Chemin de gitlab_release.yaml.")
    parser.add_argument("--output-dir", default=None,
                        help="Répertoire de sortie (défaut : output_dir de la config).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Affiche le plan sans rien écrire.")
    parser.add_argument("--check-only", action="store_true",
                        help="Construit l'export dans un dossier jetable et applique le gate "
                             "(scan), sans commit ni push. Garde-fou pour les ajouts futurs.")
    parser.add_argument("--run-tests", action="store_true",
                        help="Lance `pytest -q` côté source avant l'export (feature testée).")
    parser.add_argument("--push", action="store_true",
                        help="Pousse le snapshot vers le remote GitLab après commit.")
    args = parser.parse_args(argv)

    config_path = Path(args.config) if args.config else REPO_ROOT / "configs" / "gitlab_release.yaml"
    config = load_config(config_path)

    out_dir = Path(args.output_dir) if args.output_dir else Path(config["output_dir"])
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()

    plan = build_plan(REPO_ROOT, config)
    print(f"[prepare_gitlab_release] {len(plan['kept'])} fichiers conservés, "
          f"{len(plan['excluded'])} exclus, {len(plan['neutral_docs'])} docs neutres.")
    print(f"[prepare_gitlab_release] Sortie : {out_dir}")

    if args.dry_run:
        print("\n--- EXCLUS ---")
        for rel in plan["excluded"]:
            print(f"  ✗ {rel}")
        print("\n--- DOCS NEUTRES ---")
        for entry in plan["neutral_docs"]:
            print(f"  + {entry['source']} → {entry['dest']}")
        print("\n[dry-run] Aucune écriture effectuée.")
        return 0

    # Mode garde-fou : export jetable + gate uniquement (ajouts futurs).
    if args.check_only:
        tmp_dir = Path(tempfile.mkdtemp(prefix="gitlab-release-check-"))
        try:
            export(REPO_ROOT, tmp_dir, config)
            findings = scan_tree(tmp_dir, config, source_mode=False)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if findings:
            print(f"[prepare_gitlab_release] ÉCHEC garde-fou — {len(findings)} trace(s) non couverte(s) :")
            for f in findings[:30]:
                loc = f"{f.path}:{f.line_no}" if f.line_no else f.path
                print(f"  - {loc}  [{f.pattern}]  {f.excerpt}")
            print("Un ajout introduit une trace non couverte. Ajoutez un exclude_path / rewrite_rule "
                  "dans configs/gitlab_release.yaml.")
            return 1
        print("[prepare_gitlab_release] Garde-fou OK : l'export sortirait propre (0 trace).")
        return 0

    if args.run_tests:
        print("[prepare_gitlab_release] pytest -q …")
        rc = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=REPO_ROOT).returncode
        if rc != 0:
            print("[prepare_gitlab_release] ÉCHEC tests — export annulé.")
            return rc

    export(REPO_ROOT, out_dir, config)

    # Gate dur : aucune trace tolérée dans l'export.
    findings = scan_tree(out_dir, config, source_mode=False)
    if findings:
        print(f"[prepare_gitlab_release] ÉCHEC — {len(findings)} trace(s) résiduelle(s) :")
        for f in findings[:30]:
            loc = f"{f.path}:{f.line_no}" if f.line_no else f.path
            print(f"  - {loc}  [{f.pattern}]  {f.excerpt}")
        print("Corrigez configs/gitlab_release.yaml (exclude_paths/rewrite_rules) puis relancez.")
        return 1
    print(f"[prepare_gitlab_release] Scan propre : 0 trace dans {out_dir}.")

    commit_snapshot(out_dir, config, push=args.push)
    if not args.push:
        remote = config.get("release", {}).get("remote_name", "gitlab")
        print("\nPour publier vers GitLab :")
        print(f"    git -C {out_dir} remote add {remote} <URL_GITLAB>   # 1ère fois")
        print(f"    git -C {out_dir} push {remote} main")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
