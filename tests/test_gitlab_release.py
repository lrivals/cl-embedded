"""Tests Sprint 37 / S3708 — pipeline de publication GitLab.

Verrouille le comportement de la transformation "repo source → version GitLab" :
  * les ``exclude_paths`` sont absents de l'export ;
  * les ``rewrite_rules`` retirent les lignes/sections d'outillage interne ;
  * les docs neutres sont déposées ;
  * le scanner détecte une trace semée (exit≠0) et valide un arbre propre ;
  * ``--dry-run`` n'écrit rien ;
  * l'export est idempotent.

Aucune dépendance réseau ni `data/`. On construit un mini-repo git en tmp.

Exécution :
    pytest tests/test_gitlab_release.py -v
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import check_ai_traces as cat  # noqa: E402
import prepare_gitlab_release as pgr  # noqa: E402


# ── fixtures ────────────────────────────────────────────────────────────────────

CONFIG = {
    "output_dir": "../out",
    "exclude_paths": ["CLAUDE.md", "skills/", "graphify-out/", ".claude/"],
    "forbidden_patterns": ["claude", "anthropic", "co-authored-by", "graphify"],
    "rewrite_rules": [
        {"files": "*", "replace": [{"pattern": r"CLAUDE\.md", "repl": "les conventions du projet"}]},
        {
            "files": "*.md",
            "drop_line_patterns": [r"(?i).*graphify.*"],
            "drop_blocks": [{"start": r"(?i)^#+\s*Graphe", "end": r"^(#{1,2}\s|---\s*$)"}],
        },
    ],
    "allowlist": [],
    "neutral_docs": [{"source": "docs/gitlab/README_gitlab.md", "dest": "README.md"}],
    "release": {"commit_message": "Release snapshot", "git_user_name": "T",
                "git_user_email": "t@t.t", "remote_name": "gitlab"},
}


def _git(args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture()
def src_repo(tmp_path: Path) -> Path:
    """Mini-repo git source avec quelques fichiers propres et 'sales'."""
    repo = tmp_path / "src"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "clean.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "CLAUDE.md").write_text("internal\n", encoding="utf-8")
    (repo / "skills").mkdir()
    (repo / "skills" / "s.md").write_text("a skill\n", encoding="utf-8")
    (repo / "graphify-out").mkdir()
    (repo / "graphify-out" / "g.json").write_text("{}\n", encoding="utf-8")
    # Doc conservée mais contenant des références internes à nettoyer.
    (repo / "README.md").write_text(
        "# Title\n\nSee CLAUDE.md for context.\n\nRun graphify to update.\n\n"
        "## Graphe de connaissance\n\nlots of graphify stuff\n\n## Next\n\nkeep me\n",
        encoding="utf-8",
    )
    # gabarit doc neutre
    (repo / "docs").mkdir()
    (repo / "docs" / "gitlab").mkdir()
    (repo / "docs" / "gitlab" / "README_gitlab.md").write_text("# Clean Title\n", encoding="utf-8")

    _git(["init", "-q"], repo)
    _git(["config", "user.email", "t@t.t"], repo)
    _git(["config", "user.name", "t"], repo)
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "init"], repo)
    return repo


# ── tests ───────────────────────────────────────────────────────────────────────

def test_excluded_paths_absent_from_export(src_repo, tmp_path):
    out = tmp_path / "out"
    pgr.export(src_repo, out, CONFIG)
    assert not (out / "CLAUDE.md").exists()
    assert not (out / "skills").exists()
    assert not (out / "graphify-out").exists()
    assert (out / "src" / "clean.py").exists()


def test_rewrite_rules_strip_internal_lines(src_repo, tmp_path):
    out = tmp_path / "out"
    pgr.export(src_repo, out, CONFIG)
    # README.md est écrasé par le doc neutre → on teste une autre md conservée :
    # ici le neutral_doc remplace README, donc vérifions le contenu neutre.
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "graphify" not in readme.lower()
    assert "claude" not in readme.lower()


def test_rewrite_applied_when_no_neutral_override(src_repo, tmp_path):
    # Ajoute une doc conservée NON remplacée par un neutral_doc.
    (src_repo / "docs" / "guide.md").write_text(
        "# Guide\n\nUse graphify here.\n\n## Graphe de connaissance\n\nx\n\n## Keep\n\nyes\n",
        encoding="utf-8",
    )
    _git(["add", "-A"], src_repo)
    _git(["commit", "-q", "-m", "guide"], src_repo)
    out = tmp_path / "out"
    pgr.export(src_repo, out, CONFIG)
    guide = (out / "docs" / "guide.md").read_text(encoding="utf-8")
    assert "graphify" not in guide.lower()
    assert "Graphe de connaissance" not in guide
    assert "## Keep" in guide and "yes" in guide


def test_replace_neutralizes_code_comment(src_repo, tmp_path):
    # Référence interne dans un commentaire .py → neutralisée, pas supprimée.
    (src_repo / "src" / "rule.py").write_text(
        "# Règle CLAUDE.md : tout module mesuré est profilé.\nx = 2\n", encoding="utf-8")
    _git(["add", "-A"], src_repo)
    _git(["commit", "-q", "-m", "rule"], src_repo)
    out = tmp_path / "out"
    pgr.export(src_repo, out, CONFIG)
    code = (out / "src" / "rule.py").read_text(encoding="utf-8")
    assert "CLAUDE.md" not in code
    assert "les conventions du projet" in code
    assert "x = 2" in code  # la ligne de code est préservée


def test_neutral_docs_generated(src_repo, tmp_path):
    out = tmp_path / "out"
    pgr.export(src_repo, out, CONFIG)
    assert (out / "README.md").read_text(encoding="utf-8").startswith("# Clean Title")


def test_scanner_detects_seeded_trace(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "f.py").write_text("# made with Claude\nx = 1\n", encoding="utf-8")
    findings = cat.scan_tree(tree, CONFIG, source_mode=False)
    assert any(f.pattern == "claude" for f in findings)


def test_scanner_clean_tree(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "f.py").write_text("x = 1\n", encoding="utf-8")
    assert cat.scan_tree(tree, CONFIG, source_mode=False) == []


def test_export_passes_gate(src_repo, tmp_path):
    out = tmp_path / "out"
    pgr.export(src_repo, out, CONFIG)
    assert cat.scan_tree(out, CONFIG, source_mode=False) == []


def test_dry_run_writes_nothing(src_repo, tmp_path, monkeypatch):
    out = tmp_path / "out"
    cfg = tmp_path / "cfg.yaml"
    import yaml
    cfg.write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
    monkeypatch.setattr(pgr, "REPO_ROOT", src_repo)
    rc = pgr.main(["--config", str(cfg), "--output-dir", str(out), "--dry-run"])
    assert rc == 0
    assert not out.exists()


def test_check_only_passes_on_covered_repo(src_repo, tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yaml"
    import yaml
    cfg.write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
    monkeypatch.setattr(pgr, "REPO_ROOT", src_repo)
    rc = pgr.main(["--config", str(cfg), "--check-only"])
    assert rc == 0


def test_check_only_fails_on_uncovered_trace(src_repo, tmp_path, monkeypatch):
    # Ajoute un fichier conservé avec une trace qu'aucune règle ne couvre.
    (src_repo / "src" / "leak.py").write_text("# anthropic helper\n", encoding="utf-8")
    _git(["add", "-A"], src_repo)
    _git(["commit", "-q", "-m", "leak"], src_repo)
    cfg = tmp_path / "cfg.yaml"
    import yaml
    cfg.write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
    monkeypatch.setattr(pgr, "REPO_ROOT", src_repo)
    rc = pgr.main(["--config", str(cfg), "--check-only"])
    assert rc == 1


def test_export_idempotent(src_repo, tmp_path):
    out = tmp_path / "out"
    pgr.export(src_repo, out, CONFIG)
    first = {p.relative_to(out).as_posix(): p.read_bytes()
             for p in out.rglob("*") if p.is_file()}
    pgr.export(src_repo, out, CONFIG)
    second = {p.relative_to(out).as_posix(): p.read_bytes()
              for p in out.rglob("*") if p.is_file()}
    assert first == second
