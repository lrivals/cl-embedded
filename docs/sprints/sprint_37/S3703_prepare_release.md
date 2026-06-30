# S3703 — `scripts/prepare_gitlab_release.py` (transformation → dépôt séparé)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 37 |
| **Priorité** | 🔴 Critique — c'est l'étape de transformation obligatoire (le dépôt de travail n'est jamais poussé tel quel). |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 4h |
| **Fichiers cibles** | `scripts/prepare_gitlab_release.py` |
| **Dépendances** | `check_ai_traces.py` (S3702) ✅ · `configs/gitlab_release.yaml` (S3701) ✅ · docs neutres (S3704) ✅ · `git` ✅ |

## Contexte

Produit, dans un **dépôt git séparé** (hors du dépôt de travail), une copie propre prête à pousser
vers GitLab. Le dépôt de travail n'est jamais modifié.

## Spec

1. **Énumère les fichiers suivis** (`git ls-files`) — exclut data/ignorés/non versionnés.
2. **Retire** les `exclude_paths`.
3. **Copie** le reste ; pour les fichiers texte (incl. `.ipynb`), applique `apply_rewrites`
   (replace → drop_blocks → drop_line_patterns, matching `fnmatch`).
4. **Dépose les docs neutres** (écrasent `README.md`/`CONTRIBUTING.md`).
5. **Gate dur** : `scan_tree` sur l'export ; abandon (exit 1) + rapport si trace résiduelle.
6. **Dépôt séparé** : `git init` si absent (préserve un `.git` existant), commit snapshot **neutre**
   sans footer ; `--push` optionnel vers le remote `gitlab`.

## CLI

- `--dry-run` : plan (exclus / docs neutres) sans écriture.
- `--check-only` : export jetable (tempdir) + gate, **sans commit ni push** → garde-fou ajouts futurs.
- `--run-tests` : `pytest -q` côté source avant export (matérialise « feature testée »).
- `--output-dir`, `--config`, `--push`.

## Garanties

- `_clean_output_tree` vide le contenu mais **préserve `.git`** du dépôt séparé.
- Idempotence : sans changement source, `git status` vide → aucun nouveau commit.
- Aucun historique du dépôt de travail (donc aucun footer `Co-Authored-By`) n'atteint GitLab.
