# S3701 — `configs/gitlab_release.yaml` (source de vérité de la transformation)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 37 |
| **Priorité** | 🔴 Critique — toute la transformation est pilotée par ce fichier (couverture des ajouts futurs sans toucher au code). |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 2h |
| **Fichiers cibles** | `configs/gitlab_release.yaml` |
| **Dépendances** | PyYAML ✅ · inventaire des traces ✅ |

## Contexte

Centraliser dans **un seul fichier** ce qui distingue le dépôt de travail de la version GitLab :
chemins exclus, motifs interdits, réécritures, docs neutres, métadonnées de release. Un nouvel
ajout introduisant une trace se gère **ici** (un chemin ou un pattern), jamais dans le code.

## Spec

- `output_dir` : dépôt exporté séparé (défaut `../cl-embedded-gitlab`), surchargeable en CLI.
- `exclude_paths` : `CLAUDE.md`, `.claude/`, `skills/`, `graphify-out/`, le workflow garde-fou,
  `docs/gitlab/` (gabarits sources). Préfixe terminé par `/` ⇒ tout le sous-arbre.
- `forbidden_patterns` : regex insensibles à la casse avec **frontières de mot** (`\bclaude\b`,
  `\banthropic\b`, `\bgraphify\b`, `\bco-authored-by\b`, `\bclaude-…`, `\.claude/`,
  `noreply@anthropic\.com`, emoji 🤖, `generated with .*claude`). Les `\b` évitent les faux
  positifs dans les blobs base64 des notebooks.
- `rewrite_rules` (ordre : replace → drop_blocks → drop_line_patterns ; matching `fnmatch`,
  `*` traverse les `/`) :
  - `files: "*"` → `replace` : `CLAUDE\.md` → « les conventions du projet » (neutralise les
    commentaires `.py`/`.ipynb` sans casser la ligne) ;
  - `files: "*.md"` → `drop_line_patterns` (graphify, `.claude/`, `\bclaude\b`, `\banthropic\b`,
    `skills/`) + `drop_blocks` (section « Graphe de connaissance »).
- `allowlist` : faux positifs tolérés (vide par défaut).
- `neutral_docs` : `docs/gitlab/README_gitlab.md` → `README.md`, `docs/gitlab/CONTRIBUTING.md` →
  `CONTRIBUTING.md`.
- `release` : message neutre, identité git, nom du remote (`gitlab`).
