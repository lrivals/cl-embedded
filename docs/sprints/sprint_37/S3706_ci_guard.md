# S3706 — Garde-fou CI ajouts futurs (`ai-trace-guard.yml`)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 37 |
| **Priorité** | 🟠 Importante — c'est ce qui rend la transformation valable pour les **ajouts futurs**, pas seulement le code existant. |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 1h |
| **Fichiers cibles** | `.github/workflows/ai-trace-guard.yml` |
| **Dépendances** | `prepare_gitlab_release.py --check-only` (S3703) ✅ |

## Contexte

Le dépôt de travail contient **légitimement** des références internes partout (CLAUDE.md, skills/,
graphify, commentaires). Un scan brut du source serait donc toujours rouge et inutile. L'invariant
pertinent est : **« l'export GitLab sort-il toujours propre ? »**.

## Spec

- Déclencheurs : `pull_request` + `push` sur `main`.
- `checkout` avec `fetch-depth: 0` (nécessaire à `git ls-files`).
- Python 3.10, `pip install pyyaml`.
- Étape unique : `python scripts/prepare_gitlab_release.py --check-only` → construit l'export dans
  un dossier jetable, applique le gate, **0 commit**. Échoue si un ajout introduit une trace non
  couverte par les règles → on ajoute alors un `exclude_path`/`rewrite_rule` dans la config.
- Équivalent local : `make gitlab-check`.

## Note

Ce workflow est lui-même dans `exclude_paths` (garde-fou interne, inutile sur GitLab).
