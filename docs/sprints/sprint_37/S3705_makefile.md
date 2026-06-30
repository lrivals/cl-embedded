# S3705 — Cibles Makefile (déclencheur local)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 37 |
| **Priorité** | 🟢 Confort — point d'entrée simple du déclencheur manuel validé. |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 0.5h |
| **Fichiers cibles** | `Makefile` (racine, nouveau) |
| **Dépendances** | `scripts/prepare_gitlab_release.py` (S3703) ✅ |

## Contexte

Le mécanisme retenu est un **déclencheur local manuel**. Un `Makefile` racine offre les entrées
sans interférer avec le build firmware (`firmware/stm32f4_blink/Makefile`, inchangé).

## Spec

- `make gitlab-release-dry` → `prepare_gitlab_release.py --dry-run` (plan, aucune écriture).
- `make gitlab-release` → `prepare_gitlab_release.py --run-tests $(ARGS)` (tests + export + gate +
  commit snapshot ; `ARGS=--push` pour pousser).
- `make gitlab-check` → `prepare_gitlab_release.py --check-only` (garde-fou, sans commit).
- `make help` documente les cibles. `PYTHON ?= python` surchargeable.
