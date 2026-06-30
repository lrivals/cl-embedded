# S3704 — Docs d'onboarding neutres (`docs/gitlab/`)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 37 |
| **Priorité** | 🟠 Importante — remplace le rôle de `CLAUDE.md` pour les prochains contributeurs, sans aucune trace IA. |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 2h |
| **Fichiers cibles** | `docs/gitlab/README_gitlab.md`, `docs/gitlab/CONTRIBUTING.md` |
| **Dépendances** | contenu factuel de `README.md`/`CLAUDE.md` reformulé neutre |

## Contexte

La version GitLab exclut `CLAUDE.md` et `skills/`. Les nouveaux utilisateurs ont besoin d'un
onboarding professionnel : ces gabarits sont déposés dans l'export (`neutral_docs`) comme
`README.md` et `CONTRIBUTING.md`.

## Spec

- `README_gitlab.md` : overview projet, modèles M1–M4 + baselines, triple gap, quick start
  (training PC + firmware NUCLEO-F439ZI), structure du dépôt, contraintes de design — **sans**
  mention d'assistant IA, de `CLAUDE.md` ni de `graphify`.
- `CONTRIBUTING.md` : setup env, plan du dépôt, conventions (black/ruff, configs-over-constants,
  reproductibilité, annotations `# MEM:`, headers de poids générés, protocole UART), lancement des
  tests (pytest + Unity), contraintes hardware, workflow de contribution.

## Note

Les gabarits vivent sous `docs/gitlab/` qui est lui-même dans `exclude_paths` : ils ne sont pas
dupliqués dans l'export (seules leurs destinations racine `README.md`/`CONTRIBUTING.md` y figurent).
