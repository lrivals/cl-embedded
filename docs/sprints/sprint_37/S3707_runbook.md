# S3707 — Runbook de publication (`docs/gitlab_publication.md`)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 37 |
| **Priorité** | 🟠 Importante — mode d'emploi du processus pour l'auteur et les repreneurs. |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 1.5h |
| **Fichiers cibles** | `docs/gitlab_publication.md` |
| **Dépendances** | S3701–S3706 ✅ |

## Contexte

Documenter le workflow de bout en bout, la première mise en place GitLab, et surtout **comment
gérer les ajouts futurs** (où ajouter une règle quand une nouvelle trace apparaît).

## Spec (contenu)

- **Principe** + schéma : dépôt de travail → transformation → dépôt séparé → push manuel.
- **Quand publier** : feature complète et testée (les tests tournent avant l'export).
- **Composants** : tableau fichier→rôle.
- **Workflow** : `make gitlab-release-dry` → `make gitlab-release` → 1ʳᵉ config remote `gitlab` →
  `git push` (ou `ARGS=--push`).
- **Ajouts futurs** : nouveau fichier interne → `exclude_paths` ; nouveau marqueur →
  `forbidden_patterns` ; doc utile à nettoyer → `rewrite_rules` ; faux positif → `allowlist` ;
  garde-fou `make gitlab-check` / CI.
- **Garanties** : `.git` du source intact, dépôt séparé sans historique IA, idempotence.
