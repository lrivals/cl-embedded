# S3708 — Tests (`tests/test_gitlab_release.py`)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 37 |
| **Priorité** | 🟠 Importante — verrouille le comportement de la transformation. |
| **Statut** | ✅ Implémenté (12/12 PASS) |
| **Durée estimée** | 2h |
| **Fichiers cibles** | `tests/test_gitlab_release.py` |
| **Dépendances** | S3702 + S3703 ✅ |

## Contexte

Tests sans réseau ni `data/` : un mini-repo git est construit en tmp avec des fichiers propres et
« sales » (CLAUDE.md, skills/, graphify-out/, doc avec mentions internes).

## Couverture (12 tests)

- `exclude_paths` absents de l'export ; fichiers propres conservés.
- `replace` neutralise `CLAUDE.md` dans un commentaire `.py` (ligne de code préservée).
- `drop_blocks`/`drop_line_patterns` retirent section « Graphe » + lignes graphify d'un `.md` conservé.
- docs neutres déposées (README de l'export = gabarit neutre).
- scanner : détecte une trace semée / valide un arbre propre.
- export passe le gate (`scan_tree == []`).
- `--dry-run` n'écrit rien ; `--check-only` vert si couvert / rouge si trace non couverte.
- export **idempotent** (deux passes → contenu identique).

## Exécution

```bash
pytest tests/test_gitlab_release.py -v      # 12 PASS
pytest tests/ --collect-only -q             # 652 tests collectés, 0 erreur
```
