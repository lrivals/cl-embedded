# S4007 — Tests, build & clôture du sprint

| Champ | Valeur |
|-------|--------|
| **Sprint** | 40 |
| **Priorité** | 🟡 Importante — garde-fous & mise à jour docs |
| **Statut** | ✅ Implémenté — `TestArticleCoherence` 14 PASS / 2 skips honnêtes, firmware 0 régression, roadmap+triple_gap+CLAUDE.md à jour |
| **Durée estimée** | ~5h |
| **Dépendances** | S4001–S4006 |
| **Fichiers cibles** | `tests/test_sprint40_article.py`, `docs/roadmap_phase2.md`, `docs/triple_gap.md`, `CLAUDE.md` |
| **Références** | S3913/S3914 (clôture Sprint 39) · `skills/graphify_sprint_update.md` |

## Contexte

Garantir la cohérence de l'article (figures↔données, FR≡EN), l'absence de régression firmware, et mettre à
jour la documentation de suivi.

## Spec

### Tests Python (`tests/test_sprint40_article.py`)
| Test | Vérifie |
|------|---------|
| `test_figures_match_json` | chaque valeur affichée dérive d'un JSON exp_S36/exp_S39/exp_S40 (pas de hardcode) |
| `test_notebook_structure` | `synthesis.ipynb` produit les 5 figures attendues |
| `test_fr_en_key_values` | valeurs clés (parité, F1, latences, ratios RAM) **identiques** FR≡EN |
| `test_board_v2_na_honest` | cellules board v2 absentes → `"à mesurer"` (pas de chiffre inventé) |

### Build & non-régression
```bash
pytest tests/test_sprint40_article.py -q
pytest tests/test_int8_c_emulation.py -q                 # émulateur (référence PC)
cd firmware/stm32f4_blink && make test                    # Unity host : v2 OK, v1 inchangé, 0 régression
cd docs/article/ewc_int8_mcu && make all                  # main_fr.pdf + main_en.pdf
jupyter nbconvert --execute notebooks/cl_eval/article_ewc_int8/synthesis.ipynb --to notebook
```
> Les 2 tests TinyOL préexistants restent hors périmètre.

### Mises à jour documentaires
- `docs/roadmap_phase2.md` : entrée **Sprint 40** (format Sprint 37–39) → ✅ à la clôture.
- `docs/triple_gap.md` : § **Gap 3** — si board v2 confirme la récupération, reformuler « partiel » →
  **RAM ÷4 sans perte de métrique** (kernel calibré, MCU réel) ; sinon rester sur l'axe émulateur + honnête.
- `CLAUDE.md` : ligne de statut Sprint 40.
- Invoquer le skill **`graphify_sprint_update`** (nouveaux `.c/.py/.tex/.ipynb/.md` → update graphe).

## Vérification

Tous les blocs `pytest` / `make test` / `make all` / nbconvert passent ; roadmap rend l'entrée Sprint 40 ;
`graphify_sprint_update` exécuté.
