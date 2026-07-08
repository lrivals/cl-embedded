# S4207 — Tests, documentation et clôture du sprint

| Champ | Valeur |
|-------|--------|
| **Sprint** | 42 |
| **Priorité** | 🟡 Moyenne (mais bloquante pour clore) |
| **Durée estimée** | ~2h |
| **Statut** | ✅ Implémenté (7 juillet 2026) |
| **Dépendances** | S4201–S4206 |
| **Fichiers cibles** | `tests/test_figures_library.py` · `docs/roadmap_phase2.md` · `CLAUDE.md` |

## Tests Python (`tests/test_figures_library.py`)

| Test | Vérifie |
|------|---------|
| `test_registry_lists_catalogs` | les 3 catalogues quantification enregistrés ; un catalogue jouet ajouté dynamiquement apparaît |
| `test_no_hardcoded_results` | scan AST de `src/figures/catalogs/quant_impact.py` : aucun littéral flottant de résultat (liste blanche : constantes de mise en page/dpi) |
| `test_loaders_na_honest` | `metric_or_na` retourne le sentinel sur champ absent/`null`+`na_reason`, jamais 0 |
| `test_generate_idempotent` | deux exécutions d'un catalogue → mêmes fichiers produits (seed fixé) |
| `test_missing_experiment_raises` | expérience source absente → erreur claire, pas de figure partielle silencieuse |
| `test_a_mesurer_placeholder` | avec `exp_S40_board_v2/` absent, I6 rend le placeholder « à mesurer » (fixture tmp) |
| `test_figures_output_paths` | `savefig_png` écrit sous `docs/figures/<catalog>/` et retourne les chemins |

Cible : **7 tests PASS**, suite `pytest tests/ -v` sans régression (652+ tests collectés, 0 erreur).

## Documentation & clôture (convention fin de sprint)

1. Statuts S4200–S4207 mis à jour (📝 → ✅) + tableau Bilan de `S4200_sprint_42.md` (temps réels, notes).
2. `docs/roadmap_phase2.md` — entrée Sprint 42 avec livrables réels.
3. `CLAUDE.md` — ligne de statut sprint (résumé : infra `src/figures/` + 17 figures quantification +
   inventaire `docs/context/quantization_strategies.md`).
4. Invoquer **`graphify_sprint_update`** (nouveau module `src/figures/`, nouveau doc de contexte,
   dépendances loaders→experiments : impact graphe probable).
5. Proposer un message de commit si tout est vert.

## Critères d'acceptation

1. 7/7 tests PASS, 0 régression sur la suite existante.
2. `python scripts/generate_figures.py --all` régénère l'ensemble sans erreur sur un clone propre
   (expériences présentes) — commande documentée dans le README du catalogue.
3. Roadmap + CLAUDE.md à jour, graphify évalué.

## Réalisation (7 juillet 2026)

- `tests/test_figures_library.py` — **7/7 PASS** : `test_registry_lists_catalogs` (3 catalogues + jouet dynamique nettoyé), `test_no_hardcoded_results` (scan AST de `quant_impact.py`, liste blanche de 23 flottants de layout ; tout autre littéral échoue), `test_loaders_na_honest` (None/sentinel jamais 0, 0 réel préservé), `test_generate_idempotent` (2 runs du catalogue pipeline → PNG byte-identiques, seed fixé), `test_missing_experiment_raises` (FileNotFoundError), `test_a_mesurer_placeholder` (monkeypatch `EXPERIMENTS_DIR` → « à mesurer »), `test_figures_output_paths` (`savefig_png` sous `<out>/<catalog>/`).
- Suite : **714 tests collectés, 0 erreur de collection**. Régression figures nulle ; l'échec `test_board_recorder.py::TestDryRunOutput::test_n_params_ewc` est **préexistant et hors périmètre** (config dry-run board `ewc/monitoring` absente, aucun lien avec `src/figures/` — `scripts/board_experiment_recorder.py` non modifié).
- `generate_figures.py --all` : **17 PNG** régénérés sans warning (pédagogie 6 + pipeline 5 + impact 6).
- Statuts S4204–S4207 ✅, bilan `S4200`, `roadmap_phase2.md` et `CLAUDE.md` mis à jour ; `graphify_sprint_update` invoqué.
