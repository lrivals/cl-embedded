# S4907 — Tests + docs + clôture Sprint 49

| Champ | Valeur |
|-------|--------|
| **Sprint** | 49 |
| **Priorité** | 🟡 Moyen — verrouille la méthode et clôt le sprint. |
| **Statut** | ✅ Implémenté — `tests/test_ram_full.py` 6/6 PASS + MAJ roadmap/triple_gap + graphify_sprint_update. |
| **Durée estimée** | 2h |
| **Dépendances** | S4901–S4906 |
| **Fichiers cibles** | `tests/test_ram_full.py` (nouveau) · `docs/roadmap_phase2.md` · `docs/triple_gap.md` |
| **Références** | CR §2 · CLAUDE.md § « Fin d'une implémentation » |

## Contexte

Verrouiller la cohérence de la mesure RAM complète par des tests, mettre à jour la roadmap et le triple gap
(Gap 2 RAM), invoquer `graphify_sprint_update`.

## Spec

### 1. `tests/test_ram_full.py`

| Test | Vérifie |
|------|---------|
| `test_total_equals_components` | `total_ram_bytes == data + bss + max(pic_inference, pic_update)` |
| `test_stack_peak_positive` | pic ≥ 0 ; pic_update ≥ pic_inference (la MAJ CL creuse plus) sur cellules applicables |
| `test_na_honesty` | cellules `status:"na"` sans champ métrique fabriqué |
| `test_summary_indexed` | `summary.json` indexé `[model][dataset][condition][encoding][platform]` |
| `test_no_hardcoded_numbers` | garde AST 0-chiffre-en-dur sur `src/figures/catalogs/ram_full.py` |
| `test_terminology` | aucune occurrence de « modulable » hors CR |

Les tests tournent sur des **fixtures/JSON de structure** (pas besoin de carte) ; ils valident le schéma et les
invariants, pas des valeurs de résultat.

### 2. Docs

- `docs/roadmap_phase2.md` : bloc Sprint 49 → statut mis à jour.
- `docs/triple_gap.md` : § Gap 2 — préciser que la RAM rapportée est désormais `.data + .bss + pic de pile`
  (correction de la sous-estimation `.bss` seul).
- CLAUDE.md : ligne de statut sprint (si convention suivie).

### 3. Clôture

Invoquer `graphify_sprint_update` (évalue la pertinence d'un update du graphe).

## Format de sortie

`tests/test_ram_full.py` + éditions roadmap/triple_gap.

## Contraintes

- Tests indépendants de la carte (structure/invariants).
- Ne pas fabriquer de valeurs dans les fixtures au-delà du strict nécessaire à l'invariant.

## Vérification

```bash
python -m pytest tests/test_ram_full.py -q
grep -n "Sprint 49" docs/roadmap_phase2.md
grep -i "pic de pile\|\.data + \.bss" docs/triple_gap.md
```
