# S5107 — Tests + docs + clôture Sprint 51

| Champ | Valeur |
|-------|--------|
| **Sprint** | 51 |
| **Priorité** | 🟢 Bas — verrouille le score et clôt le sprint. |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Durée estimée** | 2h |
| **Dépendances** | S5101–S5106 |
| **Fichiers cibles** | `tests/test_system_score.py` · `docs/roadmap_phase2.md` · `docs/triple_gap.md` |
| **Références** | CLAUDE.md § « Fin d'une implémentation » |

## Contexte

Tester les invariants du score, mettre à jour roadmap/triple_gap, invoquer `graphify_sprint_update`.

## Spec

### 1. `tests/test_system_score.py`

| Test | Vérifie |
|------|---------|
| `test_score_bounded` | score ∈ [0,1] après normalisation |
| `test_dimension_orientation` | dimension coût inversée (RAM↓ améliore le score) |
| `test_pending_on_missing_dim` | dimension manquante → `pending`, jamais 0 |
| `test_sensitivity_ranks` | 2 jeux de poids → 2 classements produits |
| `test_platform_separation` | board et PC classés séparément |
| `test_no_hardcoded_numbers` | garde AST 0-chiffre-en-dur sur `system_score.py` (catalogue) |

### 2. Docs

- `roadmap_phase2.md` : bloc Sprint 51 → statut.
- `triple_gap.md` : mention du score composite (synthèse transverse Gap 1/2/3 : perf ↔ RAM ↔ latence ↔ énergie).

### 3. Clôture

`graphify_sprint_update`.

## Vérification

```bash
python -m pytest tests/test_system_score.py -q
grep -n "Sprint 51" docs/roadmap_phase2.md
```
