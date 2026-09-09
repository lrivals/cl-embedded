# S5007 — Tests + docs + clôture Sprint 50

| Champ | Valeur |
|-------|--------|
| **Sprint** | 50 |
| **Priorité** | 🟡 Moyen — verrouille l'honnêteté « à mesurer » et clôt le sprint. |
| **Statut** | ✅ Implémenté — `tests/test_s50_energy.py` 13/13 PASS (+5 tests S5007 : µJ>0 seulement si CSV, `by_component.sensor="na"`, segments dequant/mac/requant, garde AST `energy_real.py`, `energy_calibration` présent) ; suite énergie+figures 49 PASS 0 régression ; firmware `make test` 141 (2 TinyOL préexistants, `.bss` défaut invariant). Roadmap + triple_gap MAJ (§ latence INT8 paradoxe FPU mesuré). |
| **Durée estimée** | 2h |
| **Dépendances** | S5001–S5006 · `tests/test_energy_capture.py`/`test_autonomy.py` ✅ (S33) |
| **Fichiers cibles** | `tests/test_s50_energy.py` · `docs/roadmap_phase2.md` · `docs/triple_gap.md` |
| **Références** | CLAUDE.md § « Fin d'une implémentation » |

## Contexte

Tests d'honnêteté (aucune énergie sans CSV réel), MAJ roadmap/triple_gap, `graphify_sprint_update`.

## Spec

### 1. `tests/test_s50_energy.py`

| Test | Vérifie |
|------|---------|
| `test_uj_positive_only_if_csv` | µJ > 0 **uniquement** si un CSV réel a été fourni ; sinon `"à mesurer"` |
| `test_component_na_honest` | `by_component` non isolable → `"na"` + `na_reason`, jamais un delta fabriqué |
| `test_int8_latency_segments` | breakdown contient `dequant`/`mac`/`requant` |
| `test_cost_benefit_no_hardcode` | garde AST 0-chiffre-en-dur sur `energy_real.py` |
| `test_calibration_present` | `energy_calibration` dans `hw_profile_f439zi.yaml` |

Tests sur fixtures (pas de carte requise pour la structure).

### 2. Docs

- `roadmap_phase2.md` : bloc Sprint 50 → statut.
- `triple_gap.md` : § énergie — énergie/inférence mesurée, paradoxe latence FPU confirmé (Gap 2/3).

### 3. Clôture

`graphify_sprint_update`.

## Vérification

```bash
python -m pytest tests/test_s50_energy.py tests/test_energy_capture.py tests/test_autonomy.py -q
grep -n "Sprint 50" docs/roadmap_phase2.md
grep -i "énergie\|µJ\|FPU" docs/triple_gap.md
```
