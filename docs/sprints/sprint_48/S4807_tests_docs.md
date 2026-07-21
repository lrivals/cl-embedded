# S4807 — Tests + docs (clôture Sprint 48)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 48 |
| **Priorité** | 🟡 Normale — verrouille Gap 2/Gap 3 board, parité, honnêteté ; met à jour roadmap/triple_gap. |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Durée estimée** | 1h |
| **Dépendances** | S4802–S4806 |
| **Fichiers cibles** | `tests/test_sprint48_board.py`, `docs/roadmap_phase2.md`, `docs/triple_gap.md` |
| **Références** | patrons `tests/test_sprint45_board.py`, `test_sprint36_comparison.py` |

---

## Contexte

Clôture board : tests de structure/honnêteté/gaps, mise à jour roadmap et triple_gap (§ Gap 2 latence dépacking,
§ Gap 3 RAM `.bss` packée), puis évaluation d'un update du graphe de connaissance.

## Spec

### `tests/test_sprint48_board.py`

| Test | Vérifie |
|------|---------|
| `test_summary_structure` | `exp_S48_summary.json` indexé `[dataset][weight_bits][granularity][platform]`, champs board + pc + deltas. |
| `test_gap2_latency` | Toute latence DWT board renseignée < 100 000 µs (Gap 2), y compris chemin packé (dépacking). |
| `test_gap3_ram_packed` | `.bss` **packé < `.bss` non-packé** (le packing matérialise le gain) quand la cellule est mesurée. |
| `test_parity_exact` | `parity_pred == 1.000` sur les cellules mesurées (schéma émulateur == kernel). |
| `test_na_honesty` | Cellules non mesurables → `na_reason` renseigné, jamais un chiffre fabriqué (débordement SRAM, mono-classe). |
| `test_no_hardcoded_numbers` | Garde AST sur le catalogue `quant_depth_board` (0 littéral de résultat). |

Tests **skip honnête** tant que la board n'a pas streamé (aucun JSON) → passent en dur une fois les cellules produites.

### Documentation

- `docs/roadmap_phase2.md` : Sprint 48 → statut + bilan (RAM `.bss` packée mesurée, latence dépacking, parité, écart théorie↔matériel).
- `docs/triple_gap.md` : § Gap 2 (latence dépacking sub-INT8), § Gap 3 (RAM `.bss` réelle ÷8/÷16 sous packing — clôt le `TODO(dorra)` S47).
- CLAUDE.md : ligne de statut Sprint 48 (après exécution).
- Unity firmware : `make test` inchangé sur le build défaut + builds sub-INT8 verts (0 régression, 2 TinyOL préexistants hors périmètre).
- Invoquer `graphify_sprint_update`.

## Contraintes

- Tests déterministes ; skip honnête sans board ; 0 régression firmware sur le build défaut.
- Aucun chiffre avant exécution.

## Vérification

```bash
pytest tests/test_sprint48_board.py -q
cd firmware/stm32f4_blink && make test          # 0 régression build défaut
grep -c "Sprint 48" docs/roadmap_phase2.md docs/triple_gap.md
```

---

## Résolution (implémentée)

_À compléter lors de l'implémentation._
