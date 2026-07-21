# S4707 — Tests + docs (clôture Sprint 47)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 47 |
| **Priorité** | 🟡 Normale — verrouille la structure et l'honnêteté ; met à jour roadmap et triple_gap. |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Durée estimée** | 2h |
| **Dépendances** | S4702–S4706 |
| **Fichiers cibles** | `tests/test_s47_quant_depth.py`, `docs/roadmap_phase2.md`, `docs/triple_gap.md` |
| **Références** | patrons `tests/test_s46_quant_moment.py`, `tests/test_figures_library.py` (garde AST) |

---

## Contexte

Clôture PC du sprint : tests de structure/honnêteté, mise à jour de la roadmap et du triple_gap (§ Gap 3),
puis évaluation d'un update du graphe de connaissance.

## Spec

### 1. `tests/test_s47_quant_depth.py`

| Test | Vérifie |
|------|---------|
| `test_json_schema` | Chaque `exp_S47_depth/*.json` porte `weight_bits`, `granularity`, `symmetry`, `delta_auroc`, `ram_ratio_vs_fp32`, `config_snapshot`. |
| `test_bit_depth_monotonicity` | À granularité fixe, la RAM théorique **croît** quand `weight_bits` décroît (ratio ↑) ; sanity check de cohérence (pas d'assertion sur l'AUROC, dépend des données). |
| `test_quantconfig_no_regression` | `QuantConfig.legacy_c()` et presets S39 produisent les **mêmes logits** qu'avant l'extension (golden S39). |
| `test_subint8_presets` | `QuantConfig.subint8(bits, gran, sym, mode)` renseigne correctement `weight_bits`/`weight_mode`/`symmetry` ; ternaire ∈ {−1,0,1}, binaire ∈ {−1,1}. |
| `test_na_honesty` | `exp_S47_context/context.json` : HDC/Maha/TinyOL en `na_*` avec justification ; aucun champ métrique fabriqué. |
| `test_no_hardcoded_numbers` | Garde AST sur `src/figures/catalogs/quant_depth.py` : aucun littéral numérique de résultat (réutilise le patron `test_figures_library.py`). |

### 2. Documentation

- `docs/roadmap_phase2.md` : Sprint 47 → statut, bilan (résumé du sweep, cliff, per-channel, reco bits).
- `docs/triple_gap.md` (§ Gap 3) : ajouter le résultat profondeur/schéma (jusqu'où descendre à AUROC préservée,
  gain RAM théorique, réserve bit-packing → Sprint 48). **Aucun chiffre avant exécution.**
- CLAUDE.md : ligne de statut Sprint 47 (après exécution).
- Invoquer `graphify_sprint_update` (évalue la pertinence d'un update du graphe).

## Contraintes

- Tests **déterministes** (seed 42) ; passent en `pending`/skip honnête tant que les JSON ne sont pas produits,
  puis en dur une fois le sweep exécuté.
- 0 régression firmware (ce sprint est PC-only ; `make test` inchangé).

## Vérification

```bash
pytest tests/test_s47_quant_depth.py -q
pytest tests/ -k "quant_depth or figures_library" -q     # 0 régression garde AST
grep -c "Sprint 47" docs/roadmap_phase2.md
```

---

## Résolution (implémentée)

**Tests** — `tests/test_s47_quant_depth.py` **21 PASS** (le fichier préexistait avec 19 tests issus de
S4702/S4703 ; **+2 tests de la spec ajoutés**) :

- `test_na_honesty` : lit `experiments/exp_S47_context/context.json` ; vérifie `swept_models == ["ewc"]`
  et que HDC/Maha/TinyOL portent un `status` `na_*` avec `reason`/`ref` non vides et **aucun champ métrique
  fabriqué** (rejet de toute clé `auroc*`/`delta*`/`ram*` dans une cellule N/A).
- `test_no_hardcoded_numbers` : garde AST auto-contenue sur `src/figures/catalogs/quant_depth.py`
  (scan `ast.Constant` float hors liste blanche de layout, miroir de `test_figures_library.py`). Le catalogue
  est **aussi** déjà gardé dans `test_figures_library.py::HARDCODE_GUARDED_SRCS`.
- Couverture spec préexistante confirmée : `test_json_schema_and_honesty` (schéma JSON),
  `test_ram_ratio_grows_when_bits_shrink` + `test_agreement_non_increasing_with_bits` (monotonie),
  `test_no_regression_presets_golden` (0-régression golden S39), `test_subint8_fields` + grilles
  linéaire/ternaire/binaire (presets `subint8`).

`pytest tests/ -k "quant_depth or figures_library"` → **30 PASS, 0 régression** (garde AST intacte).

**Documentation** :

- `docs/roadmap_phase2.md` : bloc Sprint 47 → statut **✅ implémenté (S4701–S4708)** + bilan sweep + reco board.
- `docs/triple_gap.md` (§ Gap 3) : paragraphe « Renforcement Sprint 47 — profondeur & schéma » (jusqu'où
  descendre à AUROC préservée par dataset, per-channel repousse le cliff, affine n'aide pas, gain RAM
  **théorique** avec réserve explicite `.bss`/latence → Sprint 48).
- `CLAUDE.md` : ligne de statut Sprint 47.
- `graphify_sprint_update` invoqué (évaluation d'un update du graphe).

**Sprint PC-only** : firmware inchangé, `make test` non impacté (134, 2 TinyOL préexistants hors périmètre).
