# S3809 — Tests, documentation, roadmap, graphify

| Champ | Valeur |
|-------|--------|
| **Sprint** | 38 |
| **Priorité** | 🟠 Haute — garantit non-régression et trace la clôture du sprint. |
| **Statut** | ✅ Implémenté — `test_sprint38_autonomous.py` 10/10 PASS · Unity `test_drift_detector.c` 6/6 (0 régression) · docs/roadmap/triple_gap/CLAUDE.md à jour. |
| **Durée estimée** | 4h |
| **Dépendances** | S3802–S3808 · `tests/` ✅ · `firmware/.../tests/` ✅ · `skills/graphify_sprint_update.md` ✅ |
| **Fichiers cibles** | `tests/test_sprint38_autonomous.py`, `firmware/.../tests/test_drift_detector.c` (+`test_runner.c`), `docs/roadmap_phase2.md`, `docs/triple_gap.md`, `CLAUDE.md` |
| **Références** | S3608 (tests Sprint 36 comme modèle) · Sprint 26 (oubli — lien Gap) |

---

## Contexte

Verrouiller le comportement (logique des politiques, calibration, déterminisme du gate, Gap 2) et
documenter la clôture, sans régression sur l'EWC existant (build firmware par défaut inchangé).

## Spec

### Tests Python — `test_sprint38_autonomous.py`
- Calibration des seuils : `set_thresholds_from_normal` → `fault = P95 × 2.5`, `drift = P95 × 1.3`.
- Logique des 4 politiques : `frozen` (0 update), `always` (1 update/échantillon), `gated_*`
  (update ssi verdict ≠ NORMAL ; P3 : FAULT→pseudo 1, DRIFT→maha_update, NORMAL→rien).
- Déterminisme du gate (même séquence → mêmes verdicts).
- Structure `exp_S38_summary.json` + `economy_table` (deltas vs `always`).
- Gap 2 : toutes les latences stockées < 100 ms (si présentes).

### Test Unity firmware — `test_drift_detector.c`
Parité C↔Python sur une séquence de scores (verdicts identiques, priorité FAULT, déclenchement DRIFT,
reset). Déclaré dans `test_runner.c`. `make test` : nouveaux tests PASS, **0 régression** (les 2 TinyOL
préexistants restent hors périmètre).

### Documentation & clôture
- `docs/roadmap_phase2.md` : Sprint 38 → statut + bilan.
- `docs/triple_gap.md` : § Gap 2 (latence gate vs SGD permanent) + § Gap 3 (RAM du gate).
- `CLAUDE.md` : ligne de statut Sprint 38.
- Skill `graphify_sprint_update` : évaluer la pertinence d'un update du graphe.

## Vérification

```bash
pytest tests/test_sprint38_autonomous.py -v
cd firmware/stm32f4_blink && make test     # test_drift_detector PASS + 0 régression
```
