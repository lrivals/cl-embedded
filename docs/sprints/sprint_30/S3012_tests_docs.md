# S3012 — Tests + notebook origines désaccord + docs

| Champ | Valeur |
|-------|--------|
| **Sprint** | 30 |
| **Priorité** | 🟢 Nice-to-have (mais jamais absent — règle sprint_generation) |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 3h |
| **Dépendances** | S3001, S3003, S3006 |
| **Fichiers cibles** | `tests/test_model_pair.py`, `tests/test_disagreement.py`, `notebooks/sprint30_pairs_disagreement.ipynb` |
| **Références** | `tests/` (pytest existant), `notebooks/` |

---

## Contexte

Tâche de clôture : tests unitaires des briques net-new, notebook d'analyse des origines de désaccord (livrable scientifique), et mise à jour de la documentation.

## Sous-tâches

- **Tests Python** :
  - `tests/test_model_pair.py` : `predict_individual` aligne les sorties ; chaque règle de fusion (`or`/`and`/`soft_vote`/`weighted`) cohérente ; mode `binary` binarise correctement.
  - `tests/test_disagreement.py` : `disagreement_rate`/`cohen_kappa` sur cas connus ; `disagreement_confusion` somme = nb désaccords.
- **Tests Unity board** : si S3009 fait, vérifier réponse paire bien formée (sinon reporter).
- **Notebook** `notebooks/sprint30_pairs_disagreement.ipynb` : charge les 15 `exp_S30_PC_*`, tableau comparatif individuel vs ensemble, heatmap désaccord par paire×dataset, analyse des origines (`analyze_disagreement_origin`).
- **Docs** : MAJ `S3000_sprint_30.md` (bilan), `docs/roadmap_phase2.md`, ligne « Statut sprint » `CLAUDE.md`.
- Invoquer le skill `graphify_sprint_update`.

---

## Vérification

```bash
pytest tests/test_model_pair.py tests/test_disagreement.py -v
cd firmware/stm32f4_blink && make test
jupyter nbconvert --execute --to notebook --inplace notebooks/sprint30_pairs_disagreement.ipynb
```

---

## Bilan d'implémentation

- **Tests Python** : `test_model_pair.py` + `test_disagreement.py` = **19 PASS** (déjà présents/S3001+S3003).
- **Tests Unity board** (S3009 fait) : ajout de T80–T82 dans `test_pipeline.c` + `test_runner.c` —
  taille réponse paire 22 B, champs aux bons offsets, dispatch `PAIR_MODE` (0x90/0xA0). **3/3 PASS**
  (`make test` = 92 tests, 2 échecs TinyOL pré-existants hors périmètre).
- **Notebook** `notebooks/sprint30_pairs_disagreement.ipynb` : charge les 14 `exp_S30_PC_*` valides
  (+ N/A `maha_hdc_paderborn`), tableau indiv vs ensemble, heatmaps désaccord (taux + κ), « qui a
  raison », **origines** (score Maha désaccords vs accords, top-features), et **validation board**
  (latences séparées/combinée + `.bss`). Exécuté sans erreur via nbconvert.
- **Docs** : S3000 bilan, S3009/S3012 statut ✅, `roadmap_phase2.md`, ligne « Statut sprint » `CLAUDE.md`.
- **graphify_sprint_update** invoqué en clôture.
