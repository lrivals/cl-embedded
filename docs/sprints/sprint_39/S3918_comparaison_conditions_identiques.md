# S3918–S3919 — Comparaison INT8 PC ↔ board à conditions strictement identiques

| Champ | Valeur |
|-------|--------|
| **Sprint** | 39 |
| **Priorité** | 🔴 Critique — rend la comparaison PC↔board *scientifiquement pertinente* (parité bit-exacte, pas « chiffres proches ») |
| **Statut** | ✅ S3918 (PC) + S3919 (board réelle) implémentés — 1er juillet 2026 |
| **Durée estimée** | S3918 ~4h (PC) · S3919 ~3h (board, différé) |
| **Dépendances** | S3902 ✅ (émulateur bit-exact) · S3903 (validation) · S3907/S3908 (kernel v2) · `train_board_reference.py` · `board_pc_parity.py` |
| **Fichiers cibles** | `scripts/run_s39_matched_compare.py`, `experiments/exp_S39_matched/`, `tests/test_s39_matched.py` |

---

## ✅ Réalisé (1er juillet 2026)

**S3918 (PC).** `scripts/run_s39_matched_compare.py` produit, par `(ewc, dataset, schéma)`, un
résultat PC exécutant le **chemin board exact** — l'émulateur `forward_quant(scheme)`, jamais le
QAT S28. Source de données unique (`load_condition_arrays`), métrique partagée
(`compute_fault_f1`), checkpoint FP32 **dumpé** (`exp_S39_matched/checkpoints/`) → réutilisé tel
quel par le board (parité par construction). Sortie `matched_ewc_{ds}_{scheme}.json` : table par
échantillon `[idx, y_true, pred_fp32, pred_int8_pc]` (+ alias `pred_pc`), F1, accord vs FP32.
Tests `tests/test_s39_matched.py` **5 PASS** (côté PC = schéma board pas QAT ; source unique ;
`legacy_c`/`per_channel` déterministes bit-exact ; schéma de sortie compatible parité).

Résultats émulateur (5feat), confirmant le diagnostic Gap 3 :

| Dataset | legacy_c (v1) | per_channel_int8 (v2) | q15 (v2) | mixed (v2) | FP32 |
|---------|:-------------:|:---------------------:|:--------:|:----------:|:----:|
| pronostia | 0.066 | 0.943 | 0.962 | 0.962 | 0.962 |
| cmapss | 0.227 | 0.448 | 0.448 | 0.448 | 0.448 |

**S3919 (board réelle).** `scripts/run_s39_board.py` streame le **même schéma, mêmes poids
(checkpoint apparié), sans `--update`** ; la parité gelée est calculée en ré-émulant le schéma
sur les features réellement streamées. **Résultat : parité gelée bit-exacte = 1.000 (0 mismatch)
sur les 5 cellules board** (legacy_c/per_channel/q15 × pronostia/cmapss). Détails, latences et
`.bss` : voir `S3915_board_validation.md`. Régime online **hors périmètre** (kernel v2 =
inférence seule).

> **Honnêteté** : la parité gelée bit-exacte confirme que board et émulateur calculent la même
> chose échantillon par échantillon ; l'écart F1 board↔émulateur observé vient du **sous-échantillon
> streamé** (300 éch.) ≠ split complet, pas d'une divergence de calcul.

---

## Motivation

Une comparaison INT8 PC↔board n'est **pertinente** que si les deux côtés exécutent **la même
quantification, sur les mêmes données, dans les mêmes conditions**, et produisent **les mêmes définitions de
métriques**. Deux dérives invalident sinon toute conclusion :

1. **Schémas non appariés.** L'INT8 QAT du Sprint 28 (per-canal calibré, déquant FP32) et l'INT8 PTQ du
   Sprint 36 board (scale `1/128` fixe, accumulateur int16) sont **deux calculs différents** — c'est la
   cause même de l'écart de F1. Les juxtaposer dans un tableau serait comparer des pommes et des oranges.
   ➜ **Règle** : le côté « PC » d'une comparaison board doit être l'**émulateur exécutant le schéma exact
   du board** (`QuantConfig.legacy_c()` ou v2), jamais le modèle QAT S28.

2. **Pipeline de données non partagé.** Ordre des échantillons, normalisation hôte, seed, quantification
   d'entrée : toute divergence casse la parité. ➜ **Règle** : source unique
   (`feature_conditions.load_condition_arrays`) + même quantif d'entrée (`float_to_q7`) des deux côtés.

## Deux régimes de parité (à distinguer explicitement)

| Régime | Parité PC↔board | Justification |
|--------|:---------------:|---------------|
| **Inférence gelée** (frozen, sans `--update`) | ✅ **bit-exacte** attendue | Chemin entier déterministe ; l'émulateur reproduit acc int16, overflow, `>>7` bit-à-bit (S3902). |
| **Online** (avec `--update`/SGD) | ⚠️ approchée seulement | La MAJ accumule des flottants : board float32 ≠ PC float64/entier. Divergence documentée S26/S36. |

Le sprint **ne prétend l'exactitude que sur le régime gelé** ; l'online est reporté comme accord ≈, avec la
cause nommée (pas de chiffre forcé).

---

## S3918 — Harnais de comparaison apparié (PC, réalisable sans board)

`scripts/run_s39_matched_compare.py` : produit, pour chaque `(modèle, dataset, schéma)`, un résultat PC
**directement comparable au board futur** parce qu'il exécute le chemin board exact.

Étapes :

1. **Poids partagés** : `train_board_reference.py` (même tête `EWCMlpMulticlass`, même seed) → `ewc_head.pt`,
   chargé dans `EWCHeadWeights.from_state_dict` **et** exporté en header C (`export_weights_c.py`) — un seul
   checkpoint alimente les deux côtés.
2. **Données partagées** : `load_condition_arrays(condition, dataset)` (source unique board/PC, cf. S3508)
   → mêmes colonnes, même ordre, même normalisation. Quantif d'entrée `float_to_q7` identique.
3. **Métrique partagée** : F1/AUROC calculés par le **même** code (`src/evaluation/`), pas de redéfinition.
4. **Côté PC** : `forward_quant(w, X, scheme)` pour `scheme ∈ {legacy_c, per_channel_int8, q15,
   mixed_int8w_q15act}` + `forward_fp32` (référence).
5. **Sortie** : `experiments/exp_S39_matched/matched_{model}_{dataset}_{scheme}.json` avec table par
   échantillon `[idx, y_true, pred_fp32, pred_int8_pc]`, F1, accord vs FP32, et **empreinte prête pour la
   parité board** (mêmes clés que `board_pc_parity.py`).

Ce fichier est le « côté PC » que S3919 confrontera bit-à-bit au board — sans re-flasher, S3918 fournit déjà
la référence légitime pour les tableaux du notebook (S3911).

## S3919 — Confrontation board (Partie B, différé — carte requise)

Quand la carte est disponible : `run_s39_board.py` (S3915) streame **le même schéma, les mêmes échantillons
dans le même ordre, sans `--update`** ; `board_pc_parity.py` confronte la sortie board à
`exp_S39_matched/` de S3918.

| Critère | Cible |
|---------|-------|
| Parité **frozen** legacy_c board ↔ émulateur PC | **bit-exacte** (accord = 1.000, 0 mismatch) |
| Parité **frozen** v2 (per-channel/q15) board ↔ émulateur PC | bit-exacte (accord = 1.000) |
| Régime **online** | accord ≈ documenté (float32 board ≠ PC), cause nommée, aucun chiffre forcé |
| F1 board par schéma | == F1 émulateur S3918 (à la parité près) |

> **Honnêteté** : si la parité frozen n'est pas exacte, c'est un **bug à corriger** (ordre, normalisation,
> quantif d'entrée), pas une tolérance à élargir — l'inférence gelée INT8 est déterministe par construction.

---

## Tests (`tests/test_s39_matched.py`)

```python
def test_pc_side_uses_board_scheme_not_qat():
    # le côté PC d'une comparaison board = legacy_c/v2, jamais EWCMlpInt8Classifier (QAT S28)
    ...

def test_matched_pipeline_single_source():
    # mêmes X/ordre/normalisation des deux côtés (load_condition_arrays partagé)
    ...

def test_frozen_is_deterministic():
    # deux exécutions émulateur legacy_c sur mêmes données → sorties identiques (bit-exact)
    ...
```

## Vérification

```bash
python scripts/run_s39_matched_compare.py --model ewc --dataset pronostia --condition 5feat
pytest tests/test_s39_matched.py -v
# plus tard, board branchée :
python scripts/run_s39_board.py --scheme legacy_c --dataset pronostia
python scripts/board_pc_parity.py --exp exp_S39_matched --against exp_S39_board
```
