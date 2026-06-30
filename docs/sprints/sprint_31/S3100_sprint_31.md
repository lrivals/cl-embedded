# Sprint 31 — Méta-modèle de stacking (PC + board)

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 31 |
| **Semaine** | 7–13 juillet 2026 |
| **Statut** | ✅ Implémenté (PC + board réelle NUCLEO-F439ZI) |
| **Priorité globale** | 🔴 Critique — entraîner un méta-modèle léger arbitrant les 2 sorties d'une paire, le valider PC puis le porter et l'exécuter sur la carte (triple-modèle) |
| **Durée estimée totale** | ~24h |
| **Dépendances** | Sprint 30 ✅ (`ModelPair`, désaccord, benchmark fixe 3 paires × 5 datasets) · Sprint 27 ✅ (co-exécution board) · `scripts/export_weights_c.py` ✅ |

---

## Contexte et motivation

Le Sprint 30 a établi le benchmark fixe « paire = Mahalanobis + supervisé » avec des **règles de fusion statiques** (OR, AND, soft-vote, weighted). Ces règles sont aveugles au contexte : elles ne savent pas *quand* faire confiance à quel modèle. Le Sprint 31 remplace la fusion statique par un **méta-modèle de stacking appris** : un petit modèle qui prend les **sorties des 2 modèles de base** (+ features de désaccord/confiance) et produit la **prédiction finale arbitrée**.

Décisions validées :
- **Type** : **stacking léger** (régression logistique ou petit MLP 1 couche), vecteur d'entrée compact → **portable MCU**.
- **Cible** : **PC + board obligatoire** — implémentation C (`meta_head.c`) + exécution **triple-modèle** (Mahalanobis + supervisé + méta) validée sur NUCLEO-F439ZI.

```
Sprint 30 ✅ paires + désaccord        Sprint 31
ModelPair, disagreement_metrics  ──▶  S3101 src/ensemble/meta_learner.py
                                      S3103 scripts/train_meta_learner.py
                                      S3104 exp_S31_PC_*
                                                ↓
                                      S3105 firmware meta_head.c/.h (poids exportés)
                                      S3106 pipeline.c triple-mode + sensor_stream.py
                                      S3107 exp_S31_board_* (latence triple, RAM)
                                                ↓
                                      S3112 tests + docs
```

---

## Critères de succès

1. `src/ensemble/meta_learner.py` implémenté, entraîné **sans fuite** (prédictions out-of-fold des modèles de base).
2. Méta-modèle PC **bat ou égale** le meilleur modèle individuel **et** les règles d'ensemble fixes du Sprint 30 sur le benchmark binarisé.
3. `firmware/.../meta_head.c/.h` compile, poids exportés via `scripts/export_weights_c.py` (jamais à la main).
4. Exécution **triple-modèle** (Maha + supervisé + méta) validée board : latence < 100 ms (Gap 2), RAM `.bss` mesurée, parité board↔PC.
5. `make test` 0 nouvelles régressions (incl. `test_meta_head.c`).
6. Aucun chiffre inventé — champs « à mesurer » tant que non exécuté.

---

## Tâches

### O1 — Méta-learner PC

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3101 | Implémenter `src/ensemble/meta_learner.py` : `MetaLearner` stacking léger (régression logistique OU petit MLP 1 couche cachée, choix par config). Entrée = vecteur compact `[score_maha, prob_sup, disagreement, conf_sup, ...]`. Méthodes `fit(meta_X, y)`, `predict(meta_X)`, `predict_proba`. Entraîné sur prédictions **out-of-fold** des modèles de base (split méta dédié, anti-fuite). Annoter `# MEM:` (cible portable). | 🔴 | ✅ | `src/ensemble/meta_learner.py` | 4h |
| S3102 | `configs/meta_*.yaml` : type (`logreg`/`mlp`), liste des features d'entrée, dims couche cachée. **Aucun hyperparamètre en dur.** | 🟡 | ✅ | `configs/meta_stacking.yaml` | 1h |

### O2 — Entraînement & évaluation PC

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3103 | `scripts/train_meta_learner.py` : pour chaque paire×dataset, collecte les sorties des modèles de base (réutilise `ModelPair` + `run_cl_scenario_full`), construit `meta_X`, entraîne le méta-modèle, évalue vs (a) modèles individuels, (b) règles d'ensemble Sprint 30. Sort `results.json` (sections `meta`, `baselines`, `delta_vs_best_individual`, `delta_vs_ensemble`). | 🔴 | ✅ | `scripts/train_meta_learner.py` | 4h |
| S3104 | Expériences PC `experiments/exp_S31_PC_{pair}_{dataset}/` + `config_snapshot.yaml`. | 🔴 | ✅ | `experiments/exp_S31_PC_*/` | 3h |

### O3 — Portage board (triple-modèle)

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3105 | Implémenter `firmware/stm32f4_blink/src/meta_head.c` + `inc/meta_head.h` : méta-modèle en C (forward logreg/MLP), poids exportés via `scripts/export_weights_c.py` (jamais à la main). Toute taille via `#define`. Allocation statique. | 🔴 | ✅ | `firmware/stm32f4_blink/src/meta_head.c`, `firmware/stm32f4_blink/inc/meta_head.h` | 3h |
| S3106 | Étendre `pipeline.c` en mode triple (Maha + supervisé + méta) : nouveau FLAG (vérifier collisions de bits), réponse étendue (sorties des 2 bases + verdict méta). **MAJ `sensor_stream.py` en parallèle.** Budget `.bss` documenté. | 🔴 | ✅ | `firmware/stm32f4_blink/src/pipeline.c`, `firmware/stm32f4_blink/inc/pipeline.h`, `scripts/sensor_stream.py` | 3h |
| S3107 | Expériences board `experiments/exp_S31_board_*/` via recorder : latence **triple-modèle** (Gap 2 < 100 ms), RAM `.bss`, parité board↔PC. **RAM profiling obligatoire.** | 🔴 | ✅ | `experiments/exp_S31_board_*/` | 3h |

### O4 — Tests + docs

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3112 | Tests Python (`tests/test_meta_learner.py` : pas de fuite, méta ≥ baseline sur cas synthétique) + Unity (`tests/test_meta_head.c` : parité forward C↔Python). MAJ roadmap/CLAUDE.md. Invoquer `graphify_sprint_update`. | 🟢 | ✅ | `tests/test_meta_learner.py`, `firmware/stm32f4_blink/tests/test_meta_head.c` | 3h |

---

## Ordre d'exécution recommandé

```
S3101 (meta_learner.py)
        ↓
S3102 (configs)  →  S3103 (train_meta_learner.py)
        ↓
S3104 (exp PC)   ← valider gain vs baselines Sprint 30
        ↓
S3105 (meta_head.c, poids exportés)
        ↓
S3106 (pipeline.c triple + sensor_stream.py)  →  make test / make flash
        ↓
S3107 (exp board triple-modèle, latence/RAM)
        ↓
S3112 (tests + docs)
```

---

## Nomenclature des expériences

| Exp ID | Paire | Dataset | Mesure |
|--------|-------|---------|--------|
| exp_S31_PC_maha_ewc_monitoring | Maha + EWC + méta | Monitoring | F1/AUROC méta vs indiv vs ensemble |
| exp_S31_PC_maha_hdc_cwru | Maha + HDC + méta | CWRU | idem |
| … (paires × datasets) | | 5 datasets | |
| exp_S31_board_maha_ewc_* | triple-modèle | (≥1) | latence triple µs, .bss B, parité |

---

## Budget mémoire firmware estimé (triple-modèle)

| Composant | RAM .bss estimé | Notes |
|-----------|:---------------:|-------|
| Mahalanobis `g_detector` | 128 B | inchangé |
| Modèle supervisé | déjà alloué | global existant |
| `meta_head` `g_meta` (logreg 4 features) | +20 B | mesuré (struct `MetaHead`) |
| **Total .bss triple** | **104 596 B** | mesuré board (39.9 % de 256 Ko) ✅ |

---

## Notes d'implémentation

**S3101 anti-fuite** : le méta-modèle ne doit jamais voir les prédictions des modèles de base sur leurs propres données d'entraînement → collecter `meta_X` sur un split out-of-fold dédié (sinon le méta sur-apprend la sur-confiance des bases).

**S3105 `meta_head.c`** : suivre le pattern d'export `scripts/export_weights_c.py` → `model_weights.h` (ne jamais éditer à la main, règle CLAUDE.md). Forward = produit scalaire + sigmoïde (logreg) ou 1 couche cachée ReLU + sigmoïde (MLP).

**S3106 FLAGS** : byte saturé (TODO dorra). Le mode triple ajoute le verdict méta à la réponse DUAL_MODE généralisée du Sprint 30 (S3009) — réutiliser ce socle plutôt qu'un nouveau chemin.

---

## Questions ouvertes

- `TODO(arnaud)` : régression logistique suffisante, ou petit MLP justifié par un gain mesuré sur le benchmark ?
- `TODO(dorra)` : quantifier le méta-modèle en INT8 pour cohérence avec Gap 3, ou FP32 suffit vu sa petite taille ?
- `FIXME(gap2)` : vérifier que le triple-modèle (Maha + supervisé + méta) reste < 100 ms et tient en RAM board.

---

## Livrables

1. `src/ensemble/meta_learner.py` + `configs/meta_stacking.yaml`
2. `scripts/train_meta_learner.py` + répertoires `experiments/exp_S31_PC_*/`
3. `firmware/stm32f4_blink/src/meta_head.c` + `inc/meta_head.h`
4. `pipeline.c` + `sensor_stream.py` étendus (mode triple) + `experiments/exp_S31_board_*/`
5. `tests/test_meta_learner.py`, `firmware/stm32f4_blink/tests/test_meta_head.c`
6. MAJ `docs/roadmap_phase2.md` + `CLAUDE.md`

---

## Bilan

| Tâche | Statut | Notes |
|-------|:------:|-------|
| S3101 meta_learner.py | ✅ | `src/ensemble/meta_learner.py`, logreg/MLP, 12/12 tests PASS |
| S3102 configs méta | ✅ | `configs/meta_stacking.yaml` |
| S3103 train_meta_learner.py | ✅ | `scripts/train_meta_learner.py` |
| S3104 exp PC | ✅ | `experiments/exp_S31_PC_*` (14 + 1 skip honnête), méta ≥ ensemble 12/14 |
| S3105 meta_head.c/.h | ✅ | `meta_head.c/.h` + `export_meta_to_c()`, parité C↔Python 4/4 PASS |
| S3106 pipeline.c triple + sensor_stream | ✅ | TRIPLE_MODE `0xD0/0xE0`, réponse 27 B, `sensor_stream.py` synchronisé |
| S3107 exp board triple-modèle | ✅ | Board réelle : maha-ewc **258 µs**, maha-hdc **593 µs** (Gap 2 ✅), **parité 1.000**, `.bss=104 596 B` |
| S3112 tests + docs | ✅ | `test_meta_head.c` 4/4, `test_meta_learner.py` 12/12, docs + graphify |
