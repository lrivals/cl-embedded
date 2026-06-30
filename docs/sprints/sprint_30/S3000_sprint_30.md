# Sprint 30 — Paires de modèles parallèles (benchmark fixe + analyse de désaccord)

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 30 |
| **Semaine** | 30 juin – 6 juillet 2026 |
| **Statut** | ✅ Implémenté (PC + board réelle) |
| **Priorité globale** | 🔴 Critique — établir le benchmark fixe « paire = Mahalanobis + modèle supervisé » sur les 5 datasets, mesurer perf individuelle + ensemble, analyser le désaccord, amorcer le portage board |
| **Durée estimée totale** | ~26h |
| **Dépendances** | Sprint 27 ✅ (DUAL_MODE `pipeline.c`, co-exécution 2 modèles board) · Sprint 28 ✅ (cadre binarisé normal-vs-fault, modèles 5 datasets) · `src/training/scenarios.py` ✅ (`run_cl_scenario_full`) · `src/models/base_cl_model.py` ✅ |

---

## Contexte et motivation

Les 5 modèles du projet (TinyOL, EWC, EWC INT8, HDC, détecteurs non-supervisés dont Mahalanobis) ont été portés et validés **individuellement** sur la NUCLEO-F439ZI. Le Sprint 27 a prouvé qu'on peut **co-exécuter 2 modèles** sur la carte (DUAL_MODE : 1 trame UART → 2 forward en séquence → réponse combinée, overhead ~0 µs). Le Sprint 28 a établi un **cadre commun de détection de panne binarisée** (normal-vs-fault) permettant de comparer un modèle supervisé à des baselines anomaly sur les mêmes labels.

Sprint 30 systématise les **paires de modèles** comme **benchmark fixe jusqu'à la fin du stage** : on associe le détecteur non-supervisé **Mahalanobis** (baseline anomaly, robuste, 128 B RAM) à chacun des 3 modèles supervisés (**HDC**, **EWC**, **TinyOL**), sur les **5 datasets** (Pronostia, Monitoring, CWRU, CMAPSS, Paderborn). Pour chaque paire×dataset on mesure :

1. les **métriques d'entraînement CL** (AA / AF / BWT) de chaque modèle **individuellement** ;
2. les **métriques d'inférence** (AUROC, F1, précision/rappel) individuelles **et** d'**ensemble** ;
3. le **désaccord** entre les 2 modèles : où divergent-ils, qui a raison, et **pourquoi** (origine dans l'espace des features / score Mahalanobis / proximité de frontière).

> **Pourquoi des paires non-supervisé + supervisé ?** Mahalanobis détecte l'anomalie sans labels (robuste au drift, déployable « à froid ») ; le modèle supervisé apprend les classes de panne. Leur combinaison est le socle scientifique du Sprint 31 (méta-modèle d'arbitrage) et un cas d'usage industriel réaliste (détecteur générique + classifieur spécialisé co-résidents sur MCU).

**Découpage validé** :
- **Partie A (prioritaire)** — cadre **binarisé normal-vs-fault** partout : sorties directement comparables, ensemble/désaccord/méta-modèle propres.
- **Partie B** — paires en **tâches natives** par dataset (Mahalanobis anomalie + modèle natif : RUL CMAPSS, multi-classe CWRU, binaire Monitoring), désaccord redéfini par dataset.

```
Sprint 27 ✅ DUAL_MODE          Sprint 30
Sprint 28 ✅ binarisé        ───────────────────────────────────────────
run_cl_scenario_full()  ──▶  S3001 src/ensemble/model_pair.py
                             S3003 src/evaluation/disagreement_metrics.py
                                       ↓
                             S3005 scripts/train_model_pair.py (PartieA binarisé)
                             S3006 exp_S30_PC_* (3 paires × 5 datasets = 15)
                                       ↓
                             S3007 ModelPair mode "native" (Partie B)
                                       ↓
                             S3009 pipeline.c paires arbitraires Maha+{HDC,EWC,TinyOL}
                             S3010 exp_S30_board_* (≥1 paire, latence/RAM)
                                       ↓
                             S3012 tests + notebook origines désaccord + docs
```

---

## Critères de succès

1. `src/ensemble/model_pair.py` + `src/evaluation/disagreement_metrics.py` implémentés, importables.
2. **Partie A** : 3 paires × 5 datasets entraînées + évaluées sur PC, `results.json` avec sections `model_a`, `model_b`, `ensemble`, `disagreement` distinctes.
3. Analyse de désaccord produite avec **piste d'origine** documentée (≥1 notebook).
4. **Partie B** amorcée : ≥1 paire en tâches natives par dataset.
5. ≥1 paire portée et validée sur board (latence séparées/combinées, `.bss` mesuré).
6. `pytest tests/test_model_pair.py tests/test_disagreement.py` verts.
7. Aucun chiffre de perf/RAM/latence inventé — tout sort d'une exécution (champs « à mesurer » sinon).

---

## Tâches

### O1 — Infrastructure paires (PC)

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3001 | Implémenter `src/ensemble/model_pair.py` : classe `ModelPair(detector, classifier, mode)` wrappant un `MahalanobisDetector` + un modèle supervisé (`BaseCLModel`). Méthodes : `predict_individual(x)` → `(pred_maha, pred_sup)`, `predict_ensemble(x, rule)` (règles `or`/`and`/`soft_vote`/`weighted`), `predict_proba(x)`. Mode `"binary"` (Partie A) binarise les sorties supervisées en normal-vs-fault (réutilise logique Sprint 28). Annoter `# MEM:`. | 🔴 | ⬜ | `src/ensemble/model_pair.py`, `src/ensemble/__init__.py` | 4h |
| S3002 | Créer `configs/board_pair_{maha_hdc,maha_ewc,maha_tinyol}.yaml` : une config par paire (modèles, règle de fusion, seuils, dataset). **Aucun hyperparamètre en dur dans le code** (règle CLAUDE.md). | 🟡 | ✅ | `configs/board_pair_maha_hdc.yaml`, `configs/board_pair_maha_ewc.yaml`, `configs/board_pair_maha_tinyol.yaml` | 2h |

### O2 — Métriques de désaccord

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3003 | Implémenter `src/evaluation/disagreement_metrics.py` : `disagreement_rate(y_a, y_b)`, `cohen_kappa(y_a, y_b)`, `disagreement_confusion(y_a, y_b, y_true)` (qui a raison quand ils divergent), `per_sample_disagreement_mask(y_a, y_b)`, `analyze_disagreement_origin(X, mask, y_true, maha_scores)` (corrélation désaccord ↔ features / score Mahalanobis / proximité frontière). | 🔴 | ⬜ | `src/evaluation/disagreement_metrics.py` | 3h |

### O3 — Entraînement & évaluation Partie A (binarisé) — PRIORITAIRE

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3005 | Implémenter `scripts/train_model_pair.py` : entraîne les 2 modèles d'une paire sur un dataset (boucle CL via `run_cl_scenario_full`), évalue **entraînement** (AA/AF/BWT) et **inférence** (AUROC, F1, précision/rappel) pour chaque modèle individuellement + l'ensemble. Sort un `results.json` avec sections `model_a` / `model_b` / `ensemble` / `disagreement`. | 🔴 | ✅ | `scripts/train_model_pair.py` | 4h |
| S3006 | Expériences PC Partie A : `experiments/exp_S30_PC_{pair}_{dataset}/` — 3 paires × 5 datasets = 15 runs binarisés + `config_snapshot.yaml` chacun. (Tâche « expérience + enregistrement » obligatoire.) | 🔴 | ✅ | `experiments/exp_S30_PC_*/` | 4h |

### O4 — Partie B : tâches natives (hétérogène)

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3007 | Étendre `ModelPair` au mode `"native"` : Mahalanobis (anomalie) + modèle natif (RUL CMAPSS / multi-classe CWRU / binaire Monitoring). Désaccord redéfini par dataset (ex. RUL : seuil de criticité vs anomalie). Expériences `experiments/exp_S30_PC_native_*`. | 🟡 | ✅ | `src/ensemble/model_pair.py`, `experiments/exp_S30_PC_native_*/` | 3h |

### O5 — Portage board (généralisation DUAL_MODE)

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3009 | Généraliser `pipeline.c` DUAL_MODE pour des paires arbitraires Maha+{HDC,EWC,TinyOL} sélectionnées par FLAGS ; réponse = sorties des 2 modèles. **Mettre à jour `sensor_stream.py` en parallèle** (règle CLAUDE.md : ne jamais désynchroniser protocole UART ↔ sensor_stream). | 🟡 | ✅ | `firmware/stm32f4_blink/src/pipeline.c`, `firmware/stm32f4_blink/inc/pipeline.h`, `scripts/sensor_stream.py` | 3h |
| S3010 | Expériences board (≥1 paire) : latences séparées/combinées, `.bss`, métriques en ligne. **RAM profiling obligatoire** (nouveau chemin d'exécution multi-modèle). | 🟡 | ✅ | `scripts/board_pair_recorder.py`, `experiments/exp_S30_board_*/` | 2h |

### O6 — Tests + analyse + docs

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3012 | Tests Python (`tests/test_model_pair.py`, `tests/test_disagreement.py`) + Unity board si S3009 fait ; notebook `notebooks/sprint30_pairs_disagreement.ipynb` (analyse des origines de désaccord) ; MAJ docs + skill `graphify_sprint_update`. | 🟢 | ⬜ | `tests/test_model_pair.py`, `tests/test_disagreement.py`, `notebooks/sprint30_pairs_disagreement.ipynb` | 3h |

---

## Ordre d'exécution recommandé

```
S3001 (model_pair.py)  +  S3003 (disagreement_metrics.py)   [parallèle]
        ↓
S3002 (configs paires)
        ↓
S3005 (train_model_pair.py)
        ↓
S3006 (exp Partie A, 15 runs binarisés)   ← PRIORITAIRE
        ↓
S3007 (Partie B native)
        ↓
S3009 (pipeline.c paires + sensor_stream.py)  →  make test / make flash
        ↓
S3010 (exp board ≥1 paire)
        ↓
S3012 (tests + notebook désaccord + docs)
```

---

## Nomenclature des expériences

| Exp ID | Paire | Dataset | Cadre | Métriques |
|--------|-------|---------|-------|-----------|
| exp_S30_PC_maha_hdc_pronostia | Maha + HDC | Pronostia | binarisé | AUROC/F1 indiv + ensemble + désaccord |
| exp_S30_PC_maha_ewc_monitoring | Maha + EWC | Monitoring | binarisé | idem |
| exp_S30_PC_maha_tinyol_cwru | Maha + TinyOL | CWRU | binarisé | idem |
| … (3 paires × 5 datasets = 15) | | Pronostia/Monitoring/CWRU/CMAPSS/Paderborn | binarisé | |
| exp_S30_PC_native_maha_ewc_cmapss | Maha + EWC | CMAPSS | natif (RUL) | RMSE + anomalie + désaccord |
| exp_S30_board_maha_ewc_* | Maha + EWC | (≥1) | board | latence sép./comb. µs, .bss B |

---

## Budget mémoire firmware estimé (paire board)

| Composant | RAM .bss estimé | Notes |
|-----------|:---------------:|-------|
| Mahalanobis `g_detector` | 128 B | mean[5] + precision[5×5] |
| Modèle supervisé (HDC/EWC/TinyOL) | déjà alloué | global .bss existant |
| Firmware existant (post-Sprint 29) | ~à mesurer | dépend du modèle co-résident |
| **Total .bss paire** | **à mesurer (S3010)** | << 256 Ko attendu |

---

## Notes d'implémentation

**S3001 `ModelPair`** : Mahalanobis n'implémente pas `BaseCLModel` à l'identique (c'est un détecteur non-supervisé) — adapter via une fine couche d'adaptation plutôt que forcer l'interface. La binarisation des sorties supervisées (mode `"binary"`) doit réutiliser exactement le mapping normal-vs-fault du Sprint 28 (cohérence benchmark).

**S3003 désaccord** : récupérer `(y_true, y_pred_a)` et `(y_true, y_pred_b)` via `run_cl_scenario_full()` exécuté pour chaque modèle, puis aligner par index d'échantillon. `analyze_disagreement_origin` corrèle le masque de désaccord aux features et au score Mahalanobis (un désaccord fréquent près de la frontière de décision est une explication attendue).

**S3009 pipeline.c** : le byte FLAGS est saturé (TODO dorra, S2600). Vérifier l'absence de collision de bits avant d'ajouter un sélecteur de paire ; sinon réutiliser le mécanisme DUAL_MODE existant (`0x70`) en paramétrant le 2ᵉ modèle.

---

## Questions ouvertes

- `TODO(arnaud)` : règle de fusion d'ensemble de référence pour le benchmark fixe — OR (priorité anomalie) vs soft-vote ?
- `TODO(arnaud)` : la binarisation RUL CMAPSS (seuil normal-vs-fault) doit-elle suivre le seuil du Sprint 28 ?
- `TODO(fred)` : le cas d'usage « détecteur générique + classifieur spécialisé co-résidents » correspond-il à un besoin Edge Spectrum concret ?

---

## Livrables

1. `src/ensemble/model_pair.py` (+ `__init__.py`) — `ModelPair` modes `binary` / `native`
2. `src/evaluation/disagreement_metrics.py`
3. `configs/board_pair_{maha_hdc,maha_ewc,maha_tinyol}.yaml`
4. `scripts/train_model_pair.py`
5. 15 répertoires `experiments/exp_S30_PC_*/` (Partie A) + ≥1 `exp_S30_PC_native_*/` (Partie B)
6. `pipeline.c` + `sensor_stream.py` étendus (paires arbitraires) + ≥1 `exp_S30_board_*/`
7. `tests/test_model_pair.py`, `tests/test_disagreement.py`
8. `notebooks/sprint30_pairs_disagreement.ipynb`

---

## Bilan (à compléter)

| Tâche | Statut | Temps réel | Notes |
|-------|:------:|:----------:|-------|
| S3001 model_pair.py | ✅ | — | `ModelPair(detector, classifier, mode)` modes binary/native ; 9 tests `test_model_pair.py` PASS |
| S3002 configs paires | ✅ | — | 3 configs référençant les configs `{model}_int8_{dataset}` validées S28 |
| S3003 disagreement_metrics.py | ✅ | — | rate/kappa/confusion/mask/origin ; 10 tests `test_disagreement.py` PASS |
| S3005 train_model_pair.py | ✅ | — | Adaptateurs maha/ewc/hdc/tinyol ; results.json model_a/model_b/ensemble(4 règles)/disagreement |
| S3006 exp Partie A (15 runs) | ✅ | — | 14 mesurés + 1 N/A honnête (maha_hdc×paderborn, feature_bounds non calibrés) |
| S3007 Partie B native | ✅ | — | native_to_fault configurable ; exp natives RUL CMAPSS (RMSE 29.8) + CWRU multi-classe (F1 0.981) |
| S3009 pipeline.c + sensor_stream | ✅ | — | PAIR_MODE 0x90/0xA0/0xB0 (nibble libre, sans collision) ; réponse 22 B ; sensor_stream sync ; `.bss=104 576 B` ; +3 tests Unity |
| S3010 exp board ≥1 paire | ✅ | — | Carte réelle : maha_ewc 256 µs comb. (5+251), maha_hdc 651 µs (5+647), overhead ~0, Gap 2 ✅ ; `board_pair_recorder.py` |
| S3012 tests + notebook + docs | ✅ | — | 19 tests Python + 3 Unity PASS ; notebook origines désaccord exécuté ; docs MAJ |
