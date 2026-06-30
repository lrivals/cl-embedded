# S3007 — Partie B : paires en tâches natives (hétérogène)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 30 |
| **Priorité** | 🟡 Important |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 3h |
| **Dépendances** | S3001 (`ModelPair`) · S3003 (désaccord) · S3006 (Partie A produite) |
| **Fichiers cibles** | `src/ensemble/model_pair.py` (mode `"native"`), `experiments/exp_S30_PC_native_*/` |
| **Références** | `src/data/cmapss_loader.py` (RUL), `src/data/cwru_dataset.py` (multi-classe) |

---

## Contexte

Partie A compare tout en normal-vs-fault. Partie B conserve la **tâche native** du modèle supervisé pour mesurer la complémentarité réelle Mahalanobis (anomalie) + modèle natif. Les sorties ne sont plus directement comparables → le désaccord est **redéfini par dataset**.

| Dataset | Tâche native supervisée | Définition du désaccord avec Mahalanobis |
|---------|------------------------|------------------------------------------|
| CMAPSS | RUL (régression) | RUL < seuil de criticité ↔ anomalie détectée |
| CWRU | multi-classe faute | classe ≠ normal ↔ anomalie détectée |
| Monitoring | binaire faute | faute ↔ anomalie (déjà aligné) |
| Pronostia | condition | dégradé ↔ anomalie |
| Paderborn | degré de dommage | endommagé ↔ anomalie |

---

## Spec

- Étendre `ModelPair` mode `"native"` : ne binarise pas la sortie supervisée ; expose un mapping configurable `native_to_fault` (seuil/règle par dataset, via config).
- Le désaccord compare la **décision binaire dérivée** (faute oui/non) des 2 modèles.
- Expériences `experiments/exp_S30_PC_native_{pair}_{dataset}/` (au moins CMAPSS RUL + CWRU multi-classe).

---

## Vérification

```bash
python scripts/train_model_pair.py --config configs/board_pair_maha_ewc.yaml --dataset cmapss --mode native
ls experiments/exp_S30_PC_native_maha_ewc_cmapss/
```

---

## Bilan (S3007 ✅)

`native_to_fault(pred, rule, threshold)` ajouté à `src/ensemble/model_pair.py` (exporté via
`src/ensemble/__init__.py`) : règles **configurables** `identity` / `rul_threshold` /
`nonzero_class`, pilotées par le bloc `native:` de la config paire — aucun seuil en dur
(règle CLAUDE.md). Le mode `"native"` de `ModelPair` (déjà présent) ne binarise plus la sortie
supervisée ; le **désaccord** compare la décision binaire dérivée (`native_to_fault` sur la
sortie native du supervisé) vs `detector.predict` (Mahalanobis).

`scripts/train_model_pair.py --mode native` entraîne le supervisé en **tâche native** via les
pipelines existants : `EWCMlpRegressor` (CMAPSS RUL, `train_ewc_rul`) et `EWCMlpMulticlass`
(CWRU multi-classe, `train_ewc_multiclass`). Métrique native propre **+** AUROC/F1 binarisé **+**
désaccord. Native restreint au supervisé **EWC** (les configs natives existantes sont au schéma
EWC ; HDC/TinyOL sans config native calibrée → `status: skipped` honnête, pas de chiffre inventé).

**Expériences produites** (≥1 RUL + ≥1 multi-classe requis ✅) :

| Exp | Tâche native | métrique native | AUROC maha (faute) | F1_faute sup | désaccord |
|-----|--------------|-----------------|--------------------|--------------|-----------|
| exp_S30_PC_native_maha_ewc_cmapss | RUL (régression) | RMSE_moy = 29.82 cycles | 0.551 | 0.396 | 0.319 |
| exp_S30_PC_native_maha_ewc_cwru | multi-classe (10) | F1_macro_moy = 0.981 | 0.571 | 0.984 | 0.641 |
| exp_S30_PC_native_maha_hdc_cwru | — | N/A (skipped) | — | — | — |

**Lecture** : en tâche native, l'EWC multi-classe CWRU est excellent (F1_macro 0.981) et sa
décision faute dérivée écrase Mahalanobis (0.984 vs 0.571 AUROC) → fort désaccord (0.64), le
détecteur générique apportant peu sur ce dataset bien séparé. Sur CMAPSS RUL, l'EWC atteint
RMSE 29.8 cycles ; binarisé au seuil de criticité (RUL ≤ 30), les deux modèles restent faibles
(détection de panne CMAPSS difficile) avec un désaccord modéré (0.32).
