# S3802 — Référence PC (4 politiques de mise à jour EWC)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 38 |
| **Priorité** | 🔴 Critique — point de comparaison ; sans elle, aucune parité ni delta board↔PC. |
| **Statut** | ✅ Implémenté — `scripts/run_sprint38_pc.py` ; **grille 16 cellules** (4 politiques × 2 datasets × 2 init_modes) produite. |
| **Durée estimée** | 6h |
| **Dépendances** | S3801 ✅ · `src/models/ewc/ewc_mlp.py` ✅ (`EWCMlpClassifier`/`EWCMlpMulticlass`) · `src/models/unsupervised/mahalanobis_detector.py` ✅ (`MahalanobisDetector`) · `src/evaluation/drift_detector.py` ✅ (`SlidingWindowDriftDetector`) · `src/evaluation/feature_conditions.py` ✅ (`load_condition_arrays`) · `src/evaluation/metrics.py` ✅ (`compute_cl_metrics`, `compute_fault_f1`) · `src/evaluation/anomaly_metrics.py` ✅ |
| **Fichiers cibles** | `scripts/run_sprint38_pc.py`, `experiments/exp_S38_PC_{policy}_{dataset}/results.json` + `checkpoints/ewc_head.pt` + `drift_thresholds.json` |
| **Références** | S3602 (référence PC Sprint 36 comme modèle d'orchestration) · Sprint 26 (rapporter AF — oubli) |

---

## Contexte

La référence PC implémente les **4 politiques** côté Python et sert de vérité-terrain pour la parité
board. Elle consomme **exactement** les colonnes/split que le board verra
(`load_condition_arrays(dataset, "5feat", "ewc", seed=42)`). Elle **calibre les seuils du gate** sur
l'enrôlement healthy puis les **exporte** (consommés par le firmware → parité par construction).

## Spec

Pour chaque `(policy ∈ {frozen, always, gated_truelabel, gated_pseudolabel}, dataset ∈ {monitoring, pronostia})` :

1. **Données** : `load_condition_arrays(dataset, "5feat", "ewc", seed=42)` (split test complet).
2. **Enrôlement one-class** : isoler les `n_samples` premiers échantillons **sains** ; fitter
   `MahalanobisDetector` dessus ; `SlidingWindowDriftDetector.set_thresholds_from_normal(maha_scores_healthy)`
   (P95 × multiplicateurs ← config).
3. **Streaming séquentiel** : rejouer le split test échantillon par échantillon. À chaque pas :
   - `score = maha.anomaly_score(x)` ; `verdict = drift.update(score)`.
   - Appliquer la politique :
     - `frozen` : prédire seulement (jamais de SGD).
     - `always` : prédire → `ewc.sgd_step(x, true_label)`.
     - `gated_truelabel` : prédire → si `verdict != NORMAL` : `ewc.sgd_step(x, true_label)`.
     - `gated_pseudolabel` : prédire → `FAULT` : `ewc.sgd_step(x, 1)` ; `DRIFT` : `maha.partial_fit(x)` (adapte le normal) ; `NORMAL` : rien.
   - Le **vrai label** sert toujours au scoring (F1, acc), jamais au SGD en P3.
4. **Métriques** → `results.json` :
   - `acc_matrix`(T×T) → `compute_cl_metrics` → `aa`, `af`, `bwt`, `acc_final`.
   - `f1_faulty`, `f1_macro`, `roc_auc`.
   - **`n_updates`** (nombre de SGD réellement effectués) + **`update_rate`** = `n_updates / n_samples`.
   - **confusion verdict↔vérité** : matrice {NORMAL,DRIFT,FAULT} × {sain,faulty} (diagnostic drift↔faute).
   - `n_params`, `ram_peak_bytes` (tracemalloc — inclut le gate), `inference_latency_ms`.
5. **Artefacts** : checkpoint `ewc_head.pt` (réutilisé par le board → parité exacte frozen),
   `drift_thresholds.json` (`fault_threshold`, `drift_threshold`, `window_size`, `drift_ratio` → export firmware),
   dump `samples` `[{idx, true, pred, confidence, verdict, updated}]`.

**Axe `init_modes`** (décision utilisateur — les deux testés) :
- `pretrained` : base CL offline partagée (`train_and_eval` Sprint 36) avant le streaming ;
  `frozen` = plancher d'un **modèle déployé**.
- `scratch` : pas d'entraînement offline ; le streaming **est** l'apprentissage ; `frozen` ≈ aléatoire.

Helper local `_sgd_step(model, optimizer, x, label)` (forward + `CrossEntropyLoss` + `ewc_penalty` +
backward + step) = équivalent 1-échantillon du `ewc_sgd_step` firmware. `acc_matrix(T×T)` reconstruit
aux frontières de tâche pendant le streaming.

**Règles** : valeurs `null` tant que non exécuté (aucun chiffre inventé) ; hyperparamètres ← `board_ewc.yaml` ;
même seed (42) et même loader que le board.

## Résultats (16 cellules, `experiments/exp_S38_PC_{policy}_{ds}_{init}/`)

`update_rate` strictement ordonné **frozen=0 < gated≈0.02 < always=1.0** sur toutes les cellules.
Contraste `init_mode` net (F1_faulty) :

| | frozen | always | gated_truelabel | gated_pseudolabel |
|---|---|---|---|---|
| monitoring `pretrained` | 0.890 | 0.838 | 0.890 | 0.890 |
| pronostia  `pretrained` | 0.946 | 0.823 | 0.884 | 0.697 |
| monitoring `scratch`    | 0.255 | 0.697 | 0.187 | 0.187 |
| pronostia  `scratch`    | 0.105 | 0.705 | 0.280 | 0.190 |

Lecture : en `pretrained`, le plancher `frozen` est **déjà élevé** (modèle déployé) et `always` peut
même **sur-adapter** (légère baisse) → les politiques gated préservent à coût quasi nul. En `scratch`,
`frozen` est le **plancher absolu** (≈ classe majoritaire) et `always` le plafond appris en ligne ;
les gated, peu d'updates, restent intermédiaires/bruitées. C'est exactement le compromis recherché.

## Vérification

```bash
python scripts/run_sprint38_pc.py --config configs/sprint38_autonomous_update.yaml \
  --policy gated_pseudolabel --dataset monitoring   # → exp_S38_PC_gated_pseudolabel_monitoring/
```
- `update_rate` attendu : `frozen`=0 < `gated_*` < `always`=1.
- Confusion verdict↔vérité cohérente (FAULT majoritairement sur faulty ; DRIFT sur dérive saine).
