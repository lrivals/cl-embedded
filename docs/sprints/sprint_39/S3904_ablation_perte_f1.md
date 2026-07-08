# S3904 — Ablation chiffrée de la perte F1 INT8

| Champ | Valeur |
|-------|--------|
| **Sprint** | 39 |
| **Priorité** | 🔴 Critique — attribue la perte F1 à chaque facteur |
| **Statut** | ✅ Implémenté (1 juillet 2026) |
| **Durée estimée** | 3h |
| **Dépendances** | S3903 ✅ (émulateur validé) · `scripts/train_board_reference.py` |
| **Fichier cible** | `scripts/run_s39_int8_ablation.py` → `experiments/exp_S39_ablation/` |
| **Références** | `src/utils/int8_c_emulation.py` (`ABLATION_LADDER`) · `src/evaluation/metrics.py` (F1) |

---

## Contexte

On sait que l'INT8 board dégrade la F1 (0.92 → 0.14). Ce travail **décompose** cette perte : combien de F1
chaque correctif récupère, en activant les facteurs un par un le long de `ABLATION_LADDER` :

```
legacy_c → fix_acc32 → per_tensor_calib → per_channel_int8 → q15
```

Chaque marche n'active **qu'un** changement, ce qui isole sa contribution.

## Protocole

Pour chaque dataset (cmapss, cwru, monitoring, pronostia, paderborn) en condition board `5feat` :

1. Entraîner la tête EWC board (réutilise `train_board_reference.py`).
2. Calculer F1_faulty pour FP32 puis pour chaque marche de l'échelle.
3. Reporter Δ F1 marche↦marche (contribution du facteur isolé).

## Format de sortie (`experiments/exp_S39_ablation/{dataset}.json`)

```json
{
  "dataset": "pronostia",
  "condition": "5feat",
  "f1_fp32": 0.916,
  "ladder": [
    {"scheme": "legacy_c",         "f1": 0.14, "delta_prev": null,  "factor": "firmware actuel"},
    {"scheme": "fix_acc32",        "f1": 0.xx, "delta_prev": 0.xx,  "factor": "accumulateur int32"},
    {"scheme": "per_tensor_calib", "f1": 0.xx, "delta_prev": 0.xx,  "factor": "scale calibré (vs 1/128)"},
    {"scheme": "per_channel_int8", "f1": 0.xx, "delta_prev": 0.xx,  "factor": "par-canal"},
    {"scheme": "q15",              "f1": 0.xx, "delta_prev": 0.xx,  "factor": "16-bit"}
  ],
  "dominant_factor": "<le facteur au plus grand delta_prev>"
}
```

## Résultat attendu / interprétation

| Si le facteur dominant est… | Conclusion |
|-----------------------------|------------|
| `per_channel` ou `per_tensor_calib` | la cause racine est l'**échelle 1/128 non calibrée** (≠ QAT PC) |
| `fix_acc32` | l'**overflow int16** est le coupable principal → fix C trivial à fort impact |
| `q15` | la résolution 8-bit est insuffisante → recommander Q15 (×2 RAM) |

> Aucune valeur n'est écrite à la main : `run_s39_int8_ablation.py` produit les JSON. Le tableau ci-dessus
> est un gabarit ; les chiffres réels sortent de l'exécution.

## Vérification

```bash
python scripts/run_s39_int8_ablation.py            # → experiments/exp_S39_ablation/*.json
python -c "import json,glob; [print(json.load(open(f))['dominant_factor']) for f in glob.glob('experiments/exp_S39_ablation/*.json')]"
```

---

## Bilan d'implémentation (1 juillet 2026)

**Livré** : `scripts/run_s39_int8_ablation.py` → **5 JSON** `experiments/exp_S39_ablation/{dataset}.json`
(cmapss, cwru, monitoring, pronostia, paderborn), condition board `5feat`. Réutilise
`load_condition_arrays` (features board, S3508), les hyperparamètres EWC de `train_board_reference.py`
(dim d'entrée = dim de la condition → gère monitoring 4-feat), l'émulateur `ABLATION_LADDER` (S3902) et
`compute_fault_f1` (S3504). **PC-only, sans carte.** Aucune valeur écrite à la main.

### Résultats mesurés

| Dataset | F1 fp32 | legacy_c | fix_acc32 | per_tensor_calib | per_channel | q15 | Facteur dominant |
|---------|:------:|:-------:|:--------:|:----------------:|:-----------:|:---:|------------------|
| cmapss | 0.448 | 0.227 | 0.294 | **0.448** | 0.448 | 0.448 | `per_tensor_calib` (+0.153) |
| cwru | 0.996 | 0.929 | 0.947 | **0.994** | 0.994 | 0.996 | `per_tensor_calib` (+0.048) |
| monitoring | 0.919 | 0.118 | 0.042 | **0.920** | 0.919 | 0.919 | `per_tensor_calib` (+0.878) |
| pronostia | 0.962 | 0.066 | 0.067 | **0.946** | 0.943 | 0.962 | `per_tensor_calib` (+0.879) |
| paderborn | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 | `fix_acc32` (+0.0, dégénéré) |

### Interprétation

- **Cause racine = l'échelle `1/128` non calibrée**, pas l'overflow int16. Le facteur dominant est
  **`per_tensor_calib` sur 4/5 datasets** : passer d'un scale figé à un scale calibré récupère l'essentiel
  de la F1 (jusqu'à +0.88 sur monitoring/pronostia). C'est cohérent avec S3901 (F2/F3 : PTQ grossière à
  échelle fixe ≠ QAT PC per-canal) et **infirme** l'hypothèse que l'overflow int16 (F1) serait le coupable
  principal — `fix_acc32` seul apporte peu, voire dégrade (monitoring −0.076 : lever le wrap sans
  recalibrer laisse le clamp ReLU/1-128 actif).
- **`per_channel` et `q15` n'ajoutent quasi rien** au-delà de `per_tensor_calib` sur ces têtes 5→32→16→2
  (dynamique de poids homogène) : le 8-bit calibré suffit, Q15 (×2 RAM) n'est pas requis ici — nuance vs
  Mahalanobis grande dynamique (Sprint 34) où Q15 était nécessaire.
- **paderborn dégénéré** (`legacy_c` == `fp32` == 0.800) : class-incremental mono-classe/tâche (Sprint 35),
  la tête EWC prédit une classe quasi constante → la quantification ne change pas la prédiction, donc aucune
  perte à décomposer. `dominant_factor` retombe honnêtement sur `fix_acc32` avec `delta_prev = 0`.

> Conclusion pour le manuscrit : le correctif à fort impact est **calibrer l'échelle des poids/activations
> à l'export** (mirroir QAT PC), pas d'abord le passage int32. Réponse chiffrée à S3901 (F2/F3 > F1).
