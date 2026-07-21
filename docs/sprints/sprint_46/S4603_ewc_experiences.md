# S4603 — Expériences EWC : 3-way × {Monitoring, Pronostia}

| Champ | Valeur |
|-------|--------|
| **Sprint** | 46 |
| **Priorité** | 🔴 Critique — c'est le résultat central du sprint (EWC est le modèle prioritaire). |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Durée estimée** | 5h |
| **Dépendances** | S4602 ✅ (harnais) · loaders Monitoring/Pronostia ✅ · `configs/ewc_int8_{monitoring,pronostia}.yaml` (voie QAT réutilisée) |
| **Fichiers cibles** | `configs/quant_moment/ewc_{monitoring,pronostia}.yaml`, `experiments/exp_S46_ewc/` |
| **Références** | exp_S28_PC_ewc_hdc (QAT préservé) · exp_S39_ablation (PTQ effondrement/récupération) · S4602 |

---

## Contexte

EWC est le modèle où les trois moments coexistent proprement. Cette tâche exécute la grille
**{fp32, before, after, both} × {Monitoring, Pronostia}** via le harnais S4602 et produit les JSON.
Le message scientifique attendu (à **confirmer par la mesure**, pas à préjuger) : le QAT (`before`)
préserve la métrique, la PTQ naïve (`after` = `legacy_c`) s'effondre, la PTQ calibrée
(`after` = `per_tensor_calib`/`per_channel_int8`) récupère, et `both` (QAT puis export PTQ calibré) est
la variante **fidèle au déploiement** — potentiellement la meilleure des colonnes quantifiées.

## Spec

### 1. Grille d'expériences

| Dataset | Moments | after_scheme balayés |
|---------|---------|----------------------|
| Monitoring (D2) | fp32, before, after, both | `legacy_c`, `per_tensor_calib`, `per_channel_int8` |
| Pronostia (D4) | fp32, before, after, both | `legacy_c`, `per_tensor_calib`, `per_channel_int8` |

Le balayage `after_scheme` sur `after`/`both` permet d'isoler l'effet **calibration** (naïf vs calibré vs
per-canal) — c'est le lien direct avec l'ablation S3904.

### 2. Configs à produire

`configs/quant_moment/ewc_monitoring.yaml` et `ewc_pronostia.yaml`, format S4601 :

```yaml
model: ewc
dataset: monitoring          # / pronostia
extends: ewc_int8_monitoring # réutilise archi + hyperparamètres EWC existants
quant_moment: both           # surchargé par --moment sur la CLI
after_scheme: per_tensor_calib
seed: 42
metric: auroc
```

### 3. Exécution

```bash
for ds in monitoring pronostia; do
  python scripts/run_s46_quant_moment.py --model ewc --dataset $ds \
    --moment all --after-scheme per_tensor_calib \
    --config configs/quant_moment/ewc_$ds.yaml \
    --output experiments/exp_S46_ewc/${ds}_all.json
done
```

## Format de sortie

`experiments/exp_S46_ewc/{monitoring,pronostia}_all.json` — schéma S4602. Table de synthèse attendue
(valeurs `pending` tant que non exécuté, **aucun chiffre inventé**) :

| Dataset | fp32 AUROC | before (QAT) | after naïf | after calibré | both | Δbest vs fp32 | Gap 3 métrique |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Monitoring | pending | pending | pending | pending | pending | pending | pending |
| Pronostia | pending | pending | pending | pending | pending | pending | pending |

## Contraintes

- Métrique = **AUROC** (voie de référence EWC fixée en S4601), pas de mélange avec F1.
- `seed=42`, `config_snapshot` déposé dans `exp_S46_ewc/` (convention CLAUDE.md).
- Interpréter `before` comme **borne haute** (fake-quant inférence) et `both` comme **déploiement** dans
  le commentaire de résultats — ne pas les confondre.

## Vérification

```bash
ls experiments/exp_S46_ewc/monitoring_all.json experiments/exp_S46_ewc/pronostia_all.json
python -c "import json; d=json.load(open('experiments/exp_S46_ewc/monitoring_all.json')); \
print({k:v['metric'] for k,v in d['moments'].items()})"
# Balayage after_scheme : legacy vs calibré doivent différer (effet calibration)
```

---

## Résolution (implémentée)

✅ **Implémenté et exécuté** (PC, seed 42). Configs `configs/quant_moment/ewc_{monitoring,pronostia}.yaml`
(héritant de `ewc_int8_{monitoring,pronostia}.yaml` via `extends`) + grille exécutée via le
harnais S4602. **7 JSON** dans `experiments/exp_S46_ewc/` : `{ds}_all.json` (per_tensor_calib
canonique), `{ds}_legacy_c.json`, `{ds}_per_channel_int8.json` (balayage after_scheme) +
`config_snapshot.yaml`.

### Table de synthèse (AUROC mesurée, seed 42)

| Dataset | fp32 | before (QAT) | after naïf (`legacy_c`) | after calibré (`per_tensor`) | both (QAT→PTQ calibré) | Δboth vs fp32 | Gap 3 métrique |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Monitoring (D2) | 0.9746 | 0.9745 | **0.4975** | 0.9746 | 0.9747 | +0.00002 | ✅ |
| Pronostia (D4) | 0.9987 | 0.9978 | **0.5458** | 0.9978 | 0.9978 | −0.0009 | ✅ |

*(per_channel_int8 ≈ per_tensor_calib : Monitoring after 0.9749 / both 0.9750 ; Pronostia
after 0.9978 / both 0.9982 — même récupération.)*

### Message scientifique (confirmé par la mesure, non préjugé)

- **QAT (`before`) préserve** la métrique (Δ ≤ 0.001) — mais c'est une **borne haute**
  (fake-quant à l'inférence, jamais exécuté sur la carte).
- **PTQ naïve (`after` = `legacy_c`) s'effondre** : AUROC → quasi-aléatoire (0.498 / 0.546),
  soit **Δ ≈ −0.45 à −0.55** — cohérent avec l'ablation S3904 (échelle figée 1/128 non calibrée).
- **PTQ calibrée (`after` = `per_tensor_calib`/`per_channel_int8`) récupère tout** :
  **+0.477 (Monitoring) / +0.452 (Pronostia)** vs le naïf → retour au niveau FP32.
- **`both` (QAT puis export PTQ calibré) = variante fidèle au déploiement**, métrique préservée
  (Gap 3 métrique ✅ sur les 2 datasets), RAM des poids ÷4 (Gap 3 RAM ✅).

Le balayage `after_scheme` **isole l'effet calibration** (naïf vs calibré diffèrent de ~0.45–0.48
d'AUROC) — c'est le lien direct avec l'ablation S3904. Métrique = **AUROC uniquement** (voie de
référence EWC fixée en S4601), pas de mélange F1.

### Vérification

```
$ ls experiments/exp_S46_ewc/{monitoring,pronostia}_all.json          # présents
$ python -c "import json;d=json.load(open('experiments/exp_S46_ewc/monitoring_all.json'));\
  assert set(d['moments'])=={'fp32','before','after','both'};\
  assert d['delta_both_vs_fp32'] is not None"                          # OK
# effet calibration : after legacy_c ≠ after per_tensor_calib
#   monitoring 0.498 vs 0.975 (diff 0.477) · pronostia 0.546 vs 0.998 (diff 0.452)
```
