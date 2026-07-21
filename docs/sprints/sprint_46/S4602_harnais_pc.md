# S4602 — Harnais PC unifié des trois moments de quantification

| Champ | Valeur |
|-------|--------|
| **Sprint** | 46 |
| **Priorité** | 🔴 Critique — c'est le cœur technique du sprint ; il est le seul endroit où le chemin **both** (QAT→export PTQ) est câblé (n'existe nulle part aujourd'hui). |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Durée estimée** | 9h |
| **Dépendances** | S4601 ✅ (taxonomie + clé `quant_moment`) · `scripts/train_ewc.py::train_ewc` ✅ · `src/models/ewc/ewc_mlp_int8.py` ✅ · `src/utils/int8_c_emulation.py` ✅ · `scripts/benchmark_int8_fp32.py` (schéma JSON) ✅ · `scripts/run_s39_quant_sweep.py` (tables RAM/BOPs) ✅ |
| **Fichiers cibles** | `scripts/run_s46_quant_moment.py` (nouveau) |
| **Références** | S2801 (benchmark_int8_fp32) · S3906 (quant_sweep) · S3918 (matched_compare) · CLAUDE.md § « Reproductibilité » |

---

## Contexte

Trois moments, mais un seul harnais. `before` et `after` existent chacun dans une voie séparée (Sprint 28
QAT / Sprint 39 émulateur PTQ) ; `both` n'existe pas. Cette tâche écrit `run_s46_quant_moment.py` qui
**réutilise** les briques existantes et **ajoute** le seul maillon manquant : extraire les poids d'un
modèle QAT-entraîné et les pousser dans le noyau PTQ. Il itère (modèle × dataset × moment) et émet un JSON
homogène, aligné sur les conventions S28/S39, sans jamais réentraîner deux fois pour rien.

## Spec

### 1. Interface CLI

```bash
python scripts/run_s46_quant_moment.py \
    --model {ewc,tinyol} \
    --dataset {monitoring,pronostia} \
    --moment {fp32,before,after,both} \
    --after-scheme {legacy_c,per_tensor_calib,per_channel_int8,q15}  # requis pour after/both
    --config configs/quant_moment/<model>_<dataset>.yaml \
    --output experiments/exp_S46_<model>/<dataset>_<moment>.json \
    [--seed 42] [--n-samples N]
```

Un `--moment all` optionnel enchaîne `fp32 → before → after → both` en un run (réutilise le modèle FP32
et le modèle QAT une seule fois).

### 2. Les quatre chemins d'exécution

| Moment | Chemin | Réutilise |
|--------|--------|-----------|
| **fp32** | entraîne `EWCMlpClassifier` FP32, évalue | `train_ewc` + `EWCAdapter.build_fp32` |
| **before** | entraîne `EWCMlpInt8Classifier` (fake-quant dans la boucle), évalue **avec fake-quant à l'inférence** | `train_ewc` (branche INT8, cf. `_run_int8_comparison`) |
| **after** | entraîne FP32, **extrait** `state_dict`, `EWCHeadWeights.from_state_dict`, `forward_quant(cfg=after_scheme)` | modèle FP32 de la ligne `fp32` + `int8_c_emulation` |
| **both** | entraîne `EWCMlpInt8Classifier` (QAT), **extrait ses poids `fc1/fc2/fc3`**, `EWCHeadWeights.from_state_dict`, `forward_quant(cfg=after_scheme)` | modèle QAT de la ligne `before` + `int8_c_emulation` |

**Le maillon neuf = `both`** : la fonction `qat_weights_to_ptq(int8_model, after_scheme) -> metric` qui
prend un `EWCMlpInt8Classifier` déjà entraîné, lit ses poids (les fake-quant modules exposent les poids
FP32 sous-jacents `fcN.weight`), construit `EWCHeadWeights`, et évalue via `forward_quant`. C'est
faisable sans code lourd car les noms de couches coïncident (`fc1/fc2/fc3`).

> **Économie de calcul** : en `--moment all`, on entraîne **au plus deux fois** (un FP32, un QAT) et on
> dérive `after` du FP32 et `both` du QAT. Aucun ré-entraînement redondant.

### 3. Métriques reportées (natives au modèle)

- **EWC** : `auroc` (binarisé normal-vs-faute, comme S28), `ram_weights_bytes`, `lat_proxy_rel`
  (fp32=1.0, int8≈0.0625, q15≈0.25 — table de `run_s39_quant_sweep.py`), `delta_metric` vs fp32.
- **TinyOL** : `recon_error` / `f1`, mêmes champs RAM/latence-proxy.

Les tables `SCHEME_BITS` / `SCHEME_WEIGHT_BYTES` de `run_s39_quant_sweep.py` sont la source unique du
comptage RAM/BOPs — ne pas les redéfinir.

## Format de sortie

Un JSON par (modèle, dataset), nesté par moment (aligné S28/S39) :

```json
{
  "model": "ewc",
  "dataset": "monitoring",
  "metric_name": "auroc",
  "seed": 42,
  "config_path": "configs/quant_moment/ewc_monitoring.yaml",
  "timestamp": "<iso8601>",
  "moments": {
    "fp32":   { "metric": null, "ram_weights_bytes": null, "lat_proxy_rel": 1.0 },
    "before": { "metric": null, "ram_weights_bytes": null, "lat_proxy_rel": null, "note": "borne haute (fake-quant inférence)" },
    "after":  { "metric": null, "ram_weights_bytes": null, "lat_proxy_rel": null, "after_scheme": "per_tensor_calib" },
    "both":   { "metric": null, "ram_weights_bytes": null, "lat_proxy_rel": null, "after_scheme": "per_tensor_calib", "note": "fidèle au déploiement (noyau entier)" }
  },
  "delta_before_vs_fp32": null,
  "delta_after_vs_fp32":  null,
  "delta_both_vs_fp32":   null,
  "gap3_metric_ok_both":  null,
  "gap3_ram_ok":          null
}
```

`null` = non encore mesuré (règle « aucun chiffre inventé »). Les `delta_*` sont calculés au run.

## Contraintes

- **Déterminisme** : `set_seed(42)` avant chaque entraînement ; le modèle FP32 et le modèle QAT doivent
  être reproductibles run-à-run.
- **Aucun hyperparamètre en dur** : tout vient de la config (couches, λ, lr, epochs) ; `after_scheme`
  mappe vers un preset `QuantConfig` existant.
- **`both` ne réentraîne pas** : il consomme le modèle QAT de `before`.
- Le champ `note` distingue explicitement `before` (borne haute) de `both` (déploiement).

## Vérification

```bash
# EWC, tous moments, un dataset
python scripts/run_s46_quant_moment.py --model ewc --dataset monitoring \
    --moment all --after-scheme per_tensor_calib \
    --config configs/quant_moment/ewc_monitoring.yaml \
    --output experiments/exp_S46_ewc/monitoring_all.json

# Le JSON contient les 4 moments et 3 deltas, sans null résiduel après run
python -c "import json; d=json.load(open('experiments/exp_S46_ewc/monitoring_all.json')); \
assert set(d['moments'])=={'fp32','before','after','both'}; \
assert d['delta_both_vs_fp32'] is not None"

# Déterminisme : deux runs → mêmes métriques
```

---

## Résolution (implémentée)

✅ **Implémenté**. `scripts/run_s46_quant_moment.py` créé. Réutilise les briques existantes,
n'ajoute que le maillon `both`.

**Réutilisation (source unique, 0 duplication)** :
- `scripts/benchmark_int8_fp32.py` → `EWCAdapter` (câblage FP32/QAT réel : `build_fp32`,
  `build_int8`, `load_tasks`, `train`, `evaluate` AUROC macro-tâches), helpers
  `_first_task_train_X`, `_mean_auroc_over_tasks`, `_truncate_tasks`.
- `src/utils/int8_c_emulation.py` → `EWCHeadWeights.from_state_dict`, `forward_quant`,
  `calibrate_activations`, presets `QuantConfig`.
- `scripts/run_s39_quant_sweep.py` → `SCHEME_WEIGHT_BYTES`, `_proxies` (comptage RAM/latence-proxy).

**Les quatre chemins** :
- `fp32` : `build_fp32` → `train` → `evaluate` ; RAM = `params×4`, `lat_proxy_rel=1.0`.
- `before` : `build_int8` (QAT) → `train` → `evaluate` **avec fake-quant à l'inférence** ;
  `lat_proxy_rel=null` + `note` « borne haute (fake-quant inférence) ».
- `after` : poids FP32 figés → `_weights_from_model` → `calibrate_activations` (1re tâche) →
  `forward_quant(cfg=after_scheme)` par tâche → logit binaire `[:,0]` = score AUROC.
- **`both` (maillon neuf)** : poids **QAT appris** → même chemin PTQ que `after`. Les couches
  `fc1/fc2/fc3` de `EWCMlpInt8Classifier` (poids FP32 sous-jacents) coïncident avec la tête
  FP32 → `from_state_dict` fonctionne sans code lourd. `note` « fidèle au déploiement (noyau entier) ».

**Économie de calcul** : `--moment all` entraîne **au plus 2 fois** (1 FP32, 1 QAT) — `after`
dérivé du FP32, `both` du QAT. `set_seed(seed)` avant chaque entraînement.

**Sortie** : JSON schéma S4602 (4 moments présents, `null` si non calculé, `delta_*_vs_fp32`,
`gap3_metric_ok_both` = `|Δboth|<0.02`, `gap3_ram_ok` = ratio RAM > 1), + `config_snapshot.yaml`.
Point technique tracé : `after_scheme` (CLI `legacy_c|per_tensor_calib|per_channel_int8|q15`)
mappe vers `(preset QuantConfig, clé RAM run_s39)` via la table `AFTER_SCHEMES`.

**Note config** : la clé top-level `model:` du spec YAML entrerait en collision avec la section
`model:` (dict) de `ewc_config.yaml` lors du deep-merge `extends` → le modèle et le dataset sont
passés sur la **CLI** (`--model`/`--dataset`), les configs ne portent que `quant_moment`/
`after_scheme`/`metric` (scalaires sans collision).

**Vérification (smoke `--n-samples 200`, puis run réel)** : 4 moments calculés, deltas non-null,
Gap 3 flags OK. Voir S4603 pour les chiffres de production.
