# S3205 — Évaluation board réelle, par seuil

| Champ | Valeur |
|-------|--------|
| **Sprint** | 32 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 5h |
| **Dépendances** | S3203/S3204 · infra board Sprints 26-28 ✅ |
| **Fichiers cibles** | `experiments/exp_S32_board_*/` · `scripts/run_board_threshold_sweep.py` · `scripts/train_board_reference.py` |
| **Références** | `scripts/export_weights_c.py`, `scripts/sensor_stream.py` (`--dump-samples`), `firmware/.../model_weights_ewc.h` |

---

## Résultat (implémentation)

**Découverte clé (impact parité)** : le firmware board est câblé à **5 features**, mais les runs PC S32
utilisent les dims natives (CMAPSS=5, Battery=7, Pronostia=13). Le firmware étant **agnostique au
dataset à 5 features**, aucune refonte n'est nécessaire ; la parité exige des **modèles de référence
board 5-feat** (`scripts/train_board_reference.py`) entraînés sur EXACTEMENT les features streamées.

**Parité board↔PC architecturalement possible pour EWC + Mahalanobis seulement** (export de poids →
header). HDC (projection embarquée, dim 1000≠1024, init en ligne) et TinyOL (pas de checkpoint, archi
board distincte) → **HW-only** (latence/`.bss`, parité N/A par construction — décision utilisateur).

Nouveautés :
- **Firmware** : `inc/model_weights_ewc.h` (vide par défaut → init Xavier ; régénéré par export) +
  `pipeline.c` `ewc_head_load_or_init()` charge `g_ewc_head` depuis Flash si `EWC_HEAD_WEIGHTS_PROVIDED`
  (sinon fallback historique — **aucune régression**, 94/96 tests Unity, 2 TinyOL préexistants).
- **Export** : `export_weights_c.py --ewc-head` (EWCMlpMulticlass 5→32→16→2 == `ewc_forward`).
- **Streaming** : `sensor_stream.py --dump-samples` (features+pred par échantillon → parité) ; battery
  ajouté (`sensor_sim._load_battery`, `configs/battery_feature_subset.yaml`).
- **Driver** : `run_board_threshold_sweep.py` (train→export→build→**1 flash/cellule**→stream 4 modèles→
  parité). Streaming parité **sans `--update`** (poids figés) ; `--rate-hz 50 --protocol-version 3`.

Résultats board réelle (NUCLEO-F439ZI) : **`.bss=104 596 B` invariant au seuil** ; latences P50
Maha ≈ 5 µs / EWC ≈ 50 µs / TinyOL ≈ 5 µs / HDC ≈ 585 µs — **toutes ≪ 100 ms (Gap 2 ✅)** ;
**parité EWC+Maha exacte sur les 5 seuils CMAPSS (10/10)**. Pronostia/Battery : voir
`experiments/exp_S32_board_sweep_summary.json`.

---

## Contexte

Mesurer latence (DWT µs) et RAM (`.bss`) **sur la NUCLEO-F439ZI réelle**, par seuil, et vérifier la **parité board↔PC** des prédictions (protocole Sprints 26-28). Le seuil ne change que les labels de référence côté PC ; la sortie du modèle board doit rester identique à la prédiction PC sur la même entrée.

---

## Spec

```bash
# par (modèle, dataset, seuil) :
python scripts/export_weights_c.py --model {model} --config configs/sweep/{dataset}_thr{XX}.yaml
make -C firmware/stm32f4_blink flash
python scripts/sensor_stream.py --port /dev/ttyACM0 --dataset {dataset} \
       --output experiments/exp_S32_board_{model}_{dataset}_thr{XX}/stream.json --update
python scripts/board_experiment_recorder.py --config configs/board_{model}.yaml \
       --output experiments/exp_S32_board_{model}_{dataset}_thr{XX}
```

- **Parité** : prédictions board == prédictions PC sur les mêmes entrées (tolérance numérique nulle attendue).
- Latence < 100 ms (Gap 2) pour tous les seuils.
- RAM profiling obligatoire (`.bss`) ; attendu **invariant au seuil**.

---

## Vérification

```bash
# 1 seuil représentatif sur board réelle
python scripts/sensor_stream.py --port /dev/ttyACM0 --dataset cmapss --n-samples 200 \
       --output experiments/exp_S32_board_ewc_cmapss_thr30/stream.json
# vérifier results.json : inference_latency_ms < 100, ram_peak_bytes, parité OK
```
