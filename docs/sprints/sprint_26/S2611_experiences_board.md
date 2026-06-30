# S2611–S2613 — Expériences board NUCLEO-F439ZI

| Champ | Valeur |
|-------|--------|
| **Sprint** | 26 |
| **Priorité** | 🔴 Critique (S2611) / 🟡 Important (S2612, S2613) |
| **Statut** | ✅ Mesuré sur board réel (NUCLEO-F439ZI, 2026-06-12) — RUL ✅ ; multi-classe : critère F1 non atteint **mais diagnostiqué et résolu** (oubli catastrophique, pas un bug de portage ; voir Résultats + FIXME(gap1) RÉSOLU) |
| **Durée estimée** | S2611 : 2h / S2612 : 2h / S2613 : 1h = 5h total |
| **Dépendances** | S2601–S2610 ✅ (firmware compilé et flashé, scripts simulation opérationnels), NUCLEO-F439ZI connectée sur `/dev/ttyACM0`, résultats PC Sprint 25 (`exp_S25_01` et `exp_S25_03`) |
| **Fichiers cibles** | `experiments/exp_S26_01/`, `experiments/exp_S26_02/`, `experiments/exp_S26_03/` |
| **Référence** | Format `experiments/exp_S23_01/` (structure `config_snapshot.yaml` + `results.json` + `log.txt`), critères de succès du sprint (`S2600_sprint_26.md`) |

---

## Structure de chaque dossier expérience

```
experiments/exp_S26_XX/
├── config_snapshot.yaml     ← copy des configs YAML utilisées (firmware + loader)
├── results.json             ← métriques mesurées board + comparaison PC
└── log.txt                  ← sortie console complète du script simulate_*_board.py
```

---

## S2611 — `exp_S26_01` : EWC RUL board / CMAPSS FD001

### Objectif

Mesurer sur board : RMSE de prédiction RUL, latence DWT, footprint SRAM .bss. Comparer au RMSE PC de `exp_S25_01`.

### Commandes d'exécution

```bash
# 1. Compiler et flasher le firmware (avec poids RUL chargés)
make -C firmware/stm32f4_blink all
make -C firmware/stm32f4_blink flash

# 2. Lancer la simulation (board connectée)
python scripts/simulate_rul_board.py \
    --port /dev/ttyACM0 \
    --config configs/cmapss_feature_subset.yaml \
    --n-samples 200 \
    --task-id 0 \
    --update \
    --output experiments/exp_S26_01/board_rul_results.json \
    2>&1 | tee experiments/exp_S26_01/log.txt

# 3. Récupérer le footprint SRAM
arm-none-eabi-size firmware/stm32f4_blink/build/stm32f4_blink.elf \
    >> experiments/exp_S26_01/log.txt
```

### `config_snapshot.yaml` à créer

```yaml
# exp_S26_01 — EWC RUL board / CMAPSS FD001
# Date : 2026-07-29
experiment_id: exp_S26_01
model: ewc_head_regression
dataset: CMAPSS FD001
task: RUL continuu (regression)
board: NUCLEO-F439ZI (Cortex-M4 @ 180 MHz, 256 Ko SRAM)
firmware: ewc_head_regression.c + pipeline.c (Sprint 26)
model_weights: experiments/exp_S25_01/model_ewc_reg.pt

firmware_config:
  EWC_REG_IN: 5
  EWC_REG_H1: 32
  EWC_REG_H2: 16
  EWC_REG_OUT: 1
  EWC_REG_LR: 0.001
  EWC_REG_FISHER_DECAY: 0.99
  EWC_LAMBDA: 400.0

loader_config: configs/cmapss_feature_subset.yaml
n_samples: 200
n_tasks: 1  # FD001 seulement
```

### `results.json` attendu

```json
{
  "experiment_id": "exp_S26_01",
  "model": "EWC Regression",
  "dataset": "CMAPSS FD001",
  "board": "NUCLEO-F439ZI",
  "n_samples": 200,

  "rmse_board": null,
  "rmse_pc": null,
  "rmse_ratio_board_over_pc": null,

  "latency_p50_us": null,
  "latency_p99_us": null,
  "latency_max_us": null,
  "latency_criterion_met": null,

  "sram_bss_bytes": null,
  "sram_budget_bytes": 262144,
  "sram_criterion_met": null,

  "gap2_validated": null,
  "notes": ""
}
```

### Critères de validation

| Critère | Valeur cible | Mesure |
|---------|-------------|--------|
| RMSE board / RMSE PC | ≤ 1.10 (±10%) | `rmse_ratio_board_over_pc` |
| Latence P50 | ≤ 100 000 µs (100 ms) | `latency_p50_us` |
| SRAM .bss total | < 262 144 B (256 Ko) | `sram_bss_bytes` |

---

## S2612 — `exp_S26_02` : EWC Multi-class board / CWRU 3 tâches

### Objectif

Mesurer F1-macro board après entraînement séquentiel sur 3 tâches CWRU (task0=outer, task1=inner, task2=ball). Comparer au F1-macro PC de `exp_S25_03`.

### Commandes d'exécution

```bash
# Recompiler avec N_CLASSES=10
make -C firmware/stm32f4_blink all CFLAGS="-DEWC_MC_N_CLASSES=10"
make -C firmware/stm32f4_blink flash

python scripts/simulate_multiclass_board.py \
    --port /dev/ttyACM0 \
    --config configs/cwru_by_fault_config.yaml \
    --n-samples-per-task 100 \
    --n-classes 10 \
    --output experiments/exp_S26_02/board_mc_results.json \
    2>&1 | tee experiments/exp_S26_02/log.txt

arm-none-eabi-size firmware/stm32f4_blink/build/stm32f4_blink.elf \
    >> experiments/exp_S26_02/log.txt
```

### `config_snapshot.yaml`

```yaml
experiment_id: exp_S26_02
model: ewc_head_multiclass
dataset: CWRU bearing fault (3 tasks)
task: multi-class fault classification (10 classes)
board: NUCLEO-F439ZI
firmware: ewc_head_multiclass.c (EWC_MC_N_CLASSES=10) + pipeline.c (Sprint 26)
model_weights: experiments/exp_S25_03/model_ewc_mc.pt

firmware_config:
  EWC_MC_IN: 9
  EWC_MC_H1: 32
  EWC_MC_H2: 16
  EWC_MC_N_CLASSES: 10
  EWC_MC_LR: 0.01
  EWC_MC_FISHER_DECAY: 0.99
  EWC_LAMBDA: 400.0

n_samples_per_task: 100
n_tasks: 3
```

### `results.json` attendu

```json
{
  "experiment_id": "exp_S26_02",
  "model": "EWC Multiclass",
  "dataset": "CWRU 3 tasks",
  "board": "NUCLEO-F439ZI",
  "n_classes": 10,
  "n_tasks": 3,
  "n_samples": 300,

  "f1_macro_board": null,
  "f1_macro_pc": null,
  "f1_per_task_board": [null, null, null],

  "latency_p50_us": null,
  "latency_p99_us": null,
  "latency_criterion_met": null,

  "sram_bss_bytes": null,
  "sram_criterion_met": null,

  "gap1_validated": null,
  "gap2_validated": null,
  "notes": ""
}
```

### Critères de validation

| Critère | Valeur cible | Mesure |
|---------|-------------|--------|
| F1-macro board | ≥ 0.60 | `f1_macro_board` |
| Latence P50 | ≤ 100 000 µs | `latency_p50_us` |
| SRAM .bss total | < 262 144 B | `sram_bss_bytes` |

---

## S2613 — `exp_S26_03` : Comparaison RAM .bss des 3 têtes EWC

### Objectif

Mesurer le footprint SRAM .bss de chaque tête EWC compilée pour ARM, et comparer. Résultat attendu : regression ≈ fp32_binaire (même architecture MLP, 1 sortie au lieu de 2 → légèrement plus petit).

### Méthode

```bash
# Compiler 3 variantes séparées et logger la section .bss de EWCHead*
# Variante 1 : EWC FP32 binaire (existant)
arm-none-eabi-nm -S --size-sort firmware/stm32f4_blink/build/stm32f4_blink.elf \
    | grep -E "g_ewc_head|g_ewc_reg|g_ewc_mc|g_ewc_int8" \
    > experiments/exp_S26_03/bss_sizes.txt

arm-none-eabi-size firmware/stm32f4_blink/build/stm32f4_blink.elf \
    >> experiments/exp_S26_03/bss_sizes.txt
```

### `results.json` attendu

```json
{
  "experiment_id": "exp_S26_03",
  "description": "Comparaison footprint .bss des 3 têtes EWC sur NUCLEO-F439ZI",

  "ewc_head_fp32_binary": {
    "struct": "EWCHead",
    "bss_bytes": null,
    "comment": "EWC_IN=5, H1=32, H2=16, OUT=2 — baseline existant"
  },
  "ewc_head_regression": {
    "struct": "EWCHeadReg",
    "bss_bytes": null,
    "comment": "EWC_REG_IN=5, H1=32, H2=16, OUT=1 — Sprint 26"
  },
  "ewc_head_multiclass_n10": {
    "struct": "EWCHeadMC",
    "bss_bytes": null,
    "comment": "EWC_MC_IN=9, H1=32, H2=16, N_CLASSES=10 — Sprint 26"
  },
  "ewc_head_int8_binary": {
    "struct": "EWCHeadInt8",
    "bss_bytes": null,
    "comment": "INT8 quantifié — Sprint 22"
  },

  "total_firmware_bss_bytes": null,
  "budget_bytes": 262144,
  "margin_bytes": null
}
```

### Résultats attendus (estimés)

| Tête | SRAM .bss estimé | Commentaire |
|------|-----------------|-------------|
| `EWCHead` (binaire) | ~9 500 B | IN=5, OUT=2, existant |
| `EWCHeadReg` (reg) | ~8 884 B | IN=5, OUT=1, légèrement plus petit |
| `EWCHeadMC` (N=10) | ~14 072 B | IN=9, OUT=10, plus grand |
| `EWCHeadInt8` | ~2 400 B | INT8 poids + FP32 biais |

---

## Résultats d'implémentation

> **Mise à jour 2026-06-12 — run board réel + diagnostic multi-classe.** Le bug de framing de
> réponse v3 (23 B, champ `ram_b` u16 manquant) a été corrigé. Le F1 multi-classe = **0.507** est
> désormais confirmé GENUINE et entièrement diagnostiqué (voir « Diagnostic FIXME(gap1) » ci-dessous).

| Expérience | Statut | RMSE / F1 board | Latence P50 µs | SRAM .bss Ko |
|-----------|:------:|----------------|---------------|-------------|
| exp_S26_01 (EWC RUL board) | ✅ | RMSE = **21.23** (PC FD001 = 22.53, ratio = 0.94 ✅) | **233 µs** ✅ | total 65.2 Ko ✅ |
| exp_S26_02 (EWC multi-class board) | ⚠️ critère / ✅ diagnostiqué | F1 = **0.507** online / **0.243** inférence pure < 0.60 ❌ | infér. **130 µs** / online **403 µs** ✅ | total 65.2 Ko ✅ |
| exp_S26_03 (RAM profiling) | ✅ | — | — | Total : **65.2 Ko** / 256 Ko ✅ |

### Diagnostic FIXME(gap1) — exp_S26_02 (RÉSOLU)

Le critère F1-macro ≥ 0.60 **n'est pas atteint** (0.507 online / 0.243 inférence pure). La cause a
été identifiée et **ce n'est PAS un bug de portage board** :

- **Parité board ↔ PC exacte.** Une réimplémentation numpy fidèle du forward+SGD C
  (`scripts/diagnose_multiclass_parity.py`, chargeant les poids `.pt`) donne des logits identiques à
  PyTorch (max|diff| ≈ 5e-7). F1 board == F1 PC à l'identique : **inférence 0.243 == 0.243**,
  **online single-pass 0.507 == 0.507** (200 samples/tâche). Le board reproduit donc exactement le
  modèle PC — aucun désalignement label→index, aucune erreur de standardisation.
- **Cause réelle : oubli catastrophique.** Le modèle final exporté (`exp_S25_03/model_ewc_mc.pt`) ne
  retient que les classes de la **dernière tâche** {0,7,8,9}. F1 val du modèle FINAL par tâche :
  task0 = 0.105, task1 = 0.167, task2 = 0.989 → F1-macro tous-tâches concaténées = **0.240**.
- **Le comparateur « PC F1 = 0.981 » était trompeur** : c'est la moyenne des F1 mesurées *juste après*
  chaque tâche (`train_ewc_multiclass.py:195`, NaN pour classes non vues), qui masque l'oubli.
  `avg_forgetting_f1 = 0.847` (déjà dans `exp_S25_03/results.json`) le signalait.
- **Métrique board** : F1-macro préquentiel cumulatif sur 10 classes dans l'ordre du flux. L'online
  single-pass (`FLAG_UPDATE`) ré-apprend partiellement task0/task1 → 0.507 > inférence pure 0.243.

**Conclusion honnête** : le board valide le portage (parité numérique + Gap 2 : latence inférence
130 µs vs inférence+update 403 µs, séparées, << 100 ms ; SRAM 65.2 Ko). Le critère F1 ≥ 0.60 reste
non atteint à cause de l'oubli catastrophique du modèle EWC (λ=400) sur 3 tâches / 10 classes CWRU —
limitation du modèle CL, pas du portage. Amélioration (λ plus élevé, replay, ou rapport du F1 par
tâche) à traiter côté entraînement (hors périmètre portage Sprint 26).

> **Note board stateful** : `g_ewc_mc` est chargé une seule fois au boot (`pipeline_init`) ; les
> updates SGD persistent entre trames **et entre invocations de script**. Re-flasher (ou reset HW)
> avant chaque mesure pour repartir des poids figés pristine — sinon une mesure inférence pure lit
> des poids déjà modifiés par un run online précédent.

---

## Questions ouvertes

- `TODO(arnaud)` : Pour `exp_S26_01`, le scénario est 1 tâche (FD001). Faut-il aussi mesurer 4 tâches CL (FD001→FD002→FD003→FD004) pour la contribution Gap 1 ("données industrielles") ? Si oui, ajouter `exp_S26_04` en extension de sprint.
- `TODO(dorra)` : Si la latence `ewc_mc_sgd_step` (N=10 sorties, stack 464 B) dépasse 100 ms sur board, réduire N_CLASSES à 3 (Paderborn) et relancer avec `CFLAGS=-DEWC_MC_N_CLASSES=3`. Créer `exp_S26_02b` le cas échéant.
- `FIXME(gap2)` ✅ **RÉSOLU (2026-06-12)** pour la multi-classe : latences DWT séparées via
  `simulate_multiclass_board.py --no-update` (inférence seule) vs run normal (forward+SGD+consolidation).
  Mesuré board : **inférence pure P50 = 130 µs** vs **inférence+update P50 = 403 µs** — toutes deux
  << 100 ms. (À reproduire pour RUL/régression si besoin manuscrit.)
- `TODO(fred)` : Ajouter les résultats `exp_S26_03` au benchmark Edge Spectrum (tableau comparatif têtes EWC reg vs fp32 vs int8 en SRAM).
- `FIXME(gap1)` ✅ **RÉSOLU (2026-06-12)** : exp_S26_02 — F1-macro board = 0.507 online / 0.243
  inférence pure < 0.60. Diagnostic complet via `scripts/diagnose_multiclass_parity.py` et un run
  board inférence pure (`simulate_multiclass_board.py --no-update`). **Ce n'est pas un bug de mapping
  ni de portage** : parité numérique board ↔ PC exacte (0.243==0.243, 0.507==0.507). La cause est
  l'**oubli catastrophique** du modèle EWC multi-classe (le modèle final ne retient que les classes
  de la dernière tâche ; F1 val tous-tâches = 0.240). Le « PC F1 = 0.981 » était la moyenne des F1
  post-tâche (`train_ewc_multiclass.py:195`), trompeuse (`avg_forgetting_f1 = 0.847`). Amélioration
  du modèle CL (λ, replay) à traiter côté entraînement, hors périmètre portage Sprint 26.
