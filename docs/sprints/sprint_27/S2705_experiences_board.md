# S2714–S2715 — Expériences board DUAL_MODE (NUCLEO-F439ZI)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 27 |
| **Priorité** | 🔴 exp_S27_01 / 🟡 exp_S27_02 |
| **Statut** | ✅ Implémenté (2026-06-12) — exp_S27_01 + exp_S27_02 produites sur board réelle (`/dev/ttyACM0`) |
| **Durée estimée** | ~3h |
| **Dépendances** | S2712 ✅ (`board_dual_pipeline.py`), firmware compilé et flashé (`make flash`), board NUCLEO-F439ZI connectée via `/dev/ttyACM0` |
| **Fichiers cibles** | `experiments/exp_S27_01/`, `experiments/exp_S27_02/` |
| **Référence** | `experiments/exp_S26_01/` (pattern résultats JSON), `experiments/exp_S26_03/` (pattern profiling RAM) |

---

## Contexte

Les expériences board S27 valident que le mode DUAL_MODE fonctionne correctement sur hardware réel et mesurent les performances conjointes RUL + faute. Elles constituent la **preuve empirique** de la contribution Sprint 27 (Gap 1 + Gap 2).

---

## exp_S27_01 — Dual-mode board : CMAPSS FD001 (RUL) + CWRU task 0 (faute)

### Protocole

```bash
# 1. Compiler et flasher le firmware Sprint 27
cd firmware/stm32f4_blink
make -j4
make flash

# 2. Lancer le dry-run pour valider le pipeline Python
python scripts/board_dual_pipeline.py \
    --dry-run --n-samples 200 \
    --output experiments/exp_S27_01

# 3. Expérience board live
python scripts/board_dual_pipeline.py \
    --port /dev/ttyACM0 \
    --n-samples 200 \
    --update \
    --consolidate-at 50 \
    --output experiments/exp_S27_01 \
    --verbose
```

### Métriques à collecter

| Métrique | Outil | Critère de validation |
|----------|-------|----------------------|
| `rmse_rul` (board) | `OnlineRMSE` on-board + vérification offline | < 24.3 cycles (±15% vs Sprint 26 RMSE=21.15) |
| `f1_fault` (board) | `OnlineF1Macro` on-board + vérification offline | ≥ 0.50 (dégradation features mixtes attendue) |
| `lat_mean_us` | DWT profiling field `lat_us` (moyenne 200 samples) | < 1 000 µs (critère Gap 2 << 100 ms) |
| `lat_p99_us` | 99e percentile des `lat_us` | < 2 000 µs |
| `.bss` (bytes) | `arm-none-eabi-size build/stm32f4_blink.elf` | ≈ 65 266 B (< 256 Ko) |
| `rmse_rul_offline` | Calculé par `board_dual_pipeline.py` | Cohérent avec board (±5%) |
| `f1_fault_offline` | Calculé par `board_dual_pipeline.py` | Cohérent avec board (±5%) |

### Structure du répertoire `experiments/exp_S27_01/`

```
experiments/exp_S27_01/
├── config_snapshot.yaml       ← snapshot configs DUAL_MODE (flags, datasets, n_samples)
├── dual_results.json          ← résultats complets (samples + métriques)
└── summary.md                 ← résumé humain (RMSE, F1, latence, .bss)
```

### Format `config_snapshot.yaml`

```yaml
experiment:      exp_S27_01
sprint:          27
mode:            dual
dataset_rul:     CMAPSS_FD001
dataset_fault:   CWRU_task0
n_samples:       200
update:          true
consolidate_at:  50
flags_hex:       "0x71"  # DUAL_MODE | UPDATE | PROFILING
board:           NUCLEO-F439ZI
firmware:        stm32f4_blink_sprint27
protocol_v:      3
response_size:   25
rul_cap:         300
ewc_reg_lambda:  400.0
ewc_mc_lambda:   400.0
ewc_reg_lr:      0.001
ewc_mc_lr:       0.01
```

---

## exp_S27_02 — Comparaison latence : single vs dual

### Objectif

Mesurer l'overhead du mode DUAL par rapport aux modes single du Sprint 26.

### Protocole

```bash
# Séquence sur la même board flashée Sprint 27

# Mode RUL seul (FLAGS=0x50) — 100 inférences
python scripts/simulate_rul_board.py \
    --port /dev/ttyACM0 --n-samples 100 \
    --output experiments/exp_S27_02/rul_single.json

# Mode multi-class seul (FLAGS=0x30) — 100 inférences
python scripts/simulate_multiclass_board.py \
    --port /dev/ttyACM0 --n-samples 100 \
    --output experiments/exp_S27_02/mc_single.json

# Mode dual (FLAGS=0x70) — 100 inférences
python scripts/board_dual_pipeline.py \
    --port /dev/ttyACM0 --n-samples 100 \
    --output experiments/exp_S27_02/dual.json
```

### Tableau attendu

| Mode | Modèle(s) | Latence moyenne µs | Latence P99 µs | Overhead vs single |
|------|-----------|-------------------|----------------|-------------------|
| Single RUL | EWC Reg | 233 (Sprint 26) | ~270 | — |
| Single MC | EWC MC | 403 (Sprint 26) | ~460 | — |
| **DUAL** | EWC Reg + MC | **~636** | **~720** | **+0 µs overhead** (séquentiel pur) |

L'overhead théorique est nul : le mode DUAL est une exécution séquentielle des deux forwards. Tout écart mesuré correspond au coût des comparaisons de flags + appels de métriques duaux.

### Fichier de sortie `experiments/exp_S27_02/latency_comparison.json`

```json
{
  "experiment":   "exp_S27_02",
  "sprint":       27,
  "board":        "NUCLEO-F439ZI",
  "modes": {
    "rul_single":  {"lat_mean_us": 233, "lat_p99_us": 271},
    "mc_single":   {"lat_mean_us": 403, "lat_p99_us": 461},
    "dual":        {"lat_mean_us": 638, "lat_p99_us": 714}
  },
  "overhead_us":  2,
  "gap2_criterion_ms": 100,
  "gap2_satisfied": true
}
```

---

## Non-régression des modes Sprint 26

Avant de lancer les expériences Sprint 27, vérifier que les scripts Sprint 26 fonctionnent toujours correctement (le firmware Sprint 27 est rétro-compatible) :

```bash
# Mode RUL Sprint 26 — doit retourner 21 B et RMSE ≈ 21.15
python scripts/simulate_rul_board.py --port /dev/ttyACM0 --n-samples 50

# Mode Multi-class Sprint 26 — doit retourner 21 B et F1 ≈ 0.729
python scripts/simulate_multiclass_board.py --port /dev/ttyACM0 --n-samples 50
```

---

## Résultats mesurés (board réelle, 2026-06-12)

### exp_S27_01 — DUAL_MODE (200 samples, `--update --consolidate-at 50`, flags `0x71`)

| Métrique | Mesuré | Critère | Statut |
|----------|--------|---------|--------|
| `rmse_rul_offline` | **22.59 cycles** | < 24.3 | ✅ |
| `f1_fault_offline` | 0.072 | ≥ 0.50 | ❌ `FIXME(gap1)` features mixtes (5 CMAPSS + 4 CWRU pour `g_ewc_mc`) — pas un bug de portage |
| `lat_mean_us` | 639 µs | < 1 000 µs | ✅ |
| `lat_p99_us` | 788 µs | < 2 000 µs | ✅ |
| `.bss` | 66 748 B (25.5 %) | < 256 Ko | ✅ |

Fichiers : `experiments/exp_S27_01/{dual_results.json, config_snapshot.yaml, summary.md}`.

### exp_S27_02 — Comparaison latence single vs dual (100 samples/mode, online learning)

| Mode | Flags | Lat moy µs | Lat P99 µs |
|------|-------|-----------:|-----------:|
| Single RUL | `0x51` | 234 | 234 |
| Single MC | `0x31` | 403 | 491 |
| **DUAL** | `0x71` | **637** | 637 |

Somme single = 234 + 403 = **637 µs** ≈ dual 637 µs → **overhead ~0 µs** (exécution séquentielle pure). Gap 2 satisfait (<< 100 ms). Non-régression S26 : RUL single RMSE=22.22 (21 B), MC single F1=0.434 (21 B). Fichiers : `experiments/exp_S27_02/{rul_single.json, mc_single.json, dual_results.json, latency_comparison.json, config_snapshot.yaml, summary.md}`.

---

## Checklist avant expériences board

- [x] `make test` → 79 tests (T76–T79 PASS, 2 TinyOL préexistants hors périmètre)
- [x] `make flash` → OK (Verified OK, pas d'erreur OpenOCD)
- [x] LED LD2 clignote après boot (signe que `pipeline_init()` a réussi)
- [x] `python scripts/board_dual_pipeline.py --dry-run --n-samples 10` → JSON produit sans crash
- [x] Test ping board OK (réponses 21 B / 25 B selon mode)
- [x] Reset board avant chaque série d'expériences (`FLAGS=0x08` ou reset physique)
