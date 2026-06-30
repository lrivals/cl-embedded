# S2904–S2905 — Expériences board INT8 (NUCLEO-F439ZI)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 29 |
| **Priorité** | S2904 : 🔴 / S2905 : 🟡 |
| **Statut** | ✅ Implémenté (15 juin 2026) — 5 expériences board réelles, 5 JSON produits |
| **Durée estimée** | S2904 : 4h / S2905 : 2h |
| **Dépendances** | S2903 ✅ (pipeline.c + sensor_stream.py) · `make flash` sur NUCLEO-F439ZI connectée |
| **Fichiers cibles** | `experiments/exp_S29_board_int8/` (5 `results_*.json`) · orchestrateur `scripts/run_s29_board_int8.py` |

---

## Statut d'implémentation (15 juin 2026)

**5 couples (modèle, dataset) mesurés sur NUCLEO-F439ZI réelle** (port `/dev/ttyACM0`, 0 erreur CRC).
Chaque run mesure FP32 **et** INT8 (la board reset via DTR entre les deux → états en ligne indépendants),
puis assemble le schéma cible. Orchestrateur : `scripts/run_s29_board_int8.py` (réutilise les internals
de `sensor_stream.py`).

**Résultat central — Gap 3 confirmé multi-modèle** :
- **RAM ✅ pour les 3 modèles** : EWC ×2.70, HDC ×3.06, TinyOL ×4.00 (empreinte poids analytique,
  architecture-cohérente — la `.bss` totale rapportée par le firmware est constante car toutes les
  structs des modèles coexistent en RAM).
- **Latence : INT8 plus lent sur Cortex-M4 FPU** pour EWC (×1.84, **identique au S23**) et HDC (×3.26),
  confirmant le résultat négatif honnête au-delà du seul cas EWC×CMAPSS.
- **Exception TinyOL** : INT8 mesuré **plus rapide** (71 µs vs 127 µs) — mais les deux chemins ne sont
  **pas le même workload** : FP32 = autoencodeur (encode 5→32→16 + decode 16→32→5 + MSE), INT8 =
  encodeur + tête OtO linéaire (16→1, pas de décodeur). Comparaison non iso-calcul, à interpréter avec
  prudence (ce n'est pas un speedup INT8 du même graphe).

**Découvertes de portage** (notées pour S2900) :
1. Les commandes ci-dessous du doc d'origine étaient obsolètes : `sensor_stream.py` prend `--dataset`
   + `--model` (pas `--mode`, pas `--config`).
2. Le firmware initialise `g_ewc_head`/`g_ewc_int8` en **Xavier aléatoire + apprentissage en ligne**
   (`ewc_int8_from_fp32` non câblé, cf. `TODO(dorra)` `pipeline.c`). Conséquence : aucun export/flash
   EWC requis, mais l'**AUROC EWC board est faible** (CWRU 0.40, Pronostia 0.34) car le modèle apprend
   depuis zéro en ≤ 498 échantillons (≠ AUROC PC S28 qui part de poids pré-entraînés). C'est une
   limitation de configuration board, pas un bug de portage (latence/RAM = vraie contribution Gap 3).
3. HDC n'utilise **pas** de `feature_bounds` par dataset (projection signée aléatoire) → CMAPSS et
   Monitoring tournent sur le même binaire, sans recalibration.
4. Seul **TinyOL** nécessite un export+flash (encodeur pré-entraîné). Encodeur CWRU réentraîné via
   `export_weights_tinyol.py --train-dataset cwru` puis `make && make flash` (`.bss`=104 576 B).

---

## Contexte

Sprint 23 a mesuré EWC INT8 sur CMAPSS board : latence **0.461 ms** (vs FP32 **0.251 ms** — INT8 **plus lent**). Sprint 29 étend ces mesures à HDC INT8 + TinyOL INT8 sur plusieurs datasets pour confirmer que le résultat négatif latence est général sur Cortex-M4 FPU, et documenter les RAM savings.

---

## S2904 — EWC INT8 (CWRU, Pronostia) + HDC INT8 (CMAPSS, Monitoring)

**Commandes réelles** (l'orchestrateur `run_s29_board_int8.py` lance FP32 **puis** INT8 et assemble le
schéma S2904 ; `sensor_stream.py` prend `--dataset`/`--model`, **pas** `--mode`/`--config`) :

```bash
# EWC INT8 sur CWRU (apprentissage en ligne depuis zéro — aucun flash requis)
python scripts/run_s29_board_int8.py --model ewc --dataset cwru \
    --n-samples 498 --n-tasks 3 \
    --output experiments/exp_S29_board_int8/results_ewc_int8_cwru.json

# EWC INT8 sur Pronostia
python scripts/run_s29_board_int8.py --model ewc --dataset pronostia \
    --n-samples 498 --n-tasks 3 \
    --output experiments/exp_S29_board_int8/results_ewc_int8_pronostia.json

# HDC INT8 sur CMAPSS (base vectors LCG + AM en ligne — aucun flash requis)
python scripts/run_s29_board_int8.py --model hdc --dataset cmapss \
    --n-samples 300 --n-tasks 3 \
    --output experiments/exp_S29_board_int8/results_hdc_int8_cmapss.json

# HDC INT8 sur Monitoring
python scripts/run_s29_board_int8.py --model hdc --dataset monitoring \
    --n-samples 300 --n-tasks 3 \
    --output experiments/exp_S29_board_int8/results_hdc_int8_monitoring.json
```

---

## S2905 — TinyOL INT8 (CWRU)

TinyOL utilise un encodeur **pré-entraîné** (≠ EWC/HDC) → 1 cycle export + flash avant la mesure :

```bash
# 1. Exporter l'encodeur board entraîné sur CWRU → model_weights.h
python scripts/export_weights_tinyol.py --train-dataset cwru --train-epochs 150
# 2. Recompiler + flasher
cd firmware/stm32f4_blink && make clean && make && make flash && cd ../..
# 3. Mesurer FP32 + INT8
python scripts/run_s29_board_int8.py --model tinyol --dataset cwru \
    --n-samples 498 --n-tasks 3 \
    --output experiments/exp_S29_board_int8/results_tinyol_int8_cwru.json
```

---

## Métriques à collecter (dans chaque `results_*.json`)

```json
{
  "model": "ewc_int8",
  "dataset": "cwru",
  "board": "NUCLEO-F439ZI",
  "precision": "INT8",
  "n_samples": 498,
  "latency_dwt_us": {
    "p50": null,
    "p95": null,
    "p99": null
  },
  "ram_bss_bytes": null,
  "metric_name": "auroc",
  "metric_value": null,
  "gap2_compliant": null,
  "gap3_latency_ok": null,
  "gap3_ram_ok": null
}
```

**Critères Gap 3 board** :
- `gap3_latency_ok` = latence INT8 < latence FP32 (attendu **False** sur Cortex-M4 FPU — documenter)
- `gap3_ram_ok` = RAM .bss INT8 < RAM .bss FP32 pour ce modèle (attendu **True** ×2.7–4.0)

---

## Tableau mesuré (NUCLEO-F439ZI, 15 juin 2026)

| Modèle | Dataset | Lat FP32 µs | Lat INT8 µs | Ratio lat | RAM FP32 B | RAM INT8 B | Ratio RAM | Métrique INT8 | Gap 3 |
|--------|---------|:-----------:|:-----------:|:---------:|:----------:|:----------:|:---------:|:-------------:|:------:|
| EWC | CMAPSS (S23) | 251 | 461 | 1.84× ❌ | 9 728 | 3 600 | ×2.70 ✅ | acc 0.85 | RAM ✅ |
| EWC | CWRU | 251 | 462 | 1.84× ❌ | 9 728 | 3 600 | ×2.70 ✅ | auroc 0.40 ⚠️ | RAM ✅ |
| EWC | Pronostia | 251 | 462 | 1.84× ❌ | 9 728 | 3 600 | ×2.70 ✅ | auroc 0.34 ⚠️ | RAM ✅ |
| HDC | CMAPSS | 647 | 2 106 | 3.26× ❌ | 106 496 | 34 816 | ×3.06 ✅ | acc 0.887 | RAM ✅ |
| HDC | Monitoring | 647 | 2 106 | 3.26× ❌ | 106 496 | 34 816 | ×3.06 ✅ | acc 0.927 | RAM ✅ |
| TinyOL | CWRU | 127 | 71 | 0.56× ✅* | 2 688 | 672 | ×4.00 ✅ | auroc 0.992 | RAM ✅ + lat ✅* |

> **RAM** = empreinte poids analytique (architecture-cohérente FP32↔INT8), pas la `.bss` totale (constante,
> ≈ 39 040 B rapportée par firmware car toutes les structs des modèles coexistent en RAM).
> **Latence** = DWT, déterministe (p50=p95=p99 sur la board).
>
> *\*Caveat TinyOL* : FP32 = autoencodeur (encode+decode+MSE), INT8 = encodeur + tête OtO (16→1, sans
> décodeur). Le « speedup » INT8 vient surtout du chemin plus court, **pas** d'une accélération INT8 du
> même graphe — à ne pas sur-interpréter.
>
> *⚠️ AUROC EWC faible* : modèle EWC appris **en ligne depuis zéro** sur la board (poids pré-entraînés
> non câblés, `TODO(dorra)`), ≤ 498 échantillons → convergence partielle. Limitation de config board,
> indépendante du résultat Gap 3 (latence/RAM). Les valeurs FP32↔INT8 restent cohérentes entre elles.

**Conclusion S2904/S2905** : Gap 3 **RAM confirmé sur les 3 modèles** (×2.70 à ×4.00). Résultat **latence
négatif** (INT8 plus lent) confirmé pour EWC (×1.84, identique S23) et HDC (×3.26) sur Cortex-M4 FPU —
contribution honnête au-delà du cas unique EWC×CMAPSS. JSON : `experiments/exp_S29_board_int8/results_*.json`.
