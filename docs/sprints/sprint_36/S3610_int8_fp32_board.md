# S3610 — Axe INT8 vs FP32 sur board (EWC, frozen + online)

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 36 (rework) |
| **Tâches liées** | S3610 (firmware + driver), S3611 (runs board), S3612 (agrégation + notebook), S3613 (tests + docs) |
| **Statut** | ✅ Chaîne implémentée · ✅ **8 cellules mesurées board réelle NUCLEO-F439ZI (0 CRC)** |
| **Modèle / datasets / conditions** | EWC · Pronostia (D4) + Monitoring (D2) · `5feat` + `all` |
| **Protocoles** | `frozen` (inférence) **et** `online` (inférence + MAJ CL) |
| **Portée** | **board-only** : la référence PC reste FP32 (pas de `EWCMlpInt8Classifier` PC) |

---

## Motivation (Gap 3)

Le triple gap (`docs/triple_gap.md`) vise la **quantification INT8 pendant l'entraînement
incrémental**. Les Sprints 28 (PC) et 29 (board) ont mesuré EWC/HDC/TinyOL/Maha INT8 en
balayage large. Ce rework apporte la **comparaison appariée et focalisée INT8 vs FP32 du modèle
EWC** sur les deux datasets et conditions du Sprint 36, dans les **mêmes conditions exactes**
que la comparaison FP32 board↔PC — afin de chiffrer honnêtement le coût/bénéfice de l'INT8 :
RAM des poids (÷4 structurel), latence (FPU Cortex-M4 : pas d'accélération INT8, cf. Sprint 29),
métrique préservée, et **accord INT8↔FP32 board**.

## Chaîne firmware (résolution `TODO(dorra)`)

Avant : `pipeline.c` initialisait `g_ewc_int8` via `ewc_int8_init` mais le commentaire
`TODO(dorra): ewc_int8_from_fp32(...)` n'était jamais exécuté → le chemin `0x40`
(`PROTO_FLAG_INT8_MODE`) tournait sur une tête **Xavier non entraînée**.

Après (S3610) : juste après `ewc_head_load_or_init(&g_ewc_head)`,

```c
ewc_int8_from_fp32(&g_ewc_int8, &g_ewc_head);
```

⇒ la tête INT8 reflète exactement les poids FP32 chargés (poids exportés si
`EWC_HEAD_WEIGHTS_PROVIDED`, Xavier en fallback). **Le chemin FP32 est inchangé → 0 régression**
(`make test` : 116 tests, seuls les 2 échecs TinyOL préexistants subsistent, hors périmètre).

Le reste de la chaîne existait déjà :

- `sensor_stream.py --model ewc-int8` → `FRAME_FLAGS_INT8_MODE = 0x40` ;
- `pipeline.c` route `0x40` vers `ewc_int8_forward` / `ewc_int8_update` / `ewc_int8_consolidate` ;
- **pas de nouveau flag protocole, aucune collision** (le nibble de mode n'est pas touché).

Le **build/flash est identique** entre FP32 et INT8 (mêmes poids FP32 exportés via
`export_weights_c.py --ewc-head`, conversion INT8 au boot) — seul le flag UART diffère.

## Driver (`run_sprint36_board.py --precision {fp32,int8}`)

- Défaut `fp32` (rétro-compatible). En `int8`, on streame avec `FRAME_FLAGS_INT8_MODE`.
- Sorties : `exp_S36_board_{frozen,online}_int8_{cond}_ewc_{ds}/results.json`.
- Champs INT8 ajoutés (modèle Sprint 28/29) : `precision`, `metric_value` (F1),
  `ram_weights_fp32_bytes`, `ram_weights_int8_bytes`, `ram_ratio_fp32_over_int8`,
  `latency_us_p50/p99`, `crc_errors`, et **`agreement_int8_vs_fp32`**.

## Parité / accord (pas de parité exacte PC)

L'INT8 quantifie → **pas de parité bit-à-bit avec le PC FP32**. On mesure donc un **accord
INT8↔FP32 board** :

- **frozen** : préds board INT8 vs préds FP32 de référence (`_pc_pred_ewc` sur le même
  checkpoint = préds board FP32, par parité exacte frozen) ;
- **online** : préds board INT8 vs préds board FP32 persistées dans le `board_samples.json`
  de la passe online FP32 (mêmes échantillons, même ordre).

## RAM des poids (Gap 3)

Compte analytique des 3 couches `k→32→16→2` : `n_w = 32·k + 32·16 + 16·2`. FP32 = 4 B/poids,
INT8 = 1 B/poids ⇒ **ratio structurel 4.0**. Distinct de `.bss` (qui héberge les **deux** têtes
simultanément côté firmware). `gap3_ram_ok = (ratio ≥ 3.5)`.

## Agrégation & notebook

- `aggregate_sprint36.py` : clés additives `board_frozen_int8` / `board_online_int8` sous
  `results[dataset][condition]`, avec `latency_ratio_int8_over_fp32`,
  `delta_metric_int8_vs_fp32`, `gap3_ram_ok`, `agreement_int8_vs_fp32`. **Summary
  rétro-compatible** (champs `null` tant que l'INT8 n'a pas été streamé).
- Notebook `comparison.ipynb` : §12 « INT8 vs FP32 sur board — Gap 3 » (latence frozen+online
  log + ligne Gap 2 ; RAM poids + ratio ; F1 préservée + accord). Repère « à mesurer » si
  aucune donnée INT8.

## Résultats mesurés (board réelle NUCLEO-F439ZI, 8 cellules, 0 CRC)

| Cellule | Passe | Lat P50 | RAM ratio | F1 INT8 | F1 FP32 (réf) | Accord INT8↔FP32 |
|---------|-------|--------:|----------:|--------:|--------------:|-----------------:|
| pronostia `5feat` | frozen | 53 µs | ×4.0 | 0.138 | ~0.916 | 0.736 |
| pronostia `all` (k=13) | frozen | 68 µs | ×4.0 | 0.150 | ~0.918 | 0.681 |
| monitoring `5feat`/`all` (k=4) | frozen | 51 µs | ×4.0 | 0.134 | ~0.919 | 0.595 |
| pronostia `5feat` | online | 462 µs | ×4.0 | 0.085 | — | 0.867 |
| pronostia `all` | online | 639 µs | ×4.0 | 0.108 | — | 0.854 |
| monitoring `5feat`/`all` | online | 440 µs | ×4.0 | 0.068 | — | 0.875 |

**Lecture honnête des trois gaps :**

- **Gap 2 ✅** : toutes les latences ≪ 100 ms (frozen 51–68 µs ≈ FP32 ; online 440–639 µs).
  L'inférence INT8 ≈ FP32, mais la **MAJ online INT8 est ~2× plus lente** que FP32 (239–340 µs)
  — cohérent avec le résultat clé Cortex-M4 FPU (l'INT8 scalaire n'est pas accéléré, Sprint 29).
- **Gap 3 RAM ✅** : poids ÷4.0 (`gap3_ram_ok = True`) pour les 8 cellules.
- **Métrique NON préservée ❌** : F1 INT8 **0.07–0.15** s'effondre vs FP32 board ≈ 0.92 ;
  l'accord INT8↔FP32 n'est que **0.60–0.74** (frozen). C'est une **forte dégradation de la
  quantification post-training du firmware** (`ewc_int8_from_fp32` : poids INT8 + activations,
  sans QAT). Ce résultat est **cohérent avec le Sprint 29** (board INT8 EWC AUROC 0.25 vs FP32
  0.63) et **distinct du fake-quant QAT PC** (Sprint 28 : Δmétrique ≤ 0.006, métrique préservée).
  La quantification PTQ embarquée de la tête EWC binaire est donc **insuffisante** ; pistes :
  QAT exporté vers le firmware, ou quantification Q15 (cf. Mahalanobis Sprint 34).

> **Note** : le `TODO(dorra)` résolu (chargement INT8 ← FP32) **améliore** vs l'ancien comportement
> (tête Xavier non entraînée, accord ≈ 0.5) mais ne suffit pas à préserver la métrique — la perte
> vient de la quantification elle-même, pas de l'initialisation.

### Règle « aucun chiffre inventé »

Les chiffres ci-dessus proviennent **exclusivement** de l'exécution réelle du driver sur la
NUCLEO-F439ZI (`exp_S36_board_*_int8_*`, 0 CRC). Avant cette exécution, l'agrégat et le notebook
affichaient `null` / « à mesurer » (les fichiers ne sont produits que par un run board réel).

## Vérification

```bash
# Firmware (0 régression FP32)
make -C firmware/stm32f4_blink test

# Board (NUCLEO branchée) — pour (cond ∈ {5feat,all}) × (ds ∈ {pronostia,monitoring})
python scripts/run_sprint36_board.py --pass frozen --precision int8 --condition <c> --dataset <d>
python scripts/run_sprint36_board.py --pass online --precision int8 --condition <c> --dataset <d>

# Agrégat + notebook + tests
python scripts/aggregate_sprint36.py
jupyter nbconvert --to notebook --execute notebooks/cl_eval/pc_board_ewc/comparison.ipynb
pytest tests/test_sprint36_comparison.py -v
```
