# Sprint 48 — Portage board sub-INT8 pour EWC (RAM `.bss` réelle, latence, parité)

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 48 |
| **Semaine** | 30 juillet – 5 août 2026 |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Priorité globale** | 🔴 Critique — **matérialise sur NUCLEO-F439ZI** les schémas gagnants du Sprint 47 : mesure la **RAM `.bss` réelle** (bit-packée, ÷8/÷16 attendus) et la **latence DWT** que l'émulateur ne mesure pas, et **valide la parité board↔PC**. Répond aux deux `TODO(dorra)` du Sprint 47 (kernel bit-packé, coût du dépacking). |
| **Durée estimée totale** | ~28h (sélection/cadrage ~2h · kernels sub-INT8 firmware ~9h · export+test vectors ~5h · driver board+mesures ~6h · parité+agrégation ~3h · notebook+figures ~2h · tests+docs ~1h) |
| **Dépendances** | **Sprint 47** ✅ (configs gagnantes, émulateur = source unique du schéma) · Sprint 39 ✅ (kernel `ewc_head_int8_v2.c` + variantes de build + export `--int8-v2`) · Sprint 29/36 ✅ (pipeline board, parité, DWT) |

## Contexte et motivation

Le Sprint 47 (PC/émulateur) identifie **jusqu'où descendre en bits** et **avec quel schéma** l'AUROC de la tête
EWC est préservée — mais l'émulateur ne mesure que la **métrique** et une **RAM théorique**. Deux questions
restent ouvertes, mesurables **uniquement sur carte** :

1. **Le gain RAM sub-INT8 est-il réel ?** Un INT4 stocké dans un `int8_t` n'économise rien de plus que l'INT8 ;
   le ÷8 (INT4) / ÷16 (INT2) exige un **kernel bit-packé** (2/4 poids par octet). Ce sprint implémente le packing
   et mesure le **`.bss` réel** avec et sans.
2. **Quel est le coût latence du dépacking ?** Le MAC reste FPU (paradoxe latence FPU, Sprint 29) ; le dépacking
   ajoute des opérations entières. Mesure DWT P50/P99 (Gap 2).

Ce sprint est rédigé en **supposant la NUCLEO-F439ZI disponible** (décision utilisateur) : les cellules portent
des **mesures réelles attendues** (latence DWT, `.bss`, parité, 0 CRC), à exécuter dès accès carte. Aucun chiffre
n'est écrit avant exécution (`pending`).

Périmètre : **EWC** (mêmes datasets que S47 — Monitoring, Pronostia), schémas gagnants sélectionnés en S4708.

## Décisions de cadrage (héritées S47 + board)

- **Sélection par compilation** `-DEWC_INT4` / `-DEWC_INT2` (+ variante bit-packée), dans la lignée de
  `-DEWC_INT8_Q15` / `-DEWC_INT8_MIXED` (S39) et `-DMAHA_INT8` (S2912) — le **nibble de flags UART est saturé**,
  la sélection de profondeur passe donc par le build, pas par le protocole. **Wire format V3 (23 B) inchangé**,
  `sensor_stream.py` intact.
- **Kernel sur le squelette `ewc_head_int8_v2.c`** : typedef `ewc_v2_w_t` + `EWC_V2_W_QMAX` déjà génériques →
  QMAX 7 (INT4) / 3 (INT2), + chemin de **dépacking** optionnel.
- **Parité board↔PC par construction** : l'export réutilise les primitives de l'émulateur étendu S47 (comme
  `--int8-v2` réutilise `_weight_scales`/`_quant_weight`).
- **Aucun chiffre inventé** : `pending` tant que la carte n'a pas streamé ; N/A honnête si un schéma déborde la SRAM.

## Nœud honnête : RAM théorique (S47) vs `.bss` mesurée (S48)

Le Sprint 47 reporte une RAM **théorique** (÷8 à INT4). Ce sprint mesure le **`.bss` réel** dans **deux builds
par schéma** :

- **non-packé** (INT4/INT2 dans conteneurs `int8_t`) : `.bss` ≈ INT8 → **démontre** que le gain n'est pas gratuit ;
- **bit-packé** (2/4 poids par octet) : `.bss` réduit → **matérialise** le ÷8/÷16, au coût du dépacking (latence).

L'écart entre les deux est **le résultat scientifique** du sprint (théorie ↔ matériel), à mettre en figure.

## Tâches

### Bloc A — Sélection & cadrage

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4801 | **Sélection des configs gagnantes** (frontière/agressive/référence × datasets, issues S4708) + cadrage build `-DEWC_INT4`/`-DEWC_INT2`/`-DEWC_INTx_PACKED` | 🔴 | `docs/sprints/sprint_48/S4801_selection_cadrage.md` | 📝 Doc |

### Bloc B — Firmware & export

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4802 | **Kernels sub-INT8 firmware** : variantes de build sur `ewc_head_int8_v2.c` (QMAX 7/3), chemin **bit-packing** (dépack → MAC FPU), test Unity parité C↔Python | 🔴 | `firmware/stm32f4_blink/{src,inc}/ewc_head_int8_v2.*`, `firmware/stm32f4_blink/tests/test_ewc_subint8.c` | 📝 Doc |
| S4803 | **Export** : `export_weights_c.py --ewc-subint8 --weight-bits N [--packed]` (réutilise primitives émulateur S47 = parité) → header généré + `--ewc-subint8-test-vectors` golden | 🔴 | `scripts/export_weights_c.py`, `firmware/stm32f4_blink/inc/ewc_head_subint8_weights.h` (généré) | 📝 Doc |

### Bloc C — Mesures board

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4804 | **Driver board** `run_s48_board_depth.py` (train réf EWC → export → build par variante → flash → stream sans `--update` → parité) ; **cellules mesure réelle** : latence DWT P50/P99, `.bss` (packé/non-packé), AUROC, 0 CRC | 🔴 | `scripts/run_s48_board_depth.py`, `experiments/exp_S48_board/` | 📝 Doc |
| S4805 | **Parité + agrégation** : `board_pc_parity48.py` (réplique PC = émulateur S47) + `aggregate_sprint48.py` → `exp_S48_summary.json` (bits × granularité × plateforme ; RAM mesurée vs théorique bit-packée) | 🟠 | `scripts/board_pc_parity48.py`, `scripts/aggregate_sprint48.py`, `experiments/exp_S48_summary.json` | 📝 Doc |

### Bloc D — Assemblage & clôture

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4806 | **Notebook + figures board** : heatmaps board symétriques au PC (AUROC/RAM/latence vs bits ; `.bss` packé vs non-packé ; N/A gris) | 🟠 | `notebooks/cl_eval/quant_depth_board/comparison.ipynb`, `docs/figures/quant_depth_board/` | 📝 Doc |
| S4807 | **Tests + docs** : `test_sprint48_board.py` (Gap 2 latence<100 ms, Gap 3 `.bss` packé, parité=1.000, 0-chiffre) + MAJ roadmap/triple_gap + `graphify_sprint_update` | 🟡 | `tests/test_sprint48_board.py`, `docs/roadmap_phase2.md`, `docs/triple_gap.md` | 📝 Doc |

## Ordre d'exécution recommandé

```
S4801 (sélection configs + cadrage build)
   │
   ▼
S4802 (kernels sub-INT8 + packing + Unity)  ──►  S4803 (export --ewc-subint8 + golden)
   │                                                     │
   └──────────────────────┬──────────────────────────────┘
                          ▼
                 S4804 (driver board : flash + mesure DWT/.bss/parité)
                          │
                          ▼
                 S4805 (parité + agrégation summary)
                          │
                          ▼
                 S4806 (notebook + figures board)
                          │
                          ▼
                 S4807 (tests + roadmap + triple_gap)
```

## Sources de données (Sprint 48, lecture seule)

| Dataset | Loader / scénario CL | Rôle Sprint 48 |
| ------- | -------------------- | -------------- |
| Monitoring (D2) | `get_cl_dataloaders` — domain-incrémental, 3 tâches | Portage board schémas gagnants S47 |
| Pronostia (D4) | `get_pronostia_dataloaders` — domain-incrémental | Portage board schémas gagnants S47 |

## Livrables

1. Kernels sub-INT8 firmware (`-DEWC_INT4`/`-DEWC_INT2` ± packing) + Unity parité (S4802).
2. `export_weights_c.py --ewc-subint8` → header généré + golden vectors (S4803).
3. `run_s48_board_depth.py` + `experiments/exp_S48_board/` — mesures réelles (latence DWT, `.bss` packé/non, AUROC, CRC).
4. `board_pc_parity48.py` + `aggregate_sprint48.py` → `exp_S48_summary.json`.
5. Notebook `quant_depth_board/comparison.ipynb` + PNG `docs/figures/quant_depth_board/`.
6. `tests/test_sprint48_board.py` + MAJ roadmap/triple_gap.

## Questions ouvertes

- `TODO(dorra)` : coût latence du **dépacking** (INT4 2/octet, INT2 4/octet) sur Cortex-M4 — reste-t-il ≪ 100 ms
  (Gap 2) ? Mesure DWT S4804. Piste SIMD (`SMLAD`) / CMSIS-NN pour le MAC entier (cf. `S3910_simd_cmsis_spec.md`).
- `TODO(dorra)` : à très bas bits (INT2/ternaire), certains schémas peuvent **déborder la SRAM** ou casser
  l'AUROC → **N/A honnête** attendu, à documenter (comme PSI×gas_sensor S45).
- `TODO(arnaud)` : cohérence de la métrique board (AUROC binaire, voie S28/S4601).

## Bilan (à compléter)

| Tâche | Statut | Temps réel | Notes |
|-------|:------:|:----------:|-------|
| S4800 | 📝 Doc | — | Overview + cadrage board |
| S4801–S4807 | 📝 Doc | — | Documentés ; implémentation à venir (board supposée disponible) |
