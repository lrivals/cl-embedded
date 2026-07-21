# S4608 — Board différée : validation « both » sur NUCLEO-F439ZI

| Champ | Valeur |
|-------|--------|
| **Sprint** | 46 |
| **Priorité** | 🟢 Basse — différée (carte indisponible) ; documentée pour reprise immédiate à l'accès NUCLEO. |
| **Statut** | 📝 Doc (board différée) — spec prête ; **`« à mesurer »`** tant que la carte n'a pas streamé |
| **Durée estimée** | 1h doc + ~6h board dès accès carte |
| **Dépendances** | S4602/S4603 ✅ (chemin `both` PC) · `scripts/export_weights_c.py --ewc-head` ✅ · `firmware/.../src/ewc_head_int8.c` + `ewc_head_int8_v2.c` ✅ · `profiling.c` (DWT + `.bss`) ✅ · pattern `run_sprint36_board.py` / `board_pc_parity.py` ✅ |
| **Fichiers cibles** | `experiments/exp_S46_board/` (différé) |
| **Références** | S3915/S3916 (board INT8 v2 différée) · S4002 (board v2 article différée) · S3610 (`ewc_int8_from_fp32`) |

---

## Contexte

Le chemin **both** (QAT → export PTQ) est précisément ce que le firmware exécute : un noyau entier chargé
depuis des poids exportés. Cette tâche valide sur carte réelle que **des poids QAT exportés vers le noyau
INT8 calibré récupèrent la métrique mieux que la PTQ depuis des poids FP32** — la question `TODO(dorra)`
ouverte de S4603. Elle est **différée** car la NUCLEO est indisponible (cohérent S39/S40) ; toutes les
cellules board portent `« à mesurer »`, aucun chiffre inventé.

## Spec

### 1. Ce qui sera mesuré (dès accès carte)

| Métrique | Outil | Critère |
|----------|-------|---------|
| Latence inférence | DWT `profiling_start/stop` | ≤ 100 ms (Gap 2) |
| `.bss` | `make size` | ≤ 256 Ko (Gap 3 RAM) |
| Métrique board (AUROC/F1) `both` | stream `sensor_stream.py` | vs PC `both` |
| Parité board↔PC | `board_pc_parity.py` | frozen = 1.000 exact |
| A/B : `both` vs `after` (PTQ depuis FP32) | comparaison | `both` ≥ `after` attendu |

### 2. Chemin d'exécution (à l'accès carte)

1. Entraîner EWC QAT PC (S4603, `before`), extraire poids.
2. `export_weights_c.py --ewc-head` (poids QAT) → header C.
3. Build firmware (noyau INT8 v2 calibré / per-canal, `ewc_head_int8_v2.c`), flash.
4. Stream Monitoring + Pronostia, mesurer DWT/`.bss`/métrique/parité.
5. A/B contre la colonne `after` (poids FP32 exportés) déjà mesurée S36.

### 3. Marqueur d'honnêteté

Tant que la carte n'a pas tourné : `experiments/exp_S46_board/` **absent** ; les cellules board des
figures/notebook (S4606) et de la table de synthèse (S4600) portent littéralement **`« à mesurer »`**.

## Format de sortie

```json
// experiments/exp_S46_board/<dataset>_both.json  (produit UNIQUEMENT par un run board réel)
{
  "dataset": "monitoring",
  "moment": "both",
  "platform": "board",
  "latency_dwt_us_p50": "à mesurer",
  "bss_bytes": "à mesurer",
  "metric_board": "à mesurer",
  "parity_board_pc": "à mesurer",
  "ab_vs_after": "à mesurer"
}
```

## Contraintes

- **Rien de chiffré** écrit tant que la sonde/carte n'a pas tourné (règle « aucun chiffre inventé »).
- Ne pas toucher au protocole UART ni introduire de flag neuf : réutiliser `ewc_head_int8_v2` sous le
  chemin de build existant (`FRAME_FLAGS_INT8_MODE`, précédent S3610).
- Parité frozen exigée = 1.000 exact (checkpoint PC réutilisé), online = accord documenté.

## Vérification

```bash
# Différé : la spec est prête, la mesure attend la carte.
# À l'accès NUCLEO :
#   python scripts/run_sprint36_board.py --precision int8 ...   # pattern réutilisé
#   ls experiments/exp_S46_board/*_both.json
```

---

## Résolution (implémentée — board réelle NUCLEO-F439ZI)

✅ **Mesurée sur carte réelle** (la NUCLEO était branchée `/dev/ttyACM0` ; la tâche n'est donc
plus différée — décision utilisateur : **QAT multiclasse + flash réel**).

**Réconciliation d'architecture (le nœud de la tâche)** : le head firmware EWC est **multiclasse
2 sorties** (`EWCMlpMulticlass` — ce que `export_weights_c.py --int8-v2` et le kernel v2 consomment),
alors que le QAT S28/S4603 (`EWCMlpInt8Classifier`) est **binaire 1 sortie** → incompatible avec
l'export/firmware. On introduit donc `src/models/ewc/ewc_mlp_multiclass_int8.py::EWCMlpMulticlassInt8`
(nouveau) : miroir de `EWCMlpMulticlass` (fc1→32→16→n_classes) **+ fake-quant** repris de
`EWCMlpInt8Classifier` (poids per-canal symétrique, activations per-tensor affine). Poids sous-jacents
FP32 (fake-quant au forward) → `state_dict` (fc1/fc2/fc3) **directement exportable** comme un head FP32.

**Chemin d'exécution réel** (driver `scripts/run_sprint46_board.py`, réutilise
`run_s40_board_v2.build_and_flash_s40` + `run_sprint36_pc._temporal_tasks/_split_task`) :
1. Entraîner `EWCMlpMulticlassInt8` (QAT) en CL séquentiel — split/seed/hyperparamètres **identiques**
   à la réf PC/board S36 (`configs/sprint36_ewc_comparison.yaml`, seed 42, 5feat) → checkpoint
   `experiments/exp_S46_board/qat_ckpt/{ds}_ewc_head.pt`.
2. `export_weights_c.py --ewc-head --int8-v2` (poids **QAT**, calibration per-canal) → header v2.
3. Build `-DEWC_INT8_V2`, flash, stream (flag UART `FRAME_FLAGS_INT8_MODE` 0x40 — **aucun flag neuf**).
4. A/B contre la colonne `after` (source **FP32**) déjà mesurée S40 (`exp_S40_board_v2`).

**Résultats board réelle (frozen, 5feat, 0 CRC)** :

| Dataset | F1 `both` | F1 `after` (S40) | A/B | Latence DWT p50 | `.bss` | Parité board↔émulateur | RAM |
|---------|:---------:|:----------------:|:---:|:---------------:|:------:|:----------------------:|:---:|
| Monitoring | 0.9213 | 0.9173 | **+0.004** | 65 µs | 101 236 B | **1.000** (0 mismatch) | ÷4 |
| Pronostia | 0.9072 | 0.8995 | **+0.008** | 68 µs | 106 152 B | **1.000** (0 mismatch) | ÷4 |

- **Gap 2 ✅** : 65 / 68 µs ≪ 100 ms.
- **Gap 3 ✅** : RAM poids ÷4 (kernel v2 int8 per-canal).
- **Parité frozen = 1.000 exact** (board == émulateur bit-exact `int8_c_emulation`, par construction).
- **A/B `both` ≥ `after`** : le QAT préserve la métrique et **égale** (marginalement au-dessus) la PTQ
  calibrée sur ce head. Conclusion honnête : sur la NUCLEO, la **calibration du noyau v2** récupère
  l'essentiel ; le QAT n'ajoute pas de gain décisif au-delà (pas d'effet inventé).

**Sortie** : `experiments/exp_S46_board/{monitoring,pronostia}_both.json` (schéma S4608 : moment/
platform/latency_dwt_us_p50/bss_bytes/metric_board/parity_board_pc/ab_vs_after) + `board_samples.json`
par cellule (parité par échantillon). **Tests** : `test_s46_quant_moment.py` couvre l'exportabilité du
head QAT multiclasse et le déterminisme QAT. **Firmware inchangé** : `make test` 134 (2 TinyOL
préexistants hors périmètre, **0 régression** ; `.bss` défaut invariant sous build standard).

_(La passe `online` reste une extension future ; la validation `frozen` établit parité + coût + A/B.)_
