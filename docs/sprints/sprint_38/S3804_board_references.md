# S3804 — Board P0/P1 : références (frozen, always)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 38 |
| **Priorité** | 🟠 Haute — bornes de l'arbitrage (plancher sans MAJ, plafond avec MAJ permanente). |
| **Statut** | ✅ Implémenté — `scripts/run_sprint38_board.py` ; board réelle NUCLEO-F439ZI mesurée (frozen/always × 2 datasets × 2 init_modes). |
| **Durée estimée** | 5h |
| **Dépendances** | S3802 ✅ (checkpoint + seuils) · S3803 ✅ (firmware) · `scripts/run_sprint36_board.py` ✅ (modèle de driver) · `scripts/export_weights_c.py` ✅ · `scripts/sensor_stream.py` ✅ |
| **Fichiers cibles** | `scripts/run_sprint38_board.py` (`--policy frozen\|always`), `experiments/exp_S38_board_{frozen,always}_{dataset}/results.json` |
| **Références** | S3603/S3604 (board frozen/online Sprint 36 comme modèle) |

---

## Contexte

P0 (`frozen`) et P1 (`always`) sont les **bornes de référence** de l'étude d'économie : elles utilisent
le firmware **par défaut** (sans `-DEWC_AUTO_UPDATE`), donc le déclencheur UART historique. Le driver
réutilise les helpers de `run_sprint36_board.py` (build/flash/stream in-process, aucune modif de
`sensor_stream.py`).

## Spec

Étapes communes (par dataset) :
1. Charger le **checkpoint PC** `exp_S38_PC_{policy}_{ds}/checkpoints/ewc_head.pt` (S3802) → modèle
   flashé == modèle PC → parité exacte par construction.
2. Entraîner un Maha de référence (mêmes arrays) pour la cohérence des dims du build.
3. `export_weights_c.py --mahal --ewc-head` → headers C.
4. `make clean && make EWC_IN=5 MAHA_DIM=5 all` ; lire `.bss` ; `make flash`.

- **`--policy frozen` (P0)** : stream **sans `--update`** → latence **inférence seule** (DWT) ;
  parité **exacte** pred_board vs pred_PC. → `exp_S38_board_frozen_{ds}/results.json`.
- **`--policy always` (P1)** : stream **avec `--update`** (flag UART) → latence **inférence + SGD/échantillon** ;
  `n_updates == n_samples` ; parité **approchée** (float32 board ≠ float64 PC).
  → `exp_S38_board_always_{ds}/results.json`.

**Axe `init_modes`** (décision utilisateur) : la grille couvre `pretrained` **et** `scratch` ; le
checkpoint chargé est `exp_S38_PC_{policy}_{ds}_{init_mode}/checkpoints/ewc_head.pt`, et chaque
cellule board porte le suffixe → `exp_S38_board_{policy}_{ds}_{init_mode}`.

**Parité `always`** : le miroir PC (`_pc_always_mirror`) **part du même checkpoint** que la board et
rejoue 1 pas SGD/échantillon dans l'**ordre exact streamé** (features extraites des réponses board)
→ accord board↔PC ≈ (divergence float32/float64 seule). `frozen` : parité **exacte** via `_pc_pred_ewc`.

Métriques stockées : `inference_latency_us` (P50/P99 DWT), `bss_bytes`, `n_updates`, `acc`/`f1`/`parity_rate`,
`gap2_ok` (< 100 ms).

## Vérification

```bash
python scripts/run_sprint38_board.py --policy frozen --dataset monitoring --port /dev/ttyACM0
python scripts/run_sprint38_board.py --policy always --dataset pronostia --port /dev/ttyACM0
```
- frozen : `parity_rate == 1.0` (checkpoint identique) ; latence < always.
- always : `n_updates == n_samples` ; latence inférence+SGD cohérente avec Sprint 26 (~130→400 µs).
- Toutes latences ≪ 100 ms (Gap 2).
