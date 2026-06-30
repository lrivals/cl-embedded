# S3508 — Streaming board par condition (parité board↔PC)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 35 |
| **Priorité** | 🔴 Critique — produit les données board des heatmaps |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 5h |
| **Dépendances** | S3506 (firmware), S3507 (builds par condition), S3504 (F1 hôte), `scripts/sensor_stream.py` ✅ |
| **Fichiers cibles** | `scripts/run_feature_condition_board.py`, `experiments/exp_S35_board_{condition}_{model}_{dataset}/` |
| **Références** | `scripts/run_board_threshold_sweep.py` (driver board train→export→build→flash→stream, Sprint 32) |

---

## Contexte

Re-jouer la détection de panne **sur la NUCLEO-F439ZI réelle** pour les 3 conditions, en mesurant
`acc_final`, `f1_*` (hôte, S3504), latence DWT, `.bss`, et la **parité board↔PC** (EWC+Maha).
S'inspire du driver Sprint 32 (`run_board_threshold_sweep.py`) : train→export→build→**1 flash/condition**→stream.

## Spec

Driver `run_feature_condition_board.py` :

1. Pour chaque `(condition, dataset)` : exporter poids (S3507), builder, **1 flash**.
2. Streamer chaque modèle via `sensor_stream.py` (`--proto 3`, rate-limité), avec `--update` (online).
3. Enregistrer `experiments/exp_S35_board_{condition}_{model}_{dataset}/results.json` :

```json
{ "exp_id": "exp_S35_board_best_ewc_cwru", "condition": "best",
  "model": "ewc", "dataset": "cwru", "platform": "nucleo_f439zi",
  "n_features": <k*>,
  "online_accuracy": ..., "f1_faulty": ..., "f1_macro": ...,
  "latency_us_p50": ..., "latency_us_p99": ..., "bss_bytes": ...,
  "parity_class": "exact|hw_only", "parity_note": "..." }
```

- **Parité** : EWC + Mahalanobis = parité exacte board↔PC (poids exportés).
  HDC (projection embarquée) + TinyOL (init en ligne) = **HW-only, parité N/A** (statut Sprint 32).
- Gap 2 : vérifier latence < 100 ms par condition (`FIXME(gap2)` : condition `all` plus coûteuse).
- **Aucun chiffre board inventé** : champs « à mesurer » tant que la board n'a pas tourné.

## Vérification

```bash
python scripts/run_feature_condition_board.py --dry-run            # valide la matrice (60 cellules)
python scripts/run_feature_condition_board.py --condition all --dataset monitoring  # 1 cellule réelle
ls experiments/ | grep exp_S35_board_
```

## Implémentation (✅)

- **Source de vérité unique** : `resolve_feature_indices` + `load_condition_arrays` déplacées dans
  `src/evaluation/feature_conditions.py` → mêmes colonnes natives côté entraînement board ET côté
  `sensor_stream.py --condition` ⇒ **parité par construction** (board et PC consomment les mêmes
  nombres). `sensor_stream.py` : nouveau flag `--condition` (sélection hôte, **0 changement UART**).
- **Driver `run_feature_condition_board.py`** : pour chaque `(condition, dataset)` — entraîne les
  références board Maha (`MahalanobisDetector`) + EWC (`EWCMlpMulticlass(k,2,[32,16])`) aux dims
  `k` de la condition (`best` → `k` et indices **par modèle**), exporte (S3507), build+flash **1×**
  (dims par modèle via `-D`, `PROTO_MAX_N` si k>16), streame les 4 modèles (**sans `--update`** →
  poids figés → parité exacte), consigne `results.json` + `exp_S35_board_sweep_summary.json`.
  `--skip-existing` (resumable), `--dry-run` (matrice).
- **Parité** : EWC+Maha = exacte (reconstruction numpy `_pc_pred_*` aux dims du checkpoint) ;
  HDC+TinyOL = HW-only (parité N/A). **Cellule réelle validée** (`all×monitoring`, k=4) :
  Maha+EWC `parity_ok=True`, latences P50 Maha 3 µs / EWC 48 µs / HDC 518 µs / TinyOL 3 µs
  (**toutes ≪ 100 ms, Gap 2 ✅**), `.bss=100 096 B`.
- **Aucun chiffre inventé** : cellule en échec/non lancée → champs `"à mesurer"`.
