# S3603 — Runs board, passe GELÉE (parité exacte + latence inférence)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 36 |
| **Priorité** | 🔴 Critique — c'est la passe qui garantit la **parité exacte** des prédictions PC↔board (poids gelés) et mesure la **latence inférence seule**. |
| **Statut** | ✅ Implémenté (board réelle) |
| **Durée estimée** | 5h |
| **Dépendances** | S3601 ✅, S3602 ✅ · `scripts/run_feature_condition_board.py` ✅ (driver complet) · `scripts/train_board_reference.py` ✅ (`EWCMlpMulticlass(k)`) · `scripts/export_weights_c.py --ewc-head` ✅ · firmware EWC `firmware/stm32f4_blink/src/pipeline.c` + `inc/ewc_head.h` (`EWC_IN`) + `PROTO_MAX_N` ✅ · `scripts/sensor_stream.py --condition --proto 3` ✅ · `src/evaluation/feature_conditions.py` ✅ |
| **Fichiers cibles** | `experiments/exp_S36_board_frozen_{condition}_ewc_{dataset}/` (réutilise `run_feature_condition_board.py`, pas de code neuf nécessaire) |
| **Références** | Sprint 35 S3508 (parité EWC exacte k=1→21) · Sprint 32 (parité par construction) |

---

## Contexte

La **parité exacte** board↔PC pour EWC n'existe que **poids gelés** : on entraîne le modèle
de référence sur PC, on exporte les poids en header C, on flashe, puis on streame **sans
`--update`**. Le forward C étant identique au PyTorch, `pred_board == pred_pc` échantillon
par échantillon. C'est exactement le pipeline du driver `run_feature_condition_board.py`
(Sprint 35) — **on le réutilise tel quel**, paramétré pour EWC × {pronostia, monitoring} × {5feat, all}.

## Spec

Par `(condition, dataset)`, le driver enchaîne (réutilisé, **rien à réécrire**) :

1. **Train réf** : `train_board_reference.py` → `EWCMlpMulticlass(k→32→16→2)` sur les colonnes `resolve_feature_indices(condition,"ewc",dataset)`.
2. **Export** : `export_weights_c.py --ewc-head` → `inc/model_weights_ewc.h` (dim native lue du checkpoint).
3. **Build/flash** : `make` avec `-DEWC_IN=k` (+ `-DPROTO_MAX_N=k` si `k>16` — non requis ici : Pronostia=13, Monitoring≈5) ; **1 flash par (condition, dataset)**.
4. **Stream gelé** : `sensor_stream.py --condition ... --proto 3 --dump-samples` **sans `--update`** (split test complet).
5. **Parité** : `_parity()` compare `pred_board` vs `pred_pc` (PC = S3602).

Sortie `exp_S36_board_frozen_{condition}_ewc_{dataset}/results.json` :

```json
{
  "exp_id": "exp_S36_board_frozen_all_ewc_pronostia",
  "platform": "nucleo_f439zi", "model": "ewc", "dataset": "pronostia",
  "condition": "all", "n_features": 13, "stream_mode": "frozen (sans --update)",
  "online_accuracy": null, "f1_faulty": null, "f1_macro": null, "roc_auc": null,
  "latency_us_p50": null, "latency_us_p99": null, "bss_bytes": null,
  "parity_class": "exact", "parity_ok": null, "parity_rate": null, "n_compared": null,
  "gap2_latency_compliant": null
}
```

**Règles** :
- Latence ici = **inférence seule** (DWT, sans MAJ).
- Parité attendue **exacte** (poids gelés, mêmes features, même split).
- Chiffres « à mesurer » tant que la NUCLEO n'a pas tourné.

## Vérification

```bash
python scripts/run_feature_condition_board.py \
  --model ewc --dataset pronostia --condition all \
  --proto 3 --rate-hz 50 --no-update          # → exp_S36_board_frozen_all_ewc_pronostia/

python -c "import json; r=json.load(open('experiments/exp_S36_board_frozen_all_ewc_pronostia/results.json')); \
assert r['parity_class']=='exact' and 'latency_us_p50' in r; print('board frozen OK')"
```

## Implémentation (✅)

- [x] Driver dédié `scripts/run_sprint36_board.py --pass frozen` (EWC-only) réutilisant les helpers
      de `run_feature_condition_board.py` (`train_maha_board`, `_pc_pred_ewc`, `_bss_bytes`) et le
      streaming in-process de `sensor_stream.py` (`_stream_uart`, `_compute_stats`). **Sans `--update`**,
      sorties `exp_S36_board_frozen_*`. (Le snippet de vérif d'origine — `run_feature_condition_board.py
      --model ewc --no-update` — n'était pas applicable : ce script S35 n'a ni `--model` ni `--no-update`
      ni préfixe `exp_S36`.)
- [x] Parité **exacte** EWC vérifiée sur le **split complet** (`n_samples=len(X)`), les 2 datasets ×
      2 conditions : `parity_rate=1.0000` partout.
- [x] `.bss` et latence inférence P50 ≪ 100 ms confirmés (Gap 2).

### Résultats board réelle NUCLEO-F439ZI (12 juin 2026) — `experiments/exp_S36_board_frozen_*/`

| Cellule | k | `.bss` (B) | lat_inf P50 | parité | F1_faulty | Gap 2 |
|---------|---|-----------|-------------|--------|-----------|-------|
| 5feat·monitoring | 4 | 100 152 | 48 µs | **1.000** ✅ | 0.919 | ✅ |
| all·monitoring | 4 | 100 152 | 48 µs | **1.000** ✅ | 0.919 | ✅ |
| 5feat·pronostia | 5 | 105 036 | 50 µs | **1.000** ✅ | 0.916 | ✅ |
| all·pronostia | 13 | 144 516 | 65 µs | **1.000** ✅ | 0.918 | ✅ |

- **Parité exacte par construction** : le board est flashé avec le checkpoint EWC produit par S3602
  (`exp_S36_PC_*/checkpoints/ewc_head.pt`) ; `_pc_pred_ewc` rejoue le même checkpoint sur les features
  dumpées → 0 désaccord sur tous les échantillons streamés.
- Latence inférence seule 48–65 µs (croît avec k) ; `.bss` 100–145 Ko (38–55 % de 256 Ko).
- Un Maha de référence est entraîné par cellule **uniquement** pour la cohérence des dims du build
  (`model_weights.h`) ; il n'est pas streamé.
