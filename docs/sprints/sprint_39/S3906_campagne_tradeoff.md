# S3906 — Campagne PC trade-off latence/RAM/accuracy

| Champ | Valeur |
|-------|--------|
| **Sprint** | 39 |
| **Priorité** | 🔴 Critique — produit la grille de comparaison des schémas |
| **Statut** | ✅ Implémenté (1er juillet 2026) — 20 JSON + summary produits, 4 modèles × 5 datasets |
| **Durée estimée** | 3h |
| **Dépendances** | S3905 ✅ (configs) · S3902 ✅ (émulateur) · `src/evaluation/compute_cost.py` |
| **Fichier cible** | `scripts/run_s39_quant_sweep.py` → `experiments/exp_S39_quant_sweep/` |
| **Références** | `scripts/benchmark_int8_fp32.py` (patron campagne + JSON) · `scripts/run_s28_pc_benchmarks.py` |

---

## Contexte

Le cœur de la réponse à la question « y a-t-il des quantifications intermédiaires qui équilibrent latence,
RAM et accuracy ? ». On balaie **4 modèles × 5 datasets × 5 schémas** et on reporte, pour chaque cellule :
métrique (F1/AUROC), RAM analytique (octets de poids), et **proxy de latence** (MACs/BOPs via
`compute_cost.py` — la latence *réelle* exige la board, différée en Partie B).

## Grille

| Modèle | Schémas |
|--------|---------|
| EWC | fp32, int8_legacy, int8_perchannel, q15, mixte (via émulateur S3902) |
| Mahalanobis | fp32, int8, q15 (réutilise `mahalanobis_int8.py`, Sprint 34) |
| HDC | fp32, int8 (exact — INT8==FP32, sert de témoin) |
| TinyOL | fp32, int8 (QAT existant `tinyol_int8.py`) |

Datasets : cmapss, cwru, monitoring, pronostia, paderborn.

## Métriques par cellule (`exp_S39_quant_sweep/{model}_{dataset}.json`)

```json
{
  "model": "ewc", "dataset": "pronostia",
  "schemes": {
    "fp32":            {"metric": 0.916, "ram_weights_bytes": 2816, "bops_proxy": 1.0,  "lat_proxy_rel": 1.0},
    "int8_legacy":     {"metric": 0.14,  "ram_weights_bytes": 704,  "bops_proxy": 0.06, "lat_proxy_rel": 1.84},
    "int8_perchannel": {"metric": 0.xx,  "ram_weights_bytes": 704,  "bops_proxy": 0.06, "lat_proxy_rel": 1.84},
    "q15":             {"metric": 0.xx,  "ram_weights_bytes": 1408, "bops_proxy": 0.25, "lat_proxy_rel": 1.x},
    "mixed":           {"metric": 0.xx,  "ram_weights_bytes": 704,  "bops_proxy": 0.x,  "lat_proxy_rel": 1.x}
  },
  "metric_name": "f1_faulty"
}
```

> **Honnêteté latence** : `lat_proxy_rel` est un **proxy** (BOPs / coût analytique), pas une mesure. Le
> `compute_cost.py` fournit BOPs_fp32/BOPs_int8 = (32/8)² = 16 (Sprint 33). La latence FPU réelle (où INT8
> est *plus lent*) ne se mesure que sur board → renvoyée à S3915. Marquer explicitement `"lat_proxy": true`.

## Livrable

20 JSON (4 modèles × 5 datasets) + `summary.json` agrégé (indexé `[model][dataset][scheme]`), consommés par
le notebook S3911.

## Vérification

```bash
python scripts/run_s39_quant_sweep.py               # → experiments/exp_S39_quant_sweep/
pytest tests/test_s39_quant.py -k sweep -v
```
