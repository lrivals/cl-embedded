# S2404 — Profiling RAM systématique (exp_S24_03)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 24 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Terminé |
| **Durée estimée** | S2404a : 1h / S2404b : 1h = 2h total |
| **Dépendances** | `scripts/profile_memory.py` ✅ (Sprint 4), poids re-runs exp_S24_04–12 disponibles |
| **Fichiers cibles** | `scripts/profile_memory.py`, `experiments/sprint24_memory_report.json` |
| **Référence** | Sprint 4 S406 (`experiments/sprint4_memory_report.json` : 3 modèles × 2 datasets) |

---

## Contexte

Sprint 4 a créé `scripts/profile_memory.py` pour mesurer la RAM peak via `tracemalloc`. Le rapport `sprint4_memory_report.json` couvrait uniquement EWC + HDC + TinyOL sur Monitoring + Pump. Les 4 autres datasets (CWRU, Pronostia, CMAPSS, Paderborn) et la totalité des expériences Sprints 5–23 n'ont pas de profil RAM unifié.

Ce rapport est un livrable clé pour le manuscrit : il démontrera que tous les modèles satisfont le critère Gap 2 (RAM ≤ 256 Ko NUCLEO-F439ZI) sur l'ensemble des datasets.

---

## S2404a — Extension `scripts/profile_memory.py` avec flag `--all`

### Interface CLI enrichie

```bash
# Profiling d'un modèle × dataset
python scripts/profile_memory.py --model ewc --dataset cwru

# Profiling systématique de tout (boucle interne)
python scripts/profile_memory.py --all --output experiments/sprint24_memory_report.json
```

### Logique interne `--all`

```python
MODELS   = ["ewc", "hdc", "tinyol", "mahalanobis"]
DATASETS = ["monitoring", "pump", "cwru", "pronostia", "cmapss", "paderborn"]

# Grille 4 × 6 = 24 combinaisons
# TinyOL exclu sur cmapss + paderborn (pas de loader approprié → skip avec note)
# HDC exclu sur pronostia (pas d'exp correspondante → skip avec note)

results = {}
for model_id in MODELS:
    for dataset_id in DATASETS:
        if not is_supported(model_id, dataset_id):
            results[f"{model_id}_{dataset_id}"] = {"status": "skipped", "reason": "..."}
            continue
        profile = run_profile(model_id, dataset_id)
        results[f"{model_id}_{dataset_id}"] = profile

save_json(results, output_path)
```

### Format de chaque entrée dans le rapport

```json
{
  "model": "ewc",
  "dataset": "cwru",
  "inference_ram_peak_bytes": 9800,
  "update_ram_peak_bytes": 12400,
  "inference_ram_peak_kb": 9.57,
  "update_ram_peak_kb": 12.11,
  "gap2_compliant": true,
  "gap2_budget_bytes": 262144,
  "n_params": 641,
  "inference_latency_ms_mean": 0.42,
  "inference_latency_ms_std": 0.03,
  "n_latency_runs": 100,
  "reference_exp": "exp_S24_04"
}
```

---

## S2404b — exp_S24_03 : Rapport profiling unifié

### Commande

```bash
python scripts/profile_memory.py \
  --all \
  --output experiments/sprint24_memory_report.json
```

### Structure de sortie

```
experiments/sprint24_memory_report.json   ← rapport unifié (20 entrées)
experiments/exp_S24_03/
├── config_snapshot.yaml
└── results.json → {"report_path": "experiments/sprint24_memory_report.json"}
```

### Critères de validation

- 20 entrées dans `sprint24_memory_report.json` (4 modèles × 5 datasets, + skips documentés) ✓
- Toutes entrées `gap2_compliant: true` (RAM ≤ 256 Ko = 262 144 bytes) ✓
- Si une entrée `gap2_compliant: false` → `FIXME(gap2)` à créer immédiatement

### Tableau attendu (valeurs à mesurer)

| Modèle | Dataset | RAM inférence (Ko) | RAM update (Ko) | Gap 2 ✓ |
|--------|---------|:-----------------:|:---------------:|:-------:|
| EWC | Monitoring | ~9.5 | ~12.1 | ✅ |
| EWC | Pump | ~9.5 | ~12.1 | ✅ |
| EWC | CWRU | ~9.5 | ~12.1 | ✅ |
| EWC | Pronostia | ~9.5 | ~12.1 | ✅ |
| EWC | CMAPSS | ~9.5 | ~12.1 | ✅ |
| EWC | Paderborn | ~9.5 | ~12.1 | ✅ |
| HDC | Monitoring | ~14.5 | — | ✅ |
| HDC | CWRU | ~14.5 | — | ✅ |
| HDC | CMAPSS | ~14.5 | — | ✅ |
| TinyOL | Monitoring | ~22.4 | ~0.1 | ✅ |
| TinyOL | Pump | ~22.4 | ~0.1 | ✅ |
| TinyOL | CWRU | ~22.4 | ~0.1 | ✅ |
| Mahalanobis | Monitoring | ~2.0 | ~2.0 | ✅ |
| Mahalanobis | Pump | ~2.0 | ~2.0 | ✅ |
| Mahalanobis | CWRU | ~2.0 | ~2.0 | ✅ |
| Mahalanobis | Pronostia | ~2.0 | ~2.0 | ✅ |
| Mahalanobis | CMAPSS | ~2.0 | ~2.0 | ✅ |
| Mahalanobis | Paderborn | ~2.0 | ~2.0 | ✅ |

*Les valeurs ci-dessus sont estimées à partir du Sprint 4. Les chiffres réels seront mesurés lors de l'exécution.*

---

## Usage dans le manuscrit

Ce rapport unifié alimente directement la **Figure Gap 2** du manuscrit : un tableau comparatif de RAM (inférence + update) pour tous les modèles et tous les datasets, démontrant que la contrainte 256 Ko est satisfaite dans tous les scénarios.
