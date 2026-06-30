# S4-06 — Profiling mémoire systématique (3 modèles)

| Champ | Valeur |
|-------|--------|
| **ID** | S4-06 |
| **Sprint** | Sprint 4 — Semaine 4 (6–13 mai 2026) |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | exp_001, exp_002, exp_003, exp_004 — checkpoints disponibles |
| **Fichiers cibles** | `scripts/profile_memory.py` · `experiments/exp_004_tinyol_uint8/results/memory_report.json` |
| **Statut** | ✅ Terminé |

---

## Objectif

Produire un rapport de profiling mémoire systématique pour les 3 modèles CL (EWC, HDC, TinyOL) en un seul appel, avec comparaison FP32 vs UINT8 pour TinyOL. Ce rapport fournit les chiffres précis pour le Gap 2 du manuscrit.

> **Note** : `scripts/profile_memory.py` existe déjà (créé et étendu dans les sprints ultérieurs Sprint Phase2). Cette tâche porte sur l'**utilisation initiale** du script avec les 3 modèles de Sprint 4 et l'interprétation des résultats pour la Phase 1.

**Critère de succès** : rapport JSON complet avec RAM inférence + RAM update pour les 3 modèles, tous sous 256 Ko (budget NUCLEO-F439ZI).

---

## Commande de lancement

```bash
python scripts/profile_memory.py \
    --model all \
    --dataset monitoring \
    --ewc_config configs/ewc_config.yaml \
    --hdc_config configs/hdc_config.yaml \
    --tinyol_config configs/tinyol_config.yaml \
    --output experiments/sprint4_memory_report.json
```

---

## Métriques à mesurer pour chaque modèle

| Métrique | Description | Méthode |
|---------|-------------|---------|
| `ram_static_bytes` | Poids modèle seul (FP32) | `estimate_ram_bytes()` |
| `ram_inference_bytes` | Peak tracemalloc forward pass, 1 sample | `tracemalloc` |
| `ram_update_bytes` | Peak tracemalloc forward + backward + step | `tracemalloc` |
| `inference_latency_ms` | Temps moyen forward (100 runs, CPU) | `time.perf_counter` |
| `update_latency_ms` | Temps moyen update (50 runs, CPU) | `time.perf_counter` |
| `within_budget_64ko` | `ram_update_bytes ≤ 65 536` | calculé |
| `within_budget_256ko` | `ram_update_bytes ≤ 262 144` | calculé (NUCLEO-F439ZI) |
| `n_params` | Nombre de paramètres entraînables | `sum(p.numel())` |

---

## Format de sortie `experiments/sprint4_memory_report.json`

```json
{
  "generated": "2026-05-XX",
  "budget_64ko_bytes": 65536,
  "budget_256ko_bytes": 262144,
  "models": {
    "ewc": {
      "ram_static_bytes": 2820,
      "ram_ewc_state_bytes": 8460,
      "ram_inference_bytes": 1171,
      "ram_update_bytes": 6837,
      "inference_latency_ms": 0.036,
      "update_latency_ms": 0.637,
      "n_params": 705,
      "within_budget_64ko": true,
      "within_budget_256ko": true
    },
    "hdc": {
      "ram_static_bytes": 14344,
      "ram_inference_bytes": 14504,
      "ram_update_bytes": null,
      "inference_latency_ms": 0.048,
      "n_params": 2048,
      "within_budget_64ko": true,
      "within_budget_256ko": true
    },
    "tinyol_fp32": {
      "ram_static_bytes": null,
      "ram_inference_bytes": null,
      "ram_update_bytes": 6425,
      "inference_latency_ms": 0.010,
      "n_params": 1506,
      "within_budget_64ko": true,
      "within_budget_256ko": true
    },
    "tinyol_uint8": {
      "ram_static_bytes": null,
      "ram_inference_bytes": null,
      "ram_update_bytes": null,
      "buffer_fp32_bytes": 1800,
      "buffer_uint8_bytes": 450,
      "compression_ratio": 4.0,
      "within_budget_64ko": null,
      "within_budget_256ko": null
    }
  },
  "gap2_summary": {
    "all_within_256ko": null,
    "tightest_model": null,
    "margin_256ko_percent": null
  }
}
```

---

## Tableau de synthèse attendu

| Modèle | RAM inférence | RAM update | Latence inf. | Latence update | Budget 256Ko |
|--------|:-------------:|:----------:|:------------:|:--------------:|:------------:|
| EWC Online | 1.1 Ko | 6.7 Ko | 0.036 ms | 0.637 ms | ✅ |
| HDC Online | 14.2 Ko | — | 0.048 ms | — | ✅ |
| TinyOL FP32 | ? | 6.3 Ko | 0.010 ms | ? | ✅ |
| TinyOL UINT8 | ? | ? | 0.010 ms | ? | ? |

---

## Interprétation Gap 2

Pour chaque modèle, calculer le **pourcentage du budget NUCLEO-F439ZI (256 Ko)** utilisé :

```python
for model, data in report["models"].items():
    ram = data.get("ram_update_bytes") or data.get("ram_inference_bytes", 0)
    pct = ram / 262_144 * 100
    print(f"{model}: {ram:,} B ({pct:.1f}% du budget 256 Ko)")
```

**Objectif manuscrit** : tous les modèles ≤ 10% du budget 256 Ko → marge de 90% pour les données, la pile OS, et les futurs buffers.

---

## Critères d'acceptation

- [ ] `sprint4_memory_report.json` produit avec les 4 modèles (EWC, HDC, TinyOL FP32, TinyOL UINT8)
- [ ] `within_budget_256ko: true` pour tous les modèles
- [ ] `gap2_summary.all_within_256ko` calculé
- [ ] Latences mesurées sur 100 runs (inférence) et 50 runs (update)
- [ ] Rapport loggé dans le terminal avec format lisible (table ASCII)

---

## Questions ouvertes

- `FIXME(gap2)` : Les mesures tracemalloc PC surestiment la RAM réelle MCU (overhead Python). Ajouter une note dans le rapport JSON avec l'estimation analytique (RAM = n_params × 4 B @ FP32) pour comparaison.
- `TODO(dorra)` : Quelle est l'estimation de la RAM d'activation sur NUCLEO-F439ZI pour EWC (forward pass, batch=1) ? Les 1 171 B PC sont-ils représentatifs ?
