# S3606 — Agrégation de tous les métriques

| Champ | Valeur |
|-------|--------|
| **Sprint** | 36 |
| **Priorité** | 🟡 Important — un agrégat unique alimente le notebook ; améliore la lisibilité mais ne bloque pas la production des données. |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 3h |
| **Dépendances** | S3602–S3605 ✅ (toutes les sorties JSON) |
| **Fichiers cibles** | `scripts/aggregate_sprint36.py`, `experiments/exp_S36_summary.json` |
| **Références** | Sprint 35 `exp_S35_board_sweep_summary.json` (format d'agrégat) · Sprint 32 `exp_S32_board_sweep_summary.json` |

---

## Contexte

Les résultats sont dispersés (PC, board gelé, board online, parité) × conditions × datasets.
Un fichier d'agrégat unique simplifie le notebook (S3607) et sert de table de référence
(« tous les métriques en un endroit »), à l'image des `*_sweep_summary.json` des Sprints 32/35.

## Spec

`experiments/exp_S36_summary.json` — indexé `[dataset][condition][platform/protocol]` :

```json
{
  "generated": null,
  "model": "ewc",
  "results": {
    "pronostia": {
      "all": {
        "pc":            {"acc_final": null, "aa": null, "af": null, "bwt": null,
                          "f1_faulty": null, "f1_macro": null, "roc_auc": null,
                          "ram_peak_bytes": null, "inference_latency_ms": null,
                          "per_task_acc": {}},
        "board_frozen":  {"online_accuracy": null, "f1_faulty": null, "roc_auc": null,
                          "latency_us_p50": null, "latency_us_p99": null,
                          "bss_bytes": null, "parity_rate": null},
        "board_online":  {"online_accuracy": null, "online_forgetting": null,
                          "latency_us_p50": null, "latency_us_p99": null,
                          "parity_rate": null},
        "delta_pc_board": {"acc_final": null, "f1_faulty": null}   // |PC − board|
      },
      "5feat": { "...": "idem" }
    },
    "monitoring": { "...": "idem" }
  }
}
```

**Règles** :
- Le script **lit** les `results.json`/`*_parity_*.json` existants — aucun recalcul de métrique (réutiliser les valeurs déjà stockées).
- `delta_pc_board` = écart absolu PC vs board (acc_final, F1) pour quantifier la fidélité du portage.
- Champs `null` tant que les runs amont n'ont pas tourné.

## Vérification

```bash
python scripts/aggregate_sprint36.py        # → experiments/exp_S36_summary.json
python -c "import json; s=json.load(open('experiments/exp_S36_summary.json')); \
assert set(s['results'])=={'pronostia','monitoring'}; print('summary OK')"
```

## Implémentation (✅)

- [x] `scripts/aggregate_sprint36.py` lit en **lecture seule** les `results.json` PC/board + parité
      (aucun recalcul) et produit `experiments/exp_S36_summary.json` indexé `[dataset][condition][platform]`.
- [x] `delta_pc_board` = |PC − board_frozen| sur `acc_final`/`f1_faulty`.
- [x] Champs absents → `null` (robuste aux runs amont manquants).

### Résultats (27 juin 2026) — `experiments/exp_S36_summary.json`

| Cellule | PC acc_final | board_frozen acc | Δacc PC↔board | parity online~ |
|---------|:-----------:|:----------------:|:-------------:|:--------------:|
| 5feat·pronostia | 0.9887 | 0.9821 | 0.0066 | 0.9754 |
| all·pronostia | 0.9834 | 0.9831 | 0.0003 | 0.9626 |
| 5feat·monitoring | 0.9791 | 0.9846 | 0.0055 | 0.9887 |
| all·monitoring | 0.9791 | 0.9846 | 0.0055 | 0.9887 |

- **Δacc_final ≤ 0.007** sur les 4 cellules ⇒ fidélité du portage EWC confirmée.
