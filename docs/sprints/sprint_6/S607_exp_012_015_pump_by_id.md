# S6-06 — Run exp_012–015 : pump_by_id pour TinyOL, EWC, HDC, Mahalanobis

| Champ | Valeur |
|-------|--------|
| **ID** | S6-06 |
| **Sprint** | Sprint 6 — Phase 1 Extension (≥ 15 avril 2026) |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | `configs/pump_by_id_config.yaml` (existant), `get_pump_dataloaders_by_id()` (existant) |
| **Fichiers cibles** | `experiments/exp_012–015/` |
| **Complété le** | 14 avril 2026 |
| **Statut** | ✅ Complété |

---

## Objectif

Lancer les 4 expériences supervisées sur le scénario **pump_by_id** (5 tâches P1→P5) :

| Exp | Modèle | Script |
|-----|--------|--------|
| exp_012_tinyol_pump_by_id | TinyOL | `scripts/train_tinyol.py` |
| exp_013_ewc_pump_by_id | EWC Online + MLP | `scripts/train_ewc.py` |
| exp_014_hdc_pump_by_id | HDC (D=1024) | `scripts/train_hdc.py` |
| exp_015_mahalanobis_pump_by_id | Mahalanobis | `scripts/train_unsupervised.py` |

**Critère de succès** : 4 répertoires `experiments/exp_012–015/` créés, chacun avec
`metrics.json` (AA, AF, BWT, ram_peak_bytes, inference_latency_ms) et `acc_matrix.npy`.

---

## Commandes d'exécution

```bash
# exp_012 — TinyOL
python scripts/train_tinyol.py \
  --config configs/tinyol_config.yaml \
  --data_config configs/pump_by_id_config.yaml \
  --exp_dir experiments/exp_012_tinyol_pump_by_id

# exp_013 — EWC
python scripts/train_ewc.py \
  --config configs/ewc_pump_config.yaml \
  --data_config configs/pump_by_id_config.yaml \
  --exp_dir experiments/exp_013_ewc_pump_by_id

# exp_014 — HDC
python scripts/train_hdc.py \
  --config configs/hdc_pump_config.yaml \
  --data_config configs/pump_by_id_config.yaml \
  --exp_dir experiments/exp_014_hdc_pump_by_id

# exp_015 — Mahalanobis
python scripts/train_unsupervised.py \
  --config configs/unsupervised_config.yaml \
  --data_config configs/pump_by_id_config.yaml \
  --model mahalanobis \
  --exp_dir experiments/exp_015_mahalanobis_pump_by_id
```

---

## Structure de sortie attendue (par expérience)

```
experiments/exp_0XX_<model>_pump_by_id/
├── config_snapshot.yaml
├── results/
│   ├── metrics.json          # AA, AF, BWT, ram_peak_bytes, inference_latency_ms
│   ├── acc_matrix.npy        # matrice 5×5 (task_trained × task_evaluated)
│   └── memory_report.json
└── checkpoints/
    └── <model>_task5_final.pt  # (modèles supervisés seulement)
```

---

## Métriques à enregistrer

| Métrique | Source |
|----------|--------|
| `acc_final` | Accuracy sur toutes les tâches vues après tâche 5 |
| `avg_forgetting` (AF) | Chute moyenne d'accuracy entre pic et fin par tâche |
| `backward_transfer` (BWT) | Impact entraînement futur sur tâches passées |
| `ram_peak_bytes` | `tracemalloc` peak pendant l'entraînement complet |
| `inference_latency_ms` | Moyenne sur 100 runs, forward pass seul |

---

## Critères d'acceptation

- [x] `experiments/exp_012_tinyol_pump_by_id/results/metrics.json` existe — AA=0.563
- [x] `experiments/exp_013_ewc_pump_by_id/results/metrics.json` existe — AA=0.566
- [x] `experiments/exp_014_hdc_pump_by_id/results/metrics.json` existe — AA=0.502
- [x] `experiments/exp_015_mahalanobis_pump_by_id/results/metrics.json` existe — AA=0.447
- [x] Chaque `acc_matrix.npy` de shape `(5, 5)` (5 tâches pump_by_id)

> **Note** : la commande exp_015 du doc requiert `--dataset pump` (absent dans la spec originale — défaut `monitoring`).
> `train_unsupervised.py` corrigé à l'exécution.

---

## Vérification post-run

```bash
python -c "
import json, numpy as np, pathlib
for exp in ['exp_012_tinyol', 'exp_013_ewc', 'exp_014_hdc', 'exp_015_mahalanobis']:
    base = pathlib.Path(f'experiments/{exp}_pump_by_id')
    m = json.loads((base / 'results/metrics.json').read_text())
    mat = np.load(base / 'results/acc_matrix.npy')
    print(f'{exp}: AA={m.get(\"acc_final\", \"?\"):.3f}, shape={mat.shape}')
"
```

---

## Questions ouvertes

- `FIXME(gap1)` : Comparer les AA des 3 scénarios pump (chronologique 3 tâches, pump_by_id 5 tâches,
  temporal_window 4 tâches) pour identifier lequel est le plus difficile. Nécessaire pour le manuscrit.
