# S6-07 — Run exp_016–019 : monitoring_by_location pour EWC, HDC, TinyOL, Mahalanobis

| Champ | Valeur |
|-------|--------|
| **ID** | S6-07 |
| **Sprint** | Sprint 6 — Phase 1 Extension (≥ 15 avril 2026) |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | `configs/monitoring_by_location_config.yaml` (existant), S5-19 |
| **Fichiers cibles** | `experiments/exp_016–019/` |
| **Complété le** | 14 avril 2026 |
| **Statut** | ✅ Complété |

---

## Objectif

Lancer les 4 expériences supervisées sur le scénario **monitoring_by_location** (5 tâches ATL→SFO)
sur le Dataset 2 (Industrial Equipment Monitoring) :

| Exp | Modèle | Script |
|-----|--------|--------|
| exp_016_ewc_monitoring_by_location | EWC Online + MLP | `scripts/train_ewc.py` |
| exp_017_hdc_monitoring_by_location | HDC (D=1024) | `scripts/train_hdc.py` |
| exp_018_tinyol_monitoring_by_location | TinyOL | `scripts/train_tinyol.py` |
| exp_019_mahalanobis_monitoring_by_location | Mahalanobis | `scripts/train_unsupervised.py` |

**Critère de succès** : 4 répertoires créés, chacun avec `metrics.json` + `acc_matrix.npy` shape `(5, 5)`.

---

## Commandes d'exécution

```bash
# exp_016 — EWC
python scripts/train_ewc.py \
  --config configs/ewc_config.yaml \
  --data_config configs/monitoring_by_location_config.yaml \
  --exp_dir experiments/exp_016_ewc_monitoring_by_location

# exp_017 — HDC
python scripts/train_hdc.py \
  --config configs/hdc_config.yaml \
  --data_config configs/monitoring_by_location_config.yaml \
  --exp_dir experiments/exp_017_hdc_monitoring_by_location

# exp_018 — TinyOL
python scripts/train_tinyol.py \
  --config configs/tinyol_monitoring_config.yaml \
  --data_config configs/monitoring_by_location_config.yaml \
  --exp_dir experiments/exp_018_tinyol_monitoring_by_location

# exp_019 — Mahalanobis
python scripts/train_unsupervised.py \
  --config configs/unsupervised_config.yaml \
  --data_config configs/monitoring_by_location_config.yaml \
  --model mahalanobis \
  --exp_dir experiments/exp_019_mahalanobis_monitoring_by_location
```

---

## Structure de sortie attendue

```
experiments/exp_0XX_<model>_monitoring_by_location/
├── config_snapshot.yaml
├── results/
│   ├── metrics.json
│   ├── acc_matrix.npy    # shape (5, 5)
│   └── memory_report.json
└── checkpoints/
```

---

## Critères d'acceptation

- [x] 4 répertoires `experiments/exp_016–019/` créés
- [x] Chaque `metrics.json` contient `acc_final`, `avg_forgetting`, `backward_transfer`, `ram_peak_bytes` — AA : EWC=0.982, HDC=0.856, TinyOL=0.942, Mahalanobis=0.951
- [x] Chaque `acc_matrix.npy` de shape `(5, 5)` (5 locations : Atlanta, Chicago, Houston, New York, San Francisco)

> **Bug corrigé** : `train_tinyol.py/_load_tasks()` ne gérait pas `task_split == "by_location"` pour le dataset monitoring → tombait sur le loader by_equipment (3 tâches). Fix ajouté dans [scripts/train_tinyol.py](../../../scripts/train_tinyol.py). exp_018 re-exécuté avec 5 tâches.

---

## Vérification post-run

```bash
python -c "
import json, numpy as np, pathlib
exps = [
    ('exp_016_ewc', 'ewc'),
    ('exp_017_hdc', 'hdc'),
    ('exp_018_tinyol', 'tinyol'),
    ('exp_019_mahalanobis', 'mahalanobis'),
]
for prefix, model in exps:
    base = pathlib.Path(f'experiments/{prefix}_monitoring_by_location')
    m = json.loads((base / 'results/metrics.json').read_text())
    mat = np.load(base / 'results/acc_matrix.npy')
    print(f'{model}: AA={m.get(\"acc_final\", \"?\"):.3f}, shape={mat.shape}')
"
```

---

## Questions ouvertes

- `FIXME(gap1)` : Les 5 locations (Atlanta, Chicago, Dallas, Houston, San Francisco) provoquent-elles
  un domain shift mesurable ? La littérature (Hurtado2023CLPdM) suggère que le shift géographique
  peut être subtil si les équipements sont identiques.
