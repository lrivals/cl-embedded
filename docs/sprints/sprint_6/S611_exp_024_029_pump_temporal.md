# S6-10 — Run exp_024–029 : pump_by_temporal_window pour les 6 modèles

| Champ | Valeur |
|-------|--------|
| **ID** | S6-10 |
| **Sprint** | Sprint 6 — Phase 1 Extension (≥ 15 avril 2026) |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 3h |
| **Dépendances** | S6-02 (`get_pump_dataloaders_by_temporal_window()` implémenté), S6-03 (config YAML créé) |
| **Fichiers cibles** | `experiments/exp_024–029/` |
| **Complété le** | 2026-04-14 |
| **Statut** | ✅ Terminé |

---

## Objectif

Lancer les 6 expériences (3 supervisées + 3 non supervisées) sur le scénario
**pump_by_temporal_window** (4 tâches Q1→Q4) :

| Exp | Modèle | Script |
|-----|--------|--------|
| exp_024_tinyol_pump_temporal | TinyOL | `scripts/train_tinyol.py` |
| exp_025_ewc_pump_temporal | EWC Online + MLP | `scripts/train_ewc.py` |
| exp_026_hdc_pump_temporal | HDC (D=1024) | `scripts/train_hdc.py` |
| exp_027_mahalanobis_pump_temporal | Mahalanobis | `scripts/train_unsupervised.py` |
| exp_028_kmeans_pump_temporal | KMeans | `scripts/train_unsupervised.py` |
| exp_029_dbscan_pump_temporal | DBSCAN | `scripts/train_unsupervised.py` |

**Critère de succès** : 6 répertoires créés avec `metrics.json` + `acc_matrix.npy` shape `(4, 4)`.

---

## Commandes d'exécution

```bash
# exp_024 — TinyOL
python scripts/train_tinyol.py \
  --config configs/tinyol_config.yaml \
  --data_config configs/pump_by_temporal_window_config.yaml \
  --exp_dir experiments/exp_024_tinyol_pump_temporal

# exp_025 — EWC
python scripts/train_ewc.py \
  --config configs/ewc_pump_config.yaml \
  --data_config configs/pump_by_temporal_window_config.yaml \
  --exp_dir experiments/exp_025_ewc_pump_temporal

# exp_026 — HDC
python scripts/train_hdc.py \
  --config configs/hdc_pump_config.yaml \
  --data_config configs/pump_by_temporal_window_config.yaml \
  --exp_dir experiments/exp_026_hdc_pump_temporal

# exp_027 — Mahalanobis
python scripts/train_unsupervised.py \
  --config configs/unsupervised_config.yaml \
  --data_config configs/pump_by_temporal_window_config.yaml \
  --model mahalanobis \
  --exp_dir experiments/exp_027_mahalanobis_pump_temporal

# exp_028 — KMeans
python scripts/train_unsupervised.py \
  --config configs/unsupervised_config.yaml \
  --data_config configs/pump_by_temporal_window_config.yaml \
  --model kmeans \
  --exp_dir experiments/exp_028_kmeans_pump_temporal

# exp_029 — DBSCAN
python scripts/train_unsupervised.py \
  --config configs/unsupervised_config.yaml \
  --data_config configs/pump_by_temporal_window_config.yaml \
  --model dbscan \
  --exp_dir experiments/exp_029_dbscan_pump_temporal
```

---

## Structure de sortie attendue (par expérience)

```
experiments/exp_0XX_<model>_pump_temporal/
├── config_snapshot.yaml
├── results/
│   ├── metrics.json          # AA, AF, BWT, ram_peak_bytes, inference_latency_ms
│   ├── acc_matrix.npy        # shape (4, 4) — 4 fenêtres temporelles
│   └── memory_report.json
└── checkpoints/
```

---

## Critères d'acceptation

- [x] 6 répertoires `experiments/exp_024–029/` créés
- [x] Chaque `acc_matrix.npy` de shape `(4, 4)` (4 tâches temporelles)
- [x] `ram_peak_bytes` enregistré pour chaque modèle

---

## Vérification post-run

```bash
python -c "
import json, numpy as np, pathlib
exps = [
    'exp_024_tinyol_pump_temporal',
    'exp_025_ewc_pump_temporal',
    'exp_026_hdc_pump_temporal',
    'exp_027_mahalanobis_pump_temporal',
    'exp_028_kmeans_pump_temporal',
    'exp_029_dbscan_pump_temporal',
]
for exp in exps:
    base = pathlib.Path(f'experiments/{exp}')
    m = json.loads((base / 'results/metrics.json').read_text())
    mat = np.load(base / 'results/acc_matrix.npy')
    print(f'{exp}: AA={m.get(\"acc_final\", \"?\"):.3f}, RAM={m.get(\"ram_peak_bytes\", \"?\")}, shape={mat.shape}')
"
```

---

## Questions ouvertes

- `FIXME(gap1)` : Comparer AA sur les 3 scénarios pump (3 tâches chrono, 5 tâches par ID,
  4 tâches temporelles) — lequel induit le plus d'oubli ? Argument central du Gap 1 (validation
  sur données industrielles réelles).
- `TODO(arnaud)` : Le scénario temporel révèle-t-il un pattern clair d'usure progressive
  (accuracy qui chute sur T4) ? Si oui, mentionner dans le manuscrit comme preuve de drift temporel.
