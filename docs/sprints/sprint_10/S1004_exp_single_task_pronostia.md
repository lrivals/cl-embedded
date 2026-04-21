# S10-04 — Run exp_044–049 : Baseline Single-Task Pronostia (6 modèles)

| Champ | Valeur |
|-------|--------|
| **ID** | S10-04 |
| **Sprint** | Sprint 10 — Phase 1 Extension |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 2h |
| **Dépendances** | S10-01 (`PronostiaDataset`), S10-02 (configs validées) |
| **Fichiers cibles** | `experiments/exp_044–049/` |

---

## Objectif

Entraîner et évaluer les 6 modèles du projet sur FEMTO PRONOSTIA en mode **single-task** (toutes conditions fusionnées, pas de découpage CL). Ces expériences servent de **baseline ultime** : elles répondent à la question _"Quelle est la performance maximale atteignable sur PRONOSTIA sans contrainte CL ?"_

Les résultats de exp_044–049 sont le point de comparaison pour exp_050–055 (scénario CL by_condition).

---

## Commandes d'exécution

```bash
# exp_044 — EWC no_split
python scripts/train_ewc.py \
    --config configs/ewc_pronostia_single_task_config.yaml \
    --exp_id exp_044

# exp_045 — HDC no_split
python scripts/train_hdc.py \
    --config configs/hdc_pronostia_single_task_config.yaml \
    --exp_id exp_045

# exp_046 — TinyOL no_split
python scripts/train_tinyol.py \
    --config configs/tinyol_pronostia_single_task_config.yaml \
    --exp_id exp_046

# exp_047 — KMeans no_split
python scripts/train_unsupervised.py --model kmeans \
    --config configs/kmeans_pronostia_single_task_config.yaml \
    --exp_id exp_047

# exp_048 — Mahalanobis no_split
python scripts/train_unsupervised.py --model mahalanobis \
    --config configs/mahalanobis_pronostia_single_task_config.yaml \
    --exp_id exp_048

# exp_049 — DBSCAN no_split
python scripts/train_unsupervised.py --model dbscan \
    --config configs/dbscan_pronostia_single_task_config.yaml \
    --exp_id exp_049
```

---

## Structure de sortie attendue

```
experiments/
├── exp_044_ewc_pronostia_no_split/
│   ├── config_snapshot.yaml
│   └── results/
│       └── metrics_single_task.json
├── exp_045_hdc_pronostia_no_split/
│   └── ...
├── exp_046_tinyol_pronostia_no_split/
│   └── ...
├── exp_047_kmeans_pronostia_no_split/
│   └── ...
├── exp_048_mahalanobis_pronostia_no_split/
│   └── ...
└── exp_049_dbscan_pronostia_no_split/
    └── ...
```

### Format `metrics_single_task.json`

```json
{
  "exp_id": "exp_044_ewc_pronostia_no_split",
  "model": "ewc",
  "dataset": "pronostia",
  "scenario": "no_split",
  "accuracy": 0.0,
  "f1_score": 0.0,
  "auc_roc": 0.0,
  "ram_peak_bytes": 0,
  "inference_latency_ms": 0.0,
  "n_params": 0
}
```

---

## Métriques à collecter

| Métrique | Description | Module |
|---------|-------------|--------|
| `accuracy` | Accuracy sur le test set global | `evaluation/metrics.py` |
| `f1_score` | F1 macro (classes déséquilibrées — ~10% positifs) | `evaluation/metrics.py` |
| `auc_roc` | AUC-ROC (important pour détection de défaillance) | `evaluation/metrics.py` |
| `ram_peak_bytes` | RAM maximale mesurée à l'exécution (tracemalloc) | `evaluation/memory_profiler.py` |
| `inference_latency_ms` | Latence forward pass (moyenne 100 runs) | `evaluation/memory_profiler.py` |
| `n_params` | Nombre total de paramètres | `model.parameters()` |

> **Note sur les métriques** : le dataset est déséquilibré (~10% positifs). `f1_score` et `auc_roc` sont plus pertinents qu'`accuracy` seule pour évaluer la détection de pré-défaillance.

---

## Résultats attendus

Les modèles supervisés (EWC, TinyOL) devraient obtenir une AUC-ROC > 0.75 sur le scénario single-task si les features statistiques (RMS, kurtosis) capturent bien la dégradation. Si les résultats approchent du hasard (AUC ≈ 0.5), investiguer :
1. La qualité du label (seuil 10% adapté ?)
2. La normalisation inter-conditions
3. La capacité discriminante des 13 features choisies

---

## Critères d'acceptation

- [ ] 6 dossiers `experiments/exp_044–049/` créés avec `config_snapshot.yaml`
- [ ] 6 fichiers `metrics_single_task.json` présents et non vides
- [ ] Les métriques `ram_peak_bytes` et `inference_latency_ms` sont mesurées (non nulles)
- [ ] `exp_044` (EWC) sert de référence pour valider le loader avant de lancer les 5 autres

---

## Questions ouvertes

- `TODO(arnaud)` : Faut-il reporter l'AUC-ROC ou l'accuracy en priorité dans le manuscrit pour les expériences Pronostia ? La métrique standard IEEE PHM 2012 est l'erreur quadratique sur le RUL, mais nous avons choisi un label binaire — quelle métrique est la plus convaincante pour Gap 1 ?
- `TODO(dorra)` : Les modèles non-supervisés (KMeans, Mahalanobis, DBSCAN) entraînés sur toutes les conditions sans découpage CL vont voir des données multi-conditions ; est-ce cohérent pour la baseline ?

---

**Complété le** : _(à renseigner)_
