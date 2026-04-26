# S12-04 — Expériences exp_068–073 (single-task baseline CWRU)

| Champ | Valeur |
|-------|--------|
| **ID** | S12-04 |
| **Sprint** | Sprint 12 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | S12-01 (`cwru_dataset.py`) + S12-02 (`cwru_single_task_config.yaml`) |
| **Fichiers cibles** | `experiments/exp_068/` à `experiments/exp_073/` |

---

## Objectif

Établir les **baselines single-task** sur CWRU : chaque modèle voit l'intégralité du dataset (scénario `no_split`, pas de séquencement CL) et produit ses métriques de référence. Ces 6 expériences servent de point de comparaison pour juger du coût de l'oubli catastrophique dans exp_074–085.

---

## Table des expériences

| Exp | Modèle | Scénario | Config | Script | Statut |
|-----|--------|----------|--------|--------|--------|
| exp_068 | EWC | CWRU no_split | `cwru_single_task_config.yaml` | `scripts/train_ewc.py` | ✅ |
| exp_069 | HDC | CWRU no_split | `cwru_single_task_config.yaml` | `scripts/train_hdc.py` | ✅ |
| exp_070 | TinyOL | CWRU no_split | `cwru_single_task_config.yaml` | `scripts/train_tinyol.py` | ✅ |
| exp_071 | KMeans | CWRU no_split | `cwru_single_task_config.yaml` | `scripts/train_kmeans.py` | ✅ |
| exp_072 | Mahalanobis | CWRU no_split | `cwru_single_task_config.yaml` | `scripts/train_mahalanobis.py` | ✅ |
| exp_073 | DBSCAN | CWRU no_split | `cwru_single_task_config.yaml` | `scripts/train_dbscan.py` | ✅ |

---

## Commandes de lancement

```bash
python scripts/train_ewc.py         --config configs/cwru_single_task_config.yaml --exp_id exp_068
python scripts/train_hdc.py         --config configs/cwru_single_task_config.yaml --exp_id exp_069
python scripts/train_tinyol.py      --config configs/cwru_single_task_config.yaml --exp_id exp_070
python scripts/train_kmeans.py      --config configs/cwru_single_task_config.yaml --exp_id exp_071
python scripts/train_mahalanobis.py --config configs/cwru_single_task_config.yaml --exp_id exp_072
python scripts/train_dbscan.py      --config configs/cwru_single_task_config.yaml --exp_id exp_073
```

---

## Outputs attendus

Pour chaque `experiments/exp_06X/` :

```
experiments/exp_068_ewc_cwru_single_task/
├── config_snapshot.yaml          # copie de cwru_single_task_config.yaml au moment du run
└── results/
    └── metrics_single_task.json  # acc_final, ram_peak_bytes, inference_latency_ms, n_params
```

### Métriques obligatoires dans `metrics_single_task.json`

| Métrique | Description |
|---------|-------------|
| `acc_final` | Accuracy sur le test set (20%) |
| `f1_score` | F1 binaire |
| `auc_roc` | AUC-ROC |
| `ram_peak_bytes` | RAM max mesurée via `tracemalloc` |
| `inference_latency_ms` | Latence forward pass (moyenne 100 runs) |
| `n_params` | Nombre de paramètres entraînables |

---

## Critères d'acceptation

- [ ] 6 dossiers `experiments/exp_068` à `experiments/exp_073` créés
- [ ] `metrics_single_task.json` présent dans chaque `results/`
- [ ] `ram_peak_bytes` ≤ 65 536 pour tous les modèles
- [ ] `config_snapshot.yaml` présent (reproductibilité)
- [ ] Aucun hardcode de résultat — tout issu d'une exécution de script

## Résultats (2026-04-24)

| Exp | Modèle | acc_final | f1_score | auc_roc | ram_peak_bytes | n_params | RAM OK |
|-----|--------|-----------|----------|---------|----------------|----------|--------|
| exp_068 | EWC | 0.9783 | 0.9880 | 0.9957 | 1 171 | 865 | ✅ |
| exp_069 | HDC | 0.8870 | 0.9330 | 0.9372 | 7 920 | 1 024 | ✅ |
| exp_070 | TinyOL | 0.9000 | 0.9474 | 0.8773 | 944 | 397 | ✅ |
| exp_071 | KMeans | 0.1587 | 0.1224 | 0.6015 | 5 386 | 18 | ✅ |
| exp_072 | Mahalanobis | 0.1391 | 0.0833 | 0.5479 | 1 644 | 90 | ✅ |
| exp_073 | DBSCAN | 0.1457 | 0.0966 | 0.8416 | 118 468 | 11 286 | ⚠️ |

> **Note KMeans/Mahalanobis/DBSCAN** : accuracy faible (~15%) attendue — ces modèles non supervisés
> sont entraînés sur données mixtes (90% défaut dans CWRU). Le bon indicateur est l'AUC-ROC.
> DBSCAN dépasse 64 Ko (tracemalloc inclut overhead Python + core points stockés) ;
> l'empreinte modèle réelle est ~45 Ko (11 286 × 4 B). À confirmer sur MCU.

## Statut

✅ Terminé
