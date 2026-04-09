# S5-16 — Évaluation méthodes non-supervisées : Battery RUL + Pronostia

| Champ | Valeur |
|-------|--------|
| **ID** | S5-16 |
| **Sprint** | Sprint 5 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 5h |
| **Dépendances** | S5-15 (loaders Battery RUL + Pronostia), S5-02 à S5-14 (5 détecteurs implémentés), S5-05 (`train_unsupervised.py`) |
| **Fichiers cibles** | `scripts/train_unsupervised.py`, `experiments/exp_009_battery_rul/`, `experiments/exp_010_pronostia/`, `notebooks/02_baseline_comparison.ipynb` |

---

## Objectif

Exécuter et évaluer les 5 détecteurs d'anomalies non-supervisés (KMeans, KNN, PCA, Mahalanobis, DBSCAN) sur les deux nouveaux datasets introduits en S5-15, dans le même protocole expérimental que les expériences exp_005 à exp_008 (Dataset 2 — Equipment Monitoring).

**Critères de succès** :
- `python scripts/train_unsupervised.py --dataset battery_rul --model all` complète sans erreur
- `python scripts/train_unsupervised.py --dataset pronostia --model all` complète sans erreur
- `experiments/exp_009_battery_rul/results.json` contient AA/AF/BWT/AUROC/RAM pour les 5 modèles
- `experiments/exp_010_pronostia/results.json` identique pour Pronostia
- RAM peak ≤ 64 Ko pour chaque modèle (`FIXME(gap2)` si dépassement)
- Notebook comparatif mis à jour avec les 2 nouveaux datasets

---

## Tâches

| ID | Tâche | Priorité | Fichier cible | Durée est. | Dépendances |
|----|-------|:---:|---------------|:---:|-------------|
| S5-16a | Étendre `scripts/train_unsupervised.py` — ajouter `battery_rul` et `pronostia` comme valeurs valides de `--dataset` (via dispatch sur le loader), adapter le formatage des métriques | 🔴 | `scripts/train_unsupervised.py` | 1h | S5-15b, S5-15c |
| S5-16b | Expérience exp_009 — 5 détecteurs × Battery RUL, 3 tâches temporelles, métriques AA/AF/BWT/AUROC/RAM, `config_snapshot.yaml` | 🔴 | `experiments/exp_009_battery_rul/` | 1.5h | S5-16a |
| S5-16c | Expérience exp_010 — 5 détecteurs × Pronostia Learning_set (3 conditions), métriques identiques, `config_snapshot.yaml` | 🔴 | `experiments/exp_010_pronostia/` | 1.5h | S5-16a |
| S5-16d | Mise à jour `notebooks/02_baseline_comparison.ipynb` — nouvelles colonnes Battery RUL et Pronostia dans les tableaux AA/AF/BWT/AUROC/RAM, graphiques de comparaison cross-dataset | 🟡 | `notebooks/02_baseline_comparison.ipynb` | 1h | S5-16b, S5-16c |

**Livrable** : `exp_009` et `exp_010` résultats enregistrés, notebook comparatif étendu à 4 datasets (Monitoring, Pump, Battery RUL, Pronostia).

---

## Notes d'implémentation

### Extension de `scripts/train_unsupervised.py` (S5-16a)

Localiser la fonction `load_dataset()` (ou le bloc `argparse` + dispatch équivalent) et ajouter :

```python
# scripts/train_unsupervised.py — dans le dispatcher de dataset
from src.data.battery_rul_dataset import get_battery_dataloaders
from src.data.pronostia_dataset import preprocess_to_csv, get_pronostia_dataloaders

def load_dataset(args, cfg):
    if args.dataset == "monitoring":
        return get_cl_dataloaders(cfg)
    elif args.dataset == "pump":
        return get_pump_dataloaders(cfg)
    elif args.dataset == "battery_rul":
        return get_battery_dataloaders(cfg)
    elif args.dataset == "pronostia":
        preprocess_to_csv()   # no-op si déjà préprocessé
        return get_pronostia_dataloaders(cfg)
    else:
        raise ValueError(f"Dataset inconnu : {args.dataset}. "
                         f"Valeurs valides : monitoring, pump, battery_rul, pronostia")
```

Les loaders retournent une liste de 3 dicts `{task_id, train_loader, val_loader, n_train, n_val}` — interface identique aux datasets existants → **zéro modification du reste du script**.

### Structure des expériences (pattern existant)

```
experiments/exp_009_battery_rul/
├── config_snapshot.yaml        ← copie exacte de unsupervised_config.yaml au moment du run
├── results.json                ← métriques AA/AF/BWT/AUROC/RAM par modèle
└── checkpoints/
    ├── kmeans_task0.pkl
    ├── kmeans_task1.pkl
    ├── kmeans_task2.pkl
    ├── knn_task{0,1,2}.pkl
    ├── pca_task{0,1,2}.pkl
    ├── mahalanobis_task{0,1,2}.pkl
    └── dbscan_task{0,1,2}.pkl

experiments/exp_010_pronostia/
├── config_snapshot.yaml
├── results.json
└── checkpoints/
    └── ...
```

### Métriques attendues dans `results.json`

```json
{
  "dataset": "battery_rul",
  "exp_id": "exp_009",
  "timestamp": "2026-04-XX",
  "models": {
    "kmeans": {
      "acc_final": 0.XX,
      "avg_forgetting": 0.XX,
      "backward_transfer": 0.XX,
      "auroc": 0.XX,
      "ram_peak_bytes": XXXX,
      "n_params": XX,
      "inference_latency_ms": 0.XX
    },
    "knn": { ... },
    "pca": { ... },
    "mahalanobis": { ... },
    "dbscan": { ... }
  }
}
```

### Paramètres config à ajouter dans `configs/unsupervised_config.yaml`

```yaml
# Section à ajouter pour les nouveaux datasets

battery_rul:
  anomaly_threshold: 200       # RUL ≤ 200 → faulty=1
  n_tasks: 3
  val_ratio: 0.2
  # Les paramètres par modèle (kmeans.k_min, knn.n_neighbors, etc.) sont hérités
  # des sections existantes — pas de duplication

pronostia:
  anomaly_last_pct: 0.20       # last 20% fenêtres d'un bearing = faulty=1
  n_tasks: 3
  val_ratio: 0.2
  # Idem — paramètres modèles hérités des sections existantes
```

### Commandes d'exécution

```bash
# exp_009 — Battery RUL
python scripts/train_unsupervised.py \
  --config configs/unsupervised_config.yaml \
  --dataset battery_rul \
  --model all \
  --exp_id exp_009_battery_rul

# exp_010 — Pronostia (préprocessing automatique si nécessaire)
python scripts/train_unsupervised.py \
  --config configs/unsupervised_config.yaml \
  --dataset pronostia \
  --model all \
  --exp_id exp_010_pronostia
```

---

## Analyse des résultats attendus

### Hypothèses préliminaires

| Dataset | Modèle attendu le plus performant | Raison |
|---------|-----------------------------------|--------|
| Battery RUL | Mahalanobis | Dégradation monotone → distribution shift régulier, bien capturé par μ/Σ |
| Pronostia | KNN ou PCA | Signal vibratoire multi-modal → reconstruction error (PCA) ou voisinage local (KNN) mieux adaptés |

### Points de vigilance

- **Battery RUL — déséquilibre** : avec `ANOMALY_THRESHOLD=200`, environ 18% des cycles = anomalie. Surveiller precision/recall en plus de l'accuracy.
- **Pronostia — taille des tâches** : chaque condition contient des bearings de durée très variable (de 1h à 7h). L'AF et BWT peuvent être bruités par ce déséquilibre inter-tâches.
- **RAM — Pronostia avec N_FEATURES=12** : vérifier que DBSCAN avec stratégie `accumulate` ne dépasse pas 64 Ko sur les 3 tâches cumulées.

---

## Questions ouvertes

- `TODO(arnaud)` : les hyperparamètres (epsilon DBSCAN, k pour KMeans, percentile de seuil) ont été calibrés sur Dataset 2 (N_FEATURES=4). Faut-il re-calibrer pour Battery RUL (N_FEATURES=7) et Pronostia (N_FEATURES=12) ? Si oui, ajouter une sous-tâche de tuning dans exp_009/exp_010.
- `TODO(arnaud)` : comparer les résultats Battery RUL avec un baseline supervisé simple (régression linéaire sur RUL) pour quantifier la perte due au scénario non-supervisé — justification Gap 1.
- `TODO(fred)` : Pronostia est un dataset industriel réel (tests sur banc physique à FEMTO-ST). Pertinent pour le rapport Edge Spectrum ? Les conditions 1800/1650/1500 rpm correspondent-elles à des scénarios opérationnels réalistes ?
- `FIXME(gap1)` : Pronostia est le premier dataset **réel** du projet (vs pump/monitoring/battery qui sont simulés). Documenter explicitement dans `docs/triple_gap.md` que exp_010 contribue au Gap 1.
- `FIXME(gap2)` : si `ram_peak_bytes > 65536` pour un modèle sur Pronostia (N_FEATURES=12), documenter l'overhead Python vs RAM modèle analytique et indiquer la solution MCU (ex. buffer circulaire pour stratégie accumulate).
- `FIXME(gap3)` : explorer la quantification INT8 pour Mahalanobis sur Battery RUL (Σ⁻¹ de taille 7×7 = 49 floats = 196 B @ FP32 / 49 B @ INT8). Impact sur AUROC à mesurer dans exp_009.
