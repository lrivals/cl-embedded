# S12-05 — Expériences exp_074–079 (CL by_fault_type)

| Champ | Valeur |
|-------|--------|
| **ID** | S12-05 |
| **Sprint** | Sprint 12 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | S12-04 (exp_068–073 terminées, baseline établie) |
| **Fichiers cibles** | `experiments/exp_074/` à `experiments/exp_079/` |

---

## Objectif

Évaluer les six modèles dans le **premier scénario domain-incremental CWRU** : les données arrivent séquentiellement par type de défaut (Ball → Inner Race → Outer Race). Ce scénario teste la capacité des modèles à s'adapter à un nouveau type de défaut sans oublier les types précédents — cas représentatif d'un déploiement embarqué progressif.

Les métriques AA/AF/BWT issues de ces expériences alimenteront `FIXME(gap1)` : la table de synthèse cross-dataset CWRU vs PRONOSTIA du manuscrit.

---

## Scénario by_fault_type — rappel des 3 tâches

| Tâche | Domaine | Fichiers MAT | Fenêtres approx. | Normal inclus |
|-------|---------|-------------|-----------------|---------------|
| Task 1 | Ball Fault | B007 + B014 + B021 | ~690 défauts | ~230 normaux |
| Task 2 | Inner Race Fault | IR007 + IR014 + IR021 | ~690 défauts | ~230 normaux |
| Task 3 | Outer Race Fault | OR007 + OR014 + OR021 | ~230 défauts | ~230 normaux |

> Ordre figé : Ball → IR → OR. Correspond à une progression de complexité (Ball = défaut le plus commun).

---

## Table des expériences

| Exp | Modèle | Scénario | Config | Script | Statut |
|-----|--------|----------|--------|--------|--------|
| exp_074 | EWC | CWRU by_fault_type | `cwru_by_fault_config.yaml` | `scripts/train_ewc.py` | ✅ |
| exp_075 | HDC | CWRU by_fault_type | `cwru_by_fault_config.yaml` | `scripts/train_hdc.py` | ✅ |
| exp_076 | TinyOL | CWRU by_fault_type | `cwru_by_fault_config.yaml` | `scripts/train_tinyol.py` | ✅ |
| exp_077 | KMeans | CWRU by_fault_type | `cwru_by_fault_config.yaml` | `scripts/train_kmeans.py` | ✅ |
| exp_078 | Mahalanobis | CWRU by_fault_type | `cwru_by_fault_config.yaml` | `scripts/train_mahalanobis.py` | ✅ |
| exp_079 | DBSCAN | CWRU by_fault_type | `cwru_by_fault_config.yaml` | `scripts/train_dbscan.py` | ✅ |

---

## Commandes de lancement

```bash
python scripts/train_ewc.py         --config configs/cwru_by_fault_config.yaml --exp_id exp_074
python scripts/train_hdc.py         --config configs/cwru_by_fault_config.yaml --exp_id exp_075
python scripts/train_tinyol.py      --config configs/cwru_by_fault_config.yaml --exp_id exp_076
python scripts/train_kmeans.py      --config configs/cwru_by_fault_config.yaml --exp_id exp_077
python scripts/train_mahalanobis.py --config configs/cwru_by_fault_config.yaml --exp_id exp_078
python scripts/train_dbscan.py      --config configs/cwru_by_fault_config.yaml --exp_id exp_079
```

---

## Outputs attendus

Pour chaque `experiments/exp_07X/` :

```
experiments/exp_074_ewc_cwru_by_fault_type/
├── config_snapshot.yaml
└── results/
    └── metrics_cl.json           # métriques CL obligatoires
```

### Métriques obligatoires dans `metrics_cl.json`

| Métrique | Description | Module |
|---------|-------------|--------|
| `acc_final` | Accuracy toutes tâches après entraînement complet | `evaluation/metrics.py` |
| `avg_forgetting` (AF) | Chute moyenne d'accuracy entre pic et fin | `evaluation/metrics.py` |
| `backward_transfer` (BWT) | Impact de l'apprentissage futur sur tâches passées | `evaluation/metrics.py` |
| `per_task_acc` | `[acc_task1, acc_task2, acc_task3]` en fin de séquence | `evaluation/metrics.py` |
| `ram_peak_bytes` | RAM max mesurée via `tracemalloc` | `evaluation/memory_profiler.py` |
| `inference_latency_ms` | Latence forward pass (moyenne 100 runs) | `evaluation/memory_profiler.py` |
| `n_params` | Nombre de paramètres entraînables | automatique |

---

## Lien scientifique

`FIXME(gap1)` — Ces résultats (AA/AF/BWT sur données réelles de roulements) doivent être croisés avec exp_050–055 (PRONOSTIA by_location) pour la table comparative Gap 1 du manuscrit. Cohérence attendue : les deux datasets réels de roulements devraient montrer des tendances similaires sur l'oubli catastrophique.

---

## Critères d'acceptation

- [x] 6 dossiers `experiments/exp_074` à `experiments/exp_079` créés
- [x] `metrics_cl.json` présent dans chaque `results/`, avec `avg_forgetting` et `backward_transfer`
- [x] `per_task_acc` est une liste de 3 valeurs (une par tâche)
- [x] `ram_peak_bytes` ≤ 65 536 pour tous les modèles
- [x] `config_snapshot.yaml` présent (reproductibilité)

## Statut

✅ Terminé
