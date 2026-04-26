# S12-06 — Expériences exp_080–085 (CL by_severity)

| Champ | Valeur |
|-------|--------|
| **ID** | S12-06 |
| **Sprint** | Sprint 12 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | S12-04 (exp_068–073 terminées, baseline établie) |
| **Fichiers cibles** | `experiments/exp_080/` à `experiments/exp_085/` |

---

## Objectif

Évaluer les six modèles dans le **deuxième scénario domain-incremental CWRU** : les données arrivent séquentiellement par sévérité croissante de défaut (0.007" → 0.014" → 0.021"). Ce scénario simule une dégradation progressive du roulement — proche du cas réel embarqué où un équipement vieillit in-situ. Complémentaire du scénario by_fault_type, il teste une forme différente de drift de domaine (gradient de sévérité vs changement de type).

---

## Scénario by_severity — rappel des 3 tâches

| Tâche | Domaine | Fichiers MAT | Fenêtres approx. | Normal inclus |
|-------|---------|-------------|-----------------|---------------|
| Task 1 | Sévérité 0.007" | B007 + IR007 + OR007 | ~690 défauts | ~230 normaux |
| Task 2 | Sévérité 0.014" | B014 + IR014 + OR014 | ~690 défauts | ~230 normaux |
| Task 3 | Sévérité 0.021" | B021 + IR021 + OR021 | ~230 défauts | ~230 normaux |

> Ordre figé : 0.007" → 0.014" → 0.021". Correspond à une progression réaliste de dégradation.

---

## Table des expériences

| Exp | Modèle | Scénario | Config | Script | Statut |
|-----|--------|----------|--------|--------|--------|
| exp_080 | EWC | CWRU by_severity | `cwru_by_severity_config.yaml` | `scripts/train_ewc.py` | ✅ AA=0.952 AF=0.000 BWT=+0.007 RAM=1.1 Ko |
| exp_081 | HDC | CWRU by_severity | `cwru_by_severity_config.yaml` | `scripts/train_hdc.py` | ✅ AA=0.892 AF=0.020 BWT=-0.007 RAM=7.7 Ko |
| exp_082 | TinyOL | CWRU by_severity | `cwru_by_severity_config.yaml` | `scripts/train_tinyol.py` | ✅ AA=0.900 AF=0.000 BWT=+0.013 RAM=4.0 Ko |
| exp_083 | KMeans | CWRU by_severity | `cwru_by_severity_config.yaml` | `scripts/train_kmeans.py` | ✅ AA=0.303 AF=0.065 BWT=+0.286 RAM=5.3 Ko |
| exp_084 | Mahalanobis | CWRU by_severity | `cwru_by_severity_config.yaml` | `scripts/train_mahalanobis.py` | ✅ AA=0.394 AF=0.091 BWT=+0.396 RAM=1.6 Ko |
| exp_085 | DBSCAN | CWRU by_severity | `cwru_by_severity_config.yaml` | `scripts/train_dbscan.py` | ✅ AA=0.121 AF=0.292 BWT=-0.013 RAM=30.7 Ko |

---

## Commandes de lancement

```bash
python scripts/train_ewc.py         --config configs/cwru_by_severity_config.yaml --exp_id exp_080
python scripts/train_hdc.py         --config configs/cwru_by_severity_config.yaml --exp_id exp_081
python scripts/train_tinyol.py      --config configs/cwru_by_severity_config.yaml --exp_id exp_082
python scripts/train_kmeans.py      --config configs/cwru_by_severity_config.yaml --exp_id exp_083
python scripts/train_mahalanobis.py --config configs/cwru_by_severity_config.yaml --exp_id exp_084
python scripts/train_dbscan.py      --config configs/cwru_by_severity_config.yaml --exp_id exp_085
```

---

## Outputs attendus

Pour chaque `experiments/exp_08X/` :

```
experiments/exp_080_ewc_cwru_by_severity/
├── config_snapshot.yaml
└── results/
    └── metrics_cl.json
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

`FIXME(gap1)` — Croiser AF et BWT (by_severity) avec exp_074–079 (by_fault_type) : le scénario de dégradation progressive produit-il moins d'oubli catastrophique que le changement de type de défaut ? Ce croisement enrichit l'analyse du Triple Gap pour le manuscrit.

---

## Critères d'acceptation

- [x] 6 dossiers `experiments/exp_080` à `experiments/exp_085` créés
- [x] `metrics_cl.json` présent dans chaque `results/`, avec `avg_forgetting` et `backward_transfer`
- [x] `per_task_acc` est une liste de 3 valeurs (une par tâche)
- [x] `ram_peak_bytes` ≤ 65 536 pour tous les modèles (max : DBSCAN 30 749 B)
- [x] `config_snapshot.yaml` présent (reproductibilité)

## Statut

✅ Terminé — 26 avril 2026
