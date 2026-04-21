# S10-05 — Run exp_050–055 : CL Domain-Incremental Pronostia (3 conditions)

| Champ | Valeur |
|-------|--------|
| **ID** | S10-05 |
| **Sprint** | Sprint 10 — Phase 1 Extension |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | S10-04 (baseline single-task terminée pour comparaison) |
| **Fichiers cibles** | `experiments/exp_050–055/` |

---

## Objectif

Entraîner et évaluer les 6 modèles du projet sur FEMTO PRONOSTIA en mode **CL domain-incremental** : les 3 conditions opératoires arrivent séquentiellement (Condition 1 → 2 → 3). Ce sont les expériences qui **résolvent FIXME(gap1)** dans toute la Phase 1.

**Impact scientifique** : premier résultat CL publié sur données industrielles réelles de roulements. Permet de comparer AA/AF/BWT obtenus sur PRONOSTIA vs. les datasets semi-synthétiques Kaggle.

---

## Scénario CL

```
Tâche 1 : Condition 1 — 1 800 rpm, 4 000 N (Bearing1_1 + Bearing1_2, ~3 674 fenêtres)
    ↓
Tâche 2 : Condition 2 — 1 650 rpm, 4 200 N (Bearing2_1 + Bearing2_2, ~1 708 fenêtres)
    ↓
Tâche 3 : Condition 3 — 1 500 rpm, 5 000 N (Bearing3_1 + Bearing3_2, ~2 152 fenêtres)
```

Après chaque tâche, évaluer sur **toutes les conditions vues** (protocole standard CL).

---

## Commandes d'exécution

```bash
# exp_050 — EWC by_condition
python scripts/train_ewc.py \
    --config configs/ewc_pronostia_by_condition_config.yaml \
    --exp_id exp_050

# exp_051 — HDC by_condition
python scripts/train_hdc.py \
    --config configs/hdc_pronostia_by_condition_config.yaml \
    --exp_id exp_051

# exp_052 — TinyOL by_condition
python scripts/train_tinyol.py \
    --config configs/tinyol_pronostia_by_condition_config.yaml \
    --exp_id exp_052

# exp_053 — KMeans by_condition
python scripts/train_unsupervised.py --model kmeans \
    --config configs/kmeans_pronostia_by_condition_config.yaml \
    --exp_id exp_053

# exp_054 — Mahalanobis by_condition
python scripts/train_unsupervised.py --model mahalanobis \
    --config configs/mahalanobis_pronostia_by_condition_config.yaml \
    --exp_id exp_054

# exp_055 — DBSCAN by_condition
python scripts/train_unsupervised.py --model dbscan \
    --config configs/dbscan_pronostia_by_condition_config.yaml \
    --exp_id exp_055
```

---

## Structure de sortie attendue

```
experiments/
├── exp_050_ewc_pronostia_by_condition/
│   ├── config_snapshot.yaml
│   └── results/
│       └── metrics_cl.json
├── exp_051_hdc_pronostia_by_condition/
│   └── ...
├── exp_052_tinyol_pronostia_by_condition/
│   └── ...
├── exp_053_kmeans_pronostia_by_condition/
│   └── ...
├── exp_054_mahalanobis_pronostia_by_condition/
│   └── ...
└── exp_055_dbscan_pronostia_by_condition/
    └── ...
```

### Format `metrics_cl.json`

```json
{
  "exp_id": "exp_050_ewc_pronostia_by_condition",
  "model": "ewc",
  "dataset": "pronostia",
  "scenario": "by_condition",
  "tasks": ["condition_1", "condition_2", "condition_3"],
  "acc_matrix": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
  "aa": 0.0,
  "af": 0.0,
  "bwt": 0.0,
  "ram_peak_bytes": 0,
  "inference_latency_ms": 0.0,
  "n_params": 0
}
```

---

## Métriques CL à collecter

| Métrique | Formule | Module |
|---------|---------|--------|
| `acc_matrix` | Accuracy sur tâche j après entraînement sur tâche i | `evaluation/metrics.py` |
| `aa` (Average Accuracy) | Moyenne sur la diagonale de acc_matrix | `evaluation/metrics.py` |
| `af` (Average Forgetting) | Chute moyenne peak → fin par tâche | `evaluation/metrics.py` |
| `bwt` (Backward Transfer) | Impact tâches futures sur tâches passées | `evaluation/metrics.py` |
| `ram_peak_bytes` | RAM max mesurée (tracemalloc) | `evaluation/memory_profiler.py` |
| `inference_latency_ms` | Latence forward pass (100 runs) | `evaluation/memory_profiler.py` |

---

## Résolution des FIXME(gap1)

Une fois ces expériences complétées, mettre à jour les fichiers suivants :

```markdown
<!-- Dans roadmap_phase1.md, remplacer : -->
FIXME(gap1) : valider sur données industrielles réelles

<!-- Par : -->
✅ Gap 1 résolu : exp_050–055 (FEMTO PRONOSTIA by_condition)
```

Notebooks concernés à mettre à jour :
- `notebooks/cl_eval/monitoring_*/comparison.ipynb` — section Discussion
- `notebooks/cl_eval/pump_*/comparison.ipynb` — section Discussion
- `notebooks/03_cl_evaluation.ipynb` — conclusion générale

---

## Critères d'acceptation

- [ ] 6 dossiers `experiments/exp_050–055/` créés avec `config_snapshot.yaml`
- [ ] 6 fichiers `metrics_cl.json` présents avec `acc_matrix` (3×3) non vide
- [ ] Les métriques `aa`, `af`, `bwt` sont calculées et non nulles
- [ ] `ram_peak_bytes` ≤ 65 536 pour tous les modèles (contrainte STM32N6)
- [ ] `roadmap_phase1.md` : S10-05 marqué ✅ Impl + Doc + Exec
- [ ] Tous les `FIXME(gap1)` dans `roadmap_phase1.md` pointent vers exp_050–055

---

## Questions ouvertes

- `TODO(arnaud)` : Faut-il inclure une comparaison directe des métriques CL (AA/AF/BWT) entre les 3 datasets (Monitoring, Pump, PRONOSTIA) dans le manuscrit ? Si oui, une table de synthèse dans `roadmap_phase1.md` serait utile.
- `TODO(dorra)` : L'oubli catastrophique attendu est-il plus fort sur PRONOSTIA que sur Monitoring ? Les distributions des conditions opératoires sont-elles plus séparées que les équipements du Dataset 2 ?
- `TODO(fred)` : Les conditions opératoires de PRONOSTIA (Condition 1/2/3 à vitesses et charges différentes) correspondent-elles à des scénarios réels dans les installations Edge Spectrum ?
- `FIXME(gap1)` : Ces expériences constituent le **premier résultat CL sur données industrielles réelles** du projet. S'assurer que les résultats sont reproductibles (seed fixé, `config_snapshot.yaml` présent) avant de les citer dans le manuscrit.

---

**Complété le** : _(à renseigner)_
