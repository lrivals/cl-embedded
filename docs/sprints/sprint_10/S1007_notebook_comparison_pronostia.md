# S10-07 — Notebook comparaison Pronostia + notebook baseline single-task

| Champ | Valeur |
|-------|--------|
| **ID** | S10-07 |
| **Sprint** | Sprint 10 — Phase 1 Extension |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 3h (2h comparaison + 1h baseline) |
| **Dépendances** | S10-06 (6 notebooks individuels terminés), S10-04 (exp_044–049 disponibles) |
| **Fichiers cibles** | `notebooks/cl_eval/pronostia_by_condition/comparison.ipynb`, `notebooks/cl_eval/baselines/pronostia_single_task.ipynb` |

---

## Objectif

Créer deux notebooks de synthèse pour le dataset PRONOSTIA :

1. **`comparison.ipynb`** — compare les 6 modèles CL sur le scénario `by_condition` (exp_050–055)
2. **`pronostia_single_task.ipynb`** — baseline hors-CL (exp_044–049), référence de performance maximale

Ces deux notebooks sont le support principal pour la section Gap 1 du manuscrit.

---

## Notebook 1 — Comparaison 6 modèles CL

**Fichier** : `notebooks/cl_eval/pronostia_by_condition/comparison.ipynb`

### Structure des figures

```
notebooks/figures/cl_evaluation/comparison/pronostia/by_condition/
├── radar_comparison.png
├── barplot_aa_comparison.png
├── acc_matrix_grid.png
├── scatter_ram_vs_accuracy.png          ← Gap 2 (budget STM32 ≤ 64 Ko)
├── scatter_flops_vs_accuracy.png        ← coût calcul portable
└── scatter_latency_vs_accuracy.png      ← latence PC (⚠ non transférable MCU)
```

### Section 0 — En-tête

```python
MODEL_EXP_MAP = {
    "EWC":         "exp_050_ewc_pronostia_by_condition",
    "HDC":         "exp_051_hdc_pronostia_by_condition",
    "TinyOL":      "exp_052_tinyol_pronostia_by_condition",
    "KMeans":      "exp_053_kmeans_pronostia_by_condition",
    "Mahalanobis": "exp_054_mahalanobis_pronostia_by_condition",
    "DBSCAN":      "exp_055_dbscan_pronostia_by_condition",
}
TASK_NAMES = ["Condition 1 (1800rpm, 4000N)", "Condition 2 (1650rpm, 4200N)", "Condition 3 (1500rpm, 5000N)"]
FIGURES_DIR = Path("../../figures/cl_evaluation/comparison/pronostia/by_condition/")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
```

### Section 1 — Chargement tous modèles

```python
import json
from pathlib import Path

all_metrics = {}
acc_matrices = {}

for model_name, exp_id in MODEL_EXP_MAP.items():
    results_path = Path(f"../../experiments/{exp_id}/results/metrics_cl.json")
    if results_path.exists():
        with open(results_path) as f:
            all_metrics[model_name] = json.load(f)
        acc_matrices[model_name] = np.array(all_metrics[model_name]["acc_matrix"])
    else:
        # Fallback mock pour les expériences non encore exécutées
        print(f"⚠️ {exp_id} non disponible — utilisation de mock")
        all_metrics[model_name] = {"aa": 0.0, "af": 0.0, "bwt": 0.0,
                                    "ram_peak_bytes": 0, "n_params": 0,
                                    "inference_latency_ms": 0.0}
        acc_matrices[model_name] = np.zeros((3, 3))
```

### Section 2 — Radar multi-critères (6 modèles)

```python
from src.evaluation.plots import plot_radar_comparison
# Axes : AA, 1-AF, AUC-ROC, 1-(RAM/65536), 1-(latency/100)
save_figure(fig, FIGURES_DIR / "radar_comparison.png")
```

### Section 3 — Barplot AA par modèle

```python
from src.evaluation.plots import plot_barplot_aa_comparison
save_figure(fig, FIGURES_DIR / "barplot_aa_comparison.png")
```

### Section 4 — Grille matrices d'accuracy (6 sous-plots)

```python
from src.evaluation.plots import plot_acc_matrix_grid
# 6 sous-plots 3×3, un par modèle — même échelle de couleur
save_figure(fig, FIGURES_DIR / "acc_matrix_grid.png")
```

### Section 5 — Scatters accuracy vs. ressources (3 plots — cohérence baseline)

Les trois scatters utilisent le même `SCATTER_MARKERS` que les notebooks de comparaison Monitoring et Pump pour permettre la comparaison inter-datasets.

```python
SCATTER_MARKERS = {
    "EWC":         {"marker": "o", "color": "#1f77b4"},
    "HDC":         {"marker": "s", "color": "#ff7f0e"},
    "TinyOL":      {"marker": "^", "color": "#2ca02c"},
    "KMeans":      {"marker": "D", "color": "#d62728"},
    "Mahalanobis": {"marker": "P", "color": "#9467bd"},
    "DBSCAN":      {"marker": "X", "color": "#8c564b"},
}

# scatter_ram_vs_accuracy.png — Gap 2 : budget STM32 ≤ 64 Ko
# scatter_flops_vs_accuracy.png — coût calcul (MACs via src.evaluation.compute_macs)
# scatter_latency_vs_accuracy.png — latence PC (⚠ non transférable MCU)
```

### Section 6 — Tableau comparatif complet

```python
summary_df = pd.DataFrame([
    {
        "Modèle": model,
        "AA": f"{m['aa']:.4f}",
        "AF": f"{m['af']:.4f}",
        "BWT": f"{m['bwt']:.4f}",
        "RAM peak (Ko)": f"{m['ram_peak_bytes'] / 1024:.1f}",
        "Latence (ms)": f"{m['inference_latency_ms']:.2f}",
        "N params": m["n_params"],
        "RAM ≤ 64Ko": "✅" if m["ram_peak_bytes"] <= 65536 else "❌",
    }
    for model, m in all_metrics.items()
])
```

### Section 7 — Discussion Gap 1

```markdown
## Synthèse Gap 1 — Validation sur données industrielles réelles (FEMTO PRONOSTIA)

Le dataset FEMTO PRONOSTIA (IEEE PHM 2012) est le premier benchmark **académique reconnu**
utilisé dans ce projet. Contrairement aux datasets Kaggle (Equipment Monitoring, Pump Maintenance),
il fournit des données d'accélérométrie réelles avec des trajectoires de dégradation mesurées
jusqu'à la défaillance effective.

**`FIXME(gap1)` → ✅ Résolu** : exp_050–055 constituent la validation CL sur données industrielles réelles.

**Comparaison inter-datasets** :
| Dataset | Meilleur AA | Meilleur AF | Source |
|---------|------------|------------|--------|
| Monitoring (by_equipment) | — | — | exp_003–exp_010 |
| Pump (by_pump_id) | — | — | exp_012–exp_021 |
| **PRONOSTIA (by_condition)** | **{best_aa:.4f}** | **{best_af:.4f}** | exp_050–055 |
```

---

## Notebook 2 — Baseline Single-Task

**Fichier** : `notebooks/cl_eval/baselines/pronostia_single_task.ipynb`

### Structure des figures

```
notebooks/figures/cl_evaluation/baseline/pronostia/single_task/
├── comparison_bar_accuracy.png
├── confusion_matrices_grid.png
├── roc_curves_all_models.png
└── scatter_ram_vs_accuracy.png
```

### Mapping expériences

```python
MODEL_EXP_MAP_BASELINE = {
    "EWC":         "exp_044_ewc_pronostia_no_split",
    "HDC":         "exp_045_hdc_pronostia_no_split",
    "TinyOL":      "exp_046_tinyol_pronostia_no_split",
    "KMeans":      "exp_047_kmeans_pronostia_no_split",
    "Mahalanobis": "exp_048_mahalanobis_pronostia_no_split",
    "DBSCAN":      "exp_049_dbscan_pronostia_no_split",
}
```

### Sections du notebook baseline

1. Chargement `metrics_single_task.json` de chaque expérience
2. Barplot Accuracy + AUC-ROC par modèle
3. Grille matrices de confusion (6 modèles)
4. Courbes ROC superposées (6 modèles)
5. Scatter RAM vs. Accuracy (même `SCATTER_MARKERS`, cohérence inter-notebooks)
6. Tableau récapitulatif complet
7. Question diagnostique : _Les performances single-task justifient-elles le scénario CL ? Si AUC-ROC ≈ 0.5, le label TTF binaire est peut-être trop bruité._

---

## Critères d'acceptation

- [x] `comparison.ipynb` exécutable sans erreur, 7 figures sauvegardées dans `comparison/pronostia/by_condition/`
- [x] `pronostia_single_task.ipynb` exécutable sans erreur, 4 figures sauvegardées dans `baseline/pronostia/single_task/`
- [x] Les scatters `scatter_ram_vs_accuracy.png` utilisent le même `SCATTER_MARKERS` que les notebooks Monitoring et Pump
- [x] La section Discussion Gap 1 dans `comparison.ipynb` référence exp_050–055 avec `FIXME(gap1)` → ✅
- [x] Fallback mock fonctionnel (toutes les expériences sont disponibles)

---

## Questions ouvertes

- `TODO(arnaud)` : La comparaison inter-datasets dans la Section 7 (Monitoring vs. Pump vs. PRONOSTIA) est-elle à inclure dans ce notebook ou dans un notebook de synthèse Phase 1 séparé ?
- `TODO(fred)` : Les résultats PRONOSTIA single-task (exp_044–049) sont-ils suffisants pour une démonstration industrielle chez Edge Spectrum, ou faut-il inclure des métriques de déploiement (throughput, latence MCU) ?

---

**Complété le** : 2026-04-24
