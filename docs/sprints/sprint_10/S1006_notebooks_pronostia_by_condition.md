# S10-06 — Notebooks individuels `pronostia_by_condition/` — 6 modèles

| Champ | Valeur |
|-------|--------|
| **ID** | S10-06 |
| **Sprint** | Sprint 10 — Phase 1 Extension |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 6h (1h par modèle) |
| **Dépendances** | S10-05 (exp_050–055 exécutées) |
| **Fichiers cibles** | `notebooks/cl_eval/pronostia_by_condition/{ewc,hdc,tinyol,kmeans,mahalanobis,dbscan}.ipynb` |

---

## Objectif

Créer 6 notebooks d'évaluation CL pour le scénario **domain-incremental par condition opératoire** (3 tâches : Condition 1 → 2 → 3) sur le dataset FEMTO PRONOSTIA. Ces notebooks documentent les résultats qui résolvent `FIXME(gap1)`.

Chaque notebook est autonome : il charge les résultats d'une expérience (`exp_050` à `exp_055`), génère les figures CL standard et produit un tableau récapitulatif.

---

## Structure cible

```
notebooks/cl_eval/pronostia_by_condition/
├── ewc.ipynb          ← exp_050
├── hdc.ipynb          ← exp_051
├── tinyol.ipynb       ← exp_052
├── kmeans.ipynb       ← exp_053
├── mahalanobis.ipynb  ← exp_054
└── dbscan.ipynb       ← exp_055
```

---

## Structure des figures

```
notebooks/figures/cl_evaluation/
├── ewc/pronostia/by_condition/
│   ├── acc_matrix.png
│   ├── forgetting_curve.png
│   ├── confusion_matrix_grid.png
│   ├── roc_curves.png
│   └── feature_space_pca.png
├── hdc/pronostia/by_condition/
│   └── (même 5 figures)
├── tinyol/pronostia/by_condition/
│   └── (même 5 figures)
├── kmeans/pronostia/by_condition/
│   └── (même 5 figures)
├── mahalanobis/pronostia/by_condition/
│   └── (même 5 figures)
└── dbscan/pronostia/by_condition/
    └── (même 5 figures)
```

---

## Contenu type d'un notebook individuel

### Section 0 — En-tête

```python
MODEL_NAME = "EWC"          # à adapter par modèle
EXP_ID     = "exp_050"      # à adapter par modèle
DATASET    = "pronostia"
SCENARIO   = "by_condition"
TASK_NAMES = ["Condition 1 (1800rpm, 4000N)", "Condition 2 (1650rpm, 4200N)", "Condition 3 (1500rpm, 5000N)"]

FIGURES_DIR = Path(f"../figures/cl_evaluation/{MODEL_NAME.lower()}/{DATASET}/{SCENARIO}/")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path(f"../../experiments/{EXP_ID}_{MODEL_NAME.lower()}_pronostia_by_condition/results/")
```

### Section 1 — Chargement des résultats

```python
import json
from pathlib import Path

with open(RESULTS_DIR / "metrics_cl.json") as f:
    metrics = json.load(f)

acc_matrix = np.array(metrics["acc_matrix"])  # shape (3, 3)
aa  = metrics["aa"]
af  = metrics["af"]
bwt = metrics["bwt"]
ram_peak_bytes = metrics["ram_peak_bytes"]
```

### Section 2 — Matrice d'accuracy

```python
from src.evaluation.plots import plot_acc_matrix

fig = plot_acc_matrix(
    acc_matrix,
    task_names=TASK_NAMES,
    title=f"Accuracy matrix — {MODEL_NAME} — Pronostia by_condition"
)
save_figure(fig, FIGURES_DIR / "acc_matrix.png")
```

### Section 3 — Courbe d'oubli

```python
# Tracer acc_matrix[:, 0] (performance sur Condition 1 après chaque tâche)
# → visualise si l'apprentissage des conditions 2/3 efface Condition 1
fig, ax = plt.subplots(figsize=(8, 4))
for t in range(acc_matrix.shape[1]):
    ax.plot(
        range(t, acc_matrix.shape[0]),
        acc_matrix[t:, t],
        marker='o',
        label=TASK_NAMES[t]
    )
ax.set_xlabel("Tâche d'entraînement courante")
ax.set_ylabel("Accuracy")
ax.set_title(f"Courbe d'oubli — {MODEL_NAME}")
ax.legend()
save_figure(fig, FIGURES_DIR / "forgetting_curve.png")
```

### Section 4 — Grille de matrices de confusion (une par condition après entraînement complet)

```python
from src.evaluation.plots import plot_confusion_matrices_grid
# 3 sous-plots : confusion matrix finale sur chaque condition
save_figure(fig, FIGURES_DIR / "confusion_matrix_grid.png")
```

### Section 5 — Courbes ROC (une par condition)

```python
from src.evaluation.plots import plot_roc_curves
# AUC-ROC plus pertinent qu'accuracy seule sur dataset déséquilibré (~10% positifs)
save_figure(fig, FIGURES_DIR / "roc_curves.png")
```

### Section 6 — Espace features (PCA 2D)

```python
from src.evaluation.plots import plot_feature_space_pca
# Points colorés par condition (1/2/3) et forme par label (0/1)
# Montre la séparabilité inter-conditions dans l'espace features appris
save_figure(fig, FIGURES_DIR / "feature_space_pca.png")
```

### Section 7 — Tableau récapitulatif

```python
summary = {
    "Modèle": MODEL_NAME,
    "Dataset": "PRONOSTIA",
    "Scénario": "by_condition (3 tâches)",
    "AA (Average Accuracy)": f"{aa:.4f}",
    "AF (Average Forgetting)": f"{af:.4f}",
    "BWT (Backward Transfer)": f"{bwt:.4f}",
    "RAM peak": f"{ram_peak_bytes / 1024:.1f} Ko",
    "Latence inf.": f"{metrics['inference_latency_ms']:.2f} ms",
    "N params": metrics["n_params"],
}
pd.DataFrame([summary])
```

### Section 8 — Discussion Gap 1

```markdown
## Discussion — Gap 1

Ces résultats sur FEMTO PRONOSTIA (données industrielles réelles) complètent les validations
sur les datasets Kaggle (Equipment Monitoring, Pump Maintenance).

**Comparaison** :
- Dataset Monitoring (by_equipment, 3 tâches) : AA = ??, AF = ??
- Dataset Pump (by_pump_id, 5 tâches) : AA = ??, AF = ??
- **Dataset PRONOSTIA (by_condition, 3 tâches) : AA = {aa:.4f}, AF = {af:.4f}** ← exp_050–055

`FIXME(gap1)` : ✅ Premier résultat CL sur données industrielles réelles de roulements —
voir `docs/roadmap_phase1.md` section Sprint 10 pour la synthèse complète.
```

---

## Spécificités PRONOSTIA vs. datasets précédents

| Aspect | Dataset 2 (Monitoring) | Dataset 3 (PRONOSTIA) |
|--------|----------------------|----------------------|
| Type de drift | Drift de domaine (type équipement) | Drift de domaine (conditions opératoires) |
| Déséquilibre classes | Équilibré (~50%) | Déséquilibré (~10% positifs) |
| Métrique principale | Accuracy | AUC-ROC + Accuracy |
| Noms des tâches | pump, turbine, compressor | Condition 1, 2, 3 |
| Taille par tâche | ~2 000–4 000 | ~1 700–3 700 fenêtres |

---

## Critères d'acceptation

- [x] 6 notebooks créés dans `notebooks/cl_eval/pronostia_by_condition/` (ewc, hdc, tinyol, kmeans, mahalanobis, dbscan)
- [x] Chaque notebook s'exécute sans erreur
- [⚠️] Figures centralisées dans `notebooks/figures/cl_evaluation/comparison/pronostia/by_condition/` (non par modèle individuel — choix d'implémentation)
- [x] Section Discussion Gap 1 présente dans chaque notebook avec `FIXME(gap1)` → ✅
- [x] Tableau récapitulatif AA/AF/BWT/RAM présent

---

## Questions ouvertes

- `TODO(arnaud)` : La section Discussion Gap 1 doit-elle inclure une référence bibliographique au protocole IEEE PHM 2012 Challenge (`@PHM2012`) pour contextualiser les résultats dans la littérature ?
- `FIXME(gap1)` : Ces notebooks constituent la validation finale de Gap 1. Vérifier que les métriques citées dans le manuscrit (section Expériences) correspondent exactement aux valeurs dans `metrics_cl.json`.

---

**Complété le** : 2026-04-24
