# Sprint 8 (Phase 1) — Notebooks individuels Dataset 1 (Pump Maintenance)

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 8 — Phase 1 Extension |
| **Priorité globale** | 🔴 Critique — notebooks support présentation encadrants |
| **Durée estimée totale** | ~28h |
| **Dépendances** | Sprint 6 terminé (exp_012–015, exp_020–021, exp_024–029 exécutées, loader temporel opérationnel) |

---

## Objectif

Créer 14 notebooks d'évaluation CL pour le Dataset 1 (Large Industrial Pump Maintenance Dataset — `Large_Industrial_Pump_Maintenance_Dataset.csv`) :
- 6 notebooks individuels pour le scénario **by_pump_id** (Pump_ID 1 → 2 → 3 → 4 → 5, 5 tâches)
- 6 notebooks individuels pour le scénario **by_temporal_window** (Q1 → Q2 → Q3 → Q4, 4 tâches de 5 000 entrées par `Operational_Hours`)
- 2 notebooks de comparaison (un par scénario)

Ces notebooks documentent les deux scénarios granulaires sur Dataset 1, complémentaires au scénario chronologique (3 tâches) déjà présent dans `03_cl_evaluation.ipynb`.

**Critère de succès** : 14 notebooks exécutables du début à la fin sans erreur, avec toutes les figures sauvegardées dans la structure `notebooks/figures/cl_evaluation/{model}/{dataset}/{task}/`.

---

## Structure cible

```
notebooks/cl_eval/pump_by_pump_id/
├── tinyol.ipynb       ← exp_012
├── ewc.ipynb          ← exp_013
├── hdc.ipynb          ← exp_014
├── kmeans.ipynb       ← exp_020
├── mahalanobis.ipynb  ← exp_015
├── dbscan.ipynb       ← exp_021
└── comparison.ipynb   ← tous les modèles

notebooks/cl_eval/pump_by_temporal_window/
├── tinyol.ipynb       ← exp_024
├── ewc.ipynb          ← exp_025
├── hdc.ipynb          ← exp_026
├── kmeans.ipynb       ← exp_028
├── mahalanobis.ipynb  ← exp_027
├── dbscan.ipynb       ← exp_029
└── comparison.ipynb   ← tous les modèles
```

---

## Structure des figures

```
notebooks/figures/cl_evaluation/
├── tinyol/pump/by_pump_id/
│   ├── acc_matrix.png
│   ├── forgetting_curve.png
│   ├── confusion_matrix_grid.png
│   ├── roc_curves.png
│   └── feature_space_pca.png
├── tinyol/pump/by_temporal_window/
│   └── (même 5 figures)
├── ewc/pump/by_pump_id/ ...
├── comparison/pump/by_pump_id/
│   ├── radar_comparison.png
│   ├── barplot_aa_comparison.png
│   ├── acc_matrix_grid.png
│   └── performance_by_pump_id_bar.png  ← plot spécifique à ce scénario
├── comparison/pump/by_temporal_window/
│   ├── radar_comparison.png
│   ├── barplot_aa_comparison.png
│   └── acc_matrix_grid.png
```

---

## Tâches

### Scénario pump_by_pump_id

| ID | Tâche | Priorité | Fichier cible | Durée est. | Dépendances |
|----|-------|:---:|---------------|:---:|-------------|
| S8-01 | Notebook TinyOL — pump_by_pump_id | 🔴 | `notebooks/cl_eval/pump_by_pump_id/tinyol.ipynb` | 2h | exp_012 (Sprint 6) |
| S8-02 | Notebook EWC — pump_by_pump_id | 🔴 | `notebooks/cl_eval/pump_by_pump_id/ewc.ipynb` | 2h | exp_013 (Sprint 6) |
| S8-03 | Notebook HDC — pump_by_pump_id | 🔴 | `notebooks/cl_eval/pump_by_pump_id/hdc.ipynb` | 2h | exp_014 (Sprint 6) |
| S8-04 | Notebook KMeans — pump_by_pump_id | 🔴 | `notebooks/cl_eval/pump_by_pump_id/kmeans.ipynb` | 2h | exp_020 (Sprint 6) |
| S8-05 | Notebook Mahalanobis — pump_by_pump_id | 🔴 | `notebooks/cl_eval/pump_by_pump_id/mahalanobis.ipynb` | 2h | exp_015 (Sprint 6) |
| S8-06 | Notebook DBSCAN — pump_by_pump_id | 🔴 | `notebooks/cl_eval/pump_by_pump_id/dbscan.ipynb` | 2h | exp_021 (Sprint 6) |
| S8-13 | Notebook Comparaison — pump_by_pump_id | 🔴 | `notebooks/cl_eval/pump_by_pump_id/comparison.ipynb` | 3h | S8-01 à S8-06 |

### Scénario pump_by_temporal_window

| ID | Tâche | Priorité | Fichier cible | Durée est. | Dépendances |
|----|-------|:---:|---------------|:---:|-------------|
| S8-07 | Notebook TinyOL — pump_by_temporal_window | 🔴 | `notebooks/cl_eval/pump_by_temporal_window/tinyol.ipynb` | 2h | exp_024 (Sprint 6) |
| S8-08 | Notebook EWC — pump_by_temporal_window | 🔴 | `notebooks/cl_eval/pump_by_temporal_window/ewc.ipynb` | 2h | exp_025 (Sprint 6) |
| S8-09 | Notebook HDC — pump_by_temporal_window | 🔴 | `notebooks/cl_eval/pump_by_temporal_window/hdc.ipynb` | 2h | exp_026 (Sprint 6) |
| S8-10 | Notebook KMeans — pump_by_temporal_window | 🔴 | `notebooks/cl_eval/pump_by_temporal_window/kmeans.ipynb` | 2h | exp_028 (Sprint 6) |
| S8-11 | Notebook Mahalanobis — pump_by_temporal_window | 🔴 | `notebooks/cl_eval/pump_by_temporal_window/mahalanobis.ipynb` | 2h | exp_027 (Sprint 6) |
| S8-12 | Notebook DBSCAN — pump_by_temporal_window | 🔴 | `notebooks/cl_eval/pump_by_temporal_window/dbscan.ipynb` | 2h | exp_029 (Sprint 6) |
| S8-14 | Notebook Comparaison — pump_by_temporal_window | 🔴 | `notebooks/cl_eval/pump_by_temporal_window/comparison.ipynb` | 3h | S8-07 à S8-12 |

---

## Spécificités Dataset 1

### Noms de tâches

**by_pump_id** :
```python
TASK_NAMES = ["Pump 1", "Pump 2", "Pump 3", "Pump 4", "Pump 5"]
```

**by_temporal_window** :
```python
TASK_NAMES = [
    "Q1 (0–5k heures)",
    "Q2 (5k–10k heures)",
    "Q3 (10k–15k heures)",
    "Q4 (15k–20k heures)"
]
```

### Features
Les notebooks Dataset 1 travaillent avec les **25 features statistiques** extraites par fenêtrage (6 stats × 4 canaux + temporal_position), et non les features brutes.

### Contexte des résultats attendus
Les expériences précédentes sur Dataset 1 (scénario chronologique 3 tâches) donnaient des performances proches du hasard (AA ≈ 0.50–0.56 pour tous les modèles). Les notebooks doivent :
1. Documenter et **expliquer** ce phénomène (backbone peu discriminant, faible drift inter-tâches)
2. Comparer si les scénarios plus granulaires (5 et 4 tâches) changent la difficulté
3. Identifier si certains Pump_ID ou quartiles temporels ont des profils plus séparables

### Sections spécifiques au scénario by_pump_id

Ajouter dans les notebooks by_pump_id une section dédiée :

#### Section 7bis — Performance par Pump_ID (barplot)
```python
from src.evaluation.plots import plot_performance_by_pump_id_bar

# Extraire la performance finale par Pump_ID depuis acc_matrix[-1, :]
pump_accuracy = {i+1: float(acc_matrix[-1, i]) for i in range(5)}
fig = plot_performance_by_pump_id_bar(
    results={MODEL_NAME: pump_accuracy},
    pump_ids=[1, 2, 3, 4, 5],
    title=f"Accuracy finale par Pump_ID — {MODEL_NAME}"
)
save_figure(fig, FIGURES_DIR / "performance_by_pump_id_bar.png")
```
Commentaire : révèle si certaines pompes sont structurellement plus difficiles à classifier.

### Sections spécifiques au scénario by_temporal_window

Ajouter dans les notebooks by_temporal_window une section dédiée :

#### Section 7bis — Évolution de la performance au fil du temps
```python
# Tracer la performance sur T1 après chaque étape d'entraînement (acc_matrix[:, 0])
# pour visualiser si le modèle "oublie" les données historiques récentes
fig, ax = plt.subplots()
ax.plot(range(len(acc_matrix)), acc_matrix[:, 0], marker='o', label='Task T1 accuracy')
ax.set_xlabel("Tâche d'entraînement courante")
ax.set_ylabel("Accuracy sur T1 (Q1: 0–5k heures)")
ax.set_title(f"Oubli catastrophique T1 au fil du temps — {MODEL_NAME}")
save_figure(fig, FIGURES_DIR / "catastrophic_forgetting_t1.png")
```
Commentaire : visualise si l'apprentissage des phases récentes efface les patterns de la phase initiale.

---

## Contenu type d'un notebook de comparaison (Dataset 1)

### Section 0 — En-tête
```markdown
# Comparaison 6 modèles — Dataset 1 Pump Maintenance — {SCÉNARIO}
Scénario : {by_pump_id (5 tâches) ou by_temporal_window (4 tâches)}
Modèles : TinyOL · EWC · HDC · KMeans · Mahalanobis · DBSCAN
```

### Section 1 — Chargement tous modèles
```python
MODEL_EXP_MAP = {
    # by_pump_id
    "TinyOL": "exp_012", "EWC": "exp_013", "HDC": "exp_014",
    "KMeans": "exp_020", "Mahalanobis": "exp_015", "DBSCAN": "exp_021",
    # ou by_temporal_window
    "TinyOL": "exp_024", "EWC": "exp_025", ...
}
```

### Section 2 — Radar multi-critères (6 modèles)

### Section 3 — Barplot AA par modèle

### Section 4 — Grille matrices d'accuracy (6 sous-plots)

### Section 5 — Barplot performance par Pump_ID (uniquement by_pump_id)
```python
# Comparaison inter-modèles : quel modèle retient le mieux chaque Pump_ID ?
from src.evaluation.plots import plot_performance_by_pump_id_bar
results_all = {
    model: {i+1: float(mat[-1, i]) for i in range(5)}
    for model, mat in acc_matrices.items()
}
fig = plot_performance_by_pump_id_bar(results_all, pump_ids=[1,2,3,4,5])
save_figure(fig, FIGURES_DIR / "performance_by_pump_id_bar.png")
```

### Section 6 — Tableau comparatif complet

### Section 7 — Discussion
Questions spécifiques à Dataset 1 :
- Pourquoi toutes les méthodes supervisées approchent-elles du hasard (~0.50) ?
- Les scénarios granulaires (5 tâches Pump_ID, 4 tâches temporelles) sont-ils plus difficiles que le scénario chronologique (3 tâches) ?
- Les méthodes non supervisées performent-elles mieux que les supervisées sur ce dataset ?
- `FIXME(gap1)` : Lien avec la validation sur FEMTO PRONOSTIA (données plus diversifiées)

---

## Critères d'acceptation

- [ ] 14 notebooks créés dans `notebooks/cl_eval/pump_*/`
- [ ] Chaque notebook s'exécute sans erreur (kernel restart + run all)
- [ ] Toutes les figures sauvegardées dans `notebooks/figures/cl_evaluation/{model}/pump/{scenario}/`
- [ ] Les notebooks by_pump_id incluent le barplot `plot_performance_by_pump_id_bar()`
- [ ] Les notebooks by_temporal_window incluent la visualisation de l'oubli sur T1
- [ ] Les notebooks de comparaison chargent bien les 6 modèles
- [ ] Chaque notebook se termine par un tableau récapitulatif AA/AF/BWT/RAM
- [ ] Le fallback mock est fonctionnel pour les expériences non encore exécutées

---

## Livrable sprint 8

14 notebooks pour Dataset 1 (7 by_pump_id + 7 by_temporal_window) prêts pour présentation aux encadrants. Réponses documentées aux questions scientifiques sur les AA ≈ 0.50 observées sur ce dataset.

---

## Questions ouvertes

- `TODO(arnaud)` : Les performances proches du hasard sur Dataset 1 sont-elles à mettre en avant comme un résultat négatif intéressant (difficulté intrinsèque du dataset), ou à minimiser dans la présentation ?
- `TODO(fred)` : Dans le contexte industriel, une accuracy de 0.50 sur 5 Pump_IDs différentes est-elle acceptable si l'oubli catastrophique est faible (AF ≈ 0.01) ? Le client préfère-t-il la stabilité à la performance brute ?
- `FIXME(gap1)` : Contraster les résultats Dataset 1 (Kaggle) avec FEMTO PRONOSTIA (données réelles INSA) dans les sections discussion.
- `TODO(arnaud)` : La comparaison chronologique (3 tâches, exp_003/009/010) vs granulaire (4-5 tâches, exp_012-029) est-elle à inclure dans les notebooks comparaison ou dans un notebook séparé ?

---

> **⚠️ Après l'implémentation de ce sprint** : exécuter tous les notebooks via "Restart Kernel & Run All Cells" et vérifier l'absence d'erreurs. Contrôler que les sous-dossiers de figures sont bien créés. Mettre à jour `docs/roadmap_phase1.md` en marquant S8-01 à S8-14 comme ✅.
