# S2321–S2323 — Consolidation : tableau comparatif 4+ datasets + figures manuscrit

| Champ | Valeur |
|-------|--------|
| **Sprint** | 23 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Terminé (2026-06-02) |
| **Durée estimée** | 2h + 3h + 2h = 7h |
| **Dépendances** | Sprint 23 O2–O4 ✅ (experiments exp_S23_01–06 + exp_S23_INT8 produits), Sprint 21 ✅ (comparison_sprint21.json), Sprint 22 ✅ (exp_S22_* produits) |
| **Fichiers cibles** | `scripts/generate_comparison_sprint23.py`, `experiments/comparison_sprint23.json`, `notebooks/board_benchmark_all_datasets.ipynb`, `docs/figures/gap1_gap2_summary.png` |
| **Référence** | `scripts/compare_experiments.py` (pattern agrégation), `experiments/comparison_sprint21.json` (format de sortie) |

## Résultats d'implémentation (2026-06-02)

- **S2321** ✅ `scripts/generate_comparison_sprint23.py` — 61 enregistrements chargés depuis 27 répertoires (Sprint 18–23). JSON produit : `experiments/comparison_sprint23.json` (5 datasets × 5 modèles × 2 plateformes).
- **S2322** ✅ `notebooks/board_benchmark_all_datasets.ipynb` — heatmap acc_final, barplot latence log, Pareto, tableau INT8 vs FP32.
- **S2323** ✅ `docs/figures/gap1_gap2_summary.png` — 2083×915 px, 150 DPI. Figures additionnelles : `gap1_gap2_heatmap_acc.png`, `gap2_latency_board.png`, `pareto_acc_forgetting.png`.

**Note** : Les 6 expériences board Sprint 23 (exp_S23_01–06) sont `pending` — expériences à exécuter sur NUCLEO-F439ZI. La figure affiche les latences PC en proxy (hachures) jusqu'à l'exécution board.

---

## Contexte

Sprint 23 produit 7 nouvelles expériences board (exp_S23_01–06 + INT8). Combined avec les résultats Sprints 21–22 (CWRU, Monitoring, Pronostia), cela donne **5 datasets × 4 modèles × PC+board** — la table complète pour le chapitre "Résultats" du manuscrit.

Le script `generate_comparison_sprint23.py` étend `compare_experiments.py` (qui couvrait 3 datasets) en ajoutant CMAPSS et Paderborn, et en incluant les résultats board (pas seulement PC).

---

## S2321 — `scripts/generate_comparison_sprint23.py`

### Fonctionnalités

1. Charger tous les `results.json` de `experiments/exp_S21_*`, `exp_S22_*`, `exp_S23_*`
2. Agréger par `(dataset, model, platform)` en 3 dimensions : `dataset × model × {pc, board}`
3. Produire `experiments/comparison_sprint23.json` au format hiérarchique

### Structure du script

```python
"""
generate_comparison_sprint23.py — Tableau comparatif final 5 datasets × 4 modèles.

Extension de compare_experiments.py (Sprint 21) avec :
  - 2 nouveaux datasets : CMAPSS, Paderborn
  - 4 modèles : mahalanobis, ewc, tinyol, hdc
  - 2 plateformes : pc, nucleo_f439zi (board)
  - INT8 comme variante de ewc

Usage :
    python scripts/generate_comparison_sprint23.py \
        --output experiments/comparison_sprint23.json
"""

_DATASETS = ["cwru", "monitoring", "pronostia", "cmapss", "paderborn"]
_MODELS   = ["mahalanobis", "ewc", "ewc_int8", "tinyol", "hdc"]
_PLATFORMS = ["pc", "nucleo_f439zi"]

# Répertoires d'expériences à charger (ordre chronologique)
_EXP_DIRS = [
    # Sprint 18-19 (CWRU, Monitoring, Pronostia PC)
    "experiments/exp_110",
    "experiments/exp_111",
    # Sprint 21 (board Monitoring, CWRU, Pronostia)
    "experiments/exp_S21_01",
    "experiments/exp_S21_02",
    "experiments/exp_S21_03",
    "experiments/exp_S21_04",
    # Sprint 22 (CMAPSS PC, Paderborn PC, INT8 Python)
    "experiments/exp_S22_cmapss_base",
    "experiments/exp_S22_paderborn_base",
    # Sprint 23 (CMAPSS board, Paderborn board, HDC board, INT8 board)
    "experiments/exp_S23_01",
    "experiments/exp_S23_02",
    "experiments/exp_S23_03",
    "experiments/exp_S23_04",
    "experiments/exp_S23_05",
    "experiments/exp_S23_06",
    "experiments/exp_S23_INT8",
    "experiments/exp_S23_benchmark",
]
```

### Format de sortie `experiments/comparison_sprint23.json`

```json
{
  "metadata": {
    "sprint": 23,
    "date": "2026-07-05",
    "datasets": ["cwru", "monitoring", "pronostia", "cmapss", "paderborn"],
    "models": ["mahalanobis", "ewc", "ewc_int8", "tinyol", "hdc"],
    "platforms": ["pc", "nucleo_f439zi"],
    "gap2_budget_bytes": 65536
  },
  "results": {
    "cmapss": {
      "ewc": {
        "pc":  {"acc_final": 0.78, "avg_forgetting": 0.05, "auroc": 0.82, "ram_peak_bytes": 12800},
        "board": {"acc_final": 0.76, "latency_ms": 0.42, "ram_peak_bytes": 9728, "gap2_latency_compliant": true}
      },
      "hdc": {
        "pc":  {"acc_final": 0.71, "auroc": 0.74},
        "board": {"acc_final": 0.68, "latency_ms": 1.1, "ram_peak_bytes": 28312, "gap2_latency_compliant": true,
                  "note": "premier test HDC réel MCU — Gap 1"}
      }
    },
    "paderborn": { "...": "..." },
    "monitoring": { "...": "..." },
    "cwru": { "...": "..." },
    "pronostia": { "...": "..." }
  },
  "gap_summary": {
    "gap1_datasets": ["cwru", "monitoring", "pronostia", "cmapss", "paderborn"],
    "gap2_compliant_experiments": "à calculer — tous les exp board",
    "gap3_int8_met": "à mesurer — exp_S23_INT8"
  }
}
```

### Vérification

```bash
python scripts/generate_comparison_sprint23.py
python -c "
import json
d = json.load(open('experiments/comparison_sprint23.json'))
datasets = list(d['results'].keys())
print(f'{len(datasets)} datasets : {datasets}')
assert len(datasets) >= 4, 'Attendu >= 4 datasets'
print('comparison_sprint23.json OK')
"
```

---

## S2322 — Notebook `notebooks/board_benchmark_all_datasets.ipynb`

### Sections requises

1. **Chargement** : lire `experiments/comparison_sprint23.json`

2. **Heatmap acc_final (board)** : datasets × modèles, valeurs manquantes en gris

   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt

   # Construire le DataFrame 5 datasets × 4 modèles (acc_final board)
   df_acc = pd.DataFrame(acc_board_matrix,
       index=["CWRU", "Monitoring", "Pronostia", "CMAPSS", "Paderborn"],
       columns=["Maha", "EWC", "TinyOL", "HDC"])

   fig, ax = plt.subplots(figsize=(8, 5))
   sns.heatmap(df_acc, annot=True, fmt=".2f", cmap="YlOrRd",
               vmin=0.5, vmax=1.0, ax=ax, linewidths=0.5)
   ax.set_title("acc_final board — 5 datasets × 4 modèles (NUCLEO-F439ZI)")
   fig.savefig("docs/figures/gap1_gap2_heatmap_acc.png", dpi=150, bbox_inches="tight")
   ```

3. **Barplot latence board** : 4 modèles × 5 datasets, ligne de référence 100 ms

   ```python
   fig, ax = plt.subplots(figsize=(10, 5))
   # ... barplot grouped by dataset
   ax.axhline(y=100, color='red', linestyle='--', label='Gap 2 : 100 ms')
   ax.set_yscale('log')
   ax.set_ylabel("Latence forward (ms) — échelle log")
   fig.savefig("docs/figures/gap2_latency_board.png", dpi=150, bbox_inches="tight")
   ```

4. **Tableau forgetting vs acc_final** : courbe Pareto (acc_final, -avg_forgetting) par modèle

5. **Tableau INT8 vs FP32** : extrait de `exp_S23_INT8`, présenter la conclusion Gap 3

6. **Cellule obligatoire** :
   ```python
   print("Tableau comparatif final — Sprint 23")
   print("5 datasets × 4 modèles × PC + board NUCLEO-F439ZI")
   print("Contributions Gap 1 + Gap 2 validées")
   print(f"Gap 3 : voir notebooks/gap3_int8_board_results.ipynb")
   ```

---

## S2323 — Figure `docs/figures/gap1_gap2_summary.png`

> Produit par le notebook S2322 — ce n'est pas un fichier à créer manuellement.

### Spécifications de la figure

- **Format** : PNG, 150 DPI, 12 × 8 inches
- **2 sous-figures** côte à côte :
  - Gauche : Heatmap acc_final board (5×4)
  - Droite : Barplot latence board (4 modèles × 5 datasets) avec ligne 100 ms
- **Titre global** : "CL-Embedded : Gap 1 (5 datasets) + Gap 2 (latence ≤ 100 ms) — NUCLEO-F439ZI"
- **Légende** : modèles en couleurs consistantes avec le reste du manuscrit

```python
# Cellule finale du notebook S2322
fig, (ax_heat, ax_bar) = plt.subplots(1, 2, figsize=(14, 6))

# ... heatmap sur ax_heat
# ... barplot sur ax_bar

fig.suptitle("CL-Embedded — Gap 1 + Gap 2 — Sprint 23", fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig("docs/figures/gap1_gap2_summary.png", dpi=150, bbox_inches="tight")
print("Figure sauvegardée : docs/figures/gap1_gap2_summary.png")
```

---

## Vérification end-to-end

```bash
# 1. Générer le JSON de comparaison
python scripts/generate_comparison_sprint23.py
python -c "
import json
d = json.load(open('experiments/comparison_sprint23.json'))
print('Datasets couverts :', list(d['results'].keys()))
print('Gap 2 compliant :', d['gap_summary']['gap2_compliant_experiments'])
"

# 2. Exécuter le notebook
jupyter nbconvert --to notebook --execute \
    notebooks/board_benchmark_all_datasets.ipynb \
    --output notebooks/board_benchmark_all_datasets_executed.ipynb

# 3. Vérifier la figure
ls -lh docs/figures/gap1_gap2_summary.png
python -c "
from PIL import Image
img = Image.open('docs/figures/gap1_gap2_summary.png')
print('Taille:', img.size, '— format OK' if img.size[0] > 1000 else 'trop petit')
"
```

---

## Questions ouvertes

- `TODO(arnaud)` : 5 datasets suffisent-ils pour le chapitre "Résultats Gap 1" ? Y a-t-il une règle implicite de diversité minimale (domaine métier différent, type de signal différent) ?
- `TODO(arnaud)` : Le tableau `comparison_sprint23.json` doit-il inclure les métriques PC également, ou uniquement les résultats board pour Gap 2 ? Les résultats PC serviraient de référence haute pour situer les résultats board.
- `FIXME(gap1)` : La figure `gap1_gap2_summary.png` est le livrable visuel central du manuscrit. Valider le format avec Arnaud avant soumission (couleurs daltonisme-safe ?).
