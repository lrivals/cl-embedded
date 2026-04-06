# S3-08 — Notebook CL évaluation Dataset 1

| Champ | Valeur |
|-------|--------|
| **ID** | S3-08 |
| **Sprint** | Sprint 3 — Semaine 3 (29 avril – 6 mai 2026) |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 2h |
| **Dépendances** | S3-06 (`experiments/exp_003_tinyol_dataset1/results/metrics.json` disponible) |
| **Fichier cible** | `notebooks/03_cl_evaluation.ipynb` |
| **Complété le** | — |

---

## Objectif

Créer le notebook `notebooks/03_cl_evaluation.ipynb` qui charge les résultats de `exp_003` et produit les visualisations CL pour Dataset 1 : matrice d'accuracy, courbes de forgetting par tâche, et tableau comparatif TinyOL vs baselines.

**Critère de succès** : le notebook s'exécute de bout en bout (`Run All`) sans erreur à partir des fichiers `experiments/exp_003_tinyol_dataset1/results/metrics.json` et génère toutes les visualisations.

---

## Structure du notebook

Le notebook est organisé en 6 sections :

```
1. Setup & imports
2. Chargement des résultats exp_003
3. Matrice d'accuracy CL (heatmap)
4. Analyse du forgetting par tâche
5. Comparaison TinyOL vs baselines
6. Tableau récapitulatif pour roadmap.md
```

---

## Sous-tâches (cellules notebook)

### Cellule 1 — En-tête et imports

```python
# notebooks/03_cl_evaluation.ipynb — Cellule 1
"""
Notebook : Évaluation CL — Dataset 1 (Pump Maintenance)
Sprint 3 — S3-08

Prérequis :
  - exp_003 exécutée (S3-06) → metrics.json disponible
  - exp_001 exécutée (S1-09) → pour comparaison EWC (optionnel)

Références : tinyol_spec.md §4, DeLange2021Survey
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

# Chemins
EXP_003_DIR = Path("../experiments/exp_003_tinyol_dataset1/results")
METRICS_PATH = EXP_003_DIR / "metrics.json"

assert METRICS_PATH.exists(), f"Lancer S3-06 d'abord : {METRICS_PATH} introuvable"

%matplotlib inline
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 11
```

### Cellule 2 — Chargement des résultats

```python
# Cellule 2 — Chargement metrics.json
with open(METRICS_PATH) as f:
    results = json.load(f)

print(f"Expérience : {results['exp_id']}")
print(f"Modèle     : {results['model']}")
print(f"Dataset    : {results['dataset']}")
print(f"Timestamp  : {results['timestamp']}")
print(f"Seed       : {results['seed']}")
print()
print(f"AA  (Average Accuracy)    : {results['acc_final']:.4f}")
print(f"AF  (Average Forgetting)  : {results['avg_forgetting']:.4f}")
print(f"BWT (Backward Transfer)   : {results['backward_transfer']:.4f}")
print(f"RAM peak update           : {results['ram_peak_bytes']} B")
print(f"Latence inférence         : {results['inference_latency_ms']:.3f} ms")
print(f"Params OtO                : {results['n_params_oto']}")
print(f"Params encodeur           : {results['n_params_encoder']}")
```

### Cellule 3 — Matrice d'accuracy CL (heatmap)

```python
# Cellule 3 — Heatmap matrice accuracy
acc_matrix_raw = results["acc_matrix"]
n_tasks = len(acc_matrix_raw)

# Reconstruire matrice carrée (NaN pour cases futures)
mat = np.full((n_tasks, n_tasks), np.nan)
for i, row in enumerate(acc_matrix_raw):
    for j, val in enumerate(row):
        mat[i, j] = val

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlGn", aspect="auto")

ax.set_xlabel("Tâche évaluée")
ax.set_ylabel("Après entraînement sur tâche")
ax.set_title("TinyOL — Matrice accuracy CL\nDataset 1 (Domain-Incremental, 3 tâches)")
ax.set_xticks(range(n_tasks))
ax.set_xticklabels([f"T{i+1}" for i in range(n_tasks)])
ax.set_yticks(range(n_tasks))
ax.set_yticklabels([f"T{i+1}" for i in range(n_tasks)])

for i in range(n_tasks):
    for j in range(i + 1):
        val = mat[i, j]
        color = "white" if val < 0.5 else "black"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                fontsize=10, color=color, fontweight="bold")

plt.colorbar(im, ax=ax, label="Accuracy", fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(EXP_003_DIR / "cl_accuracy_matrix_notebook.png", dpi=150, bbox_inches="tight")
plt.show()

# Lecture : la diagonale = performance immédiate après entraînement sur tᵢ
# La sous-diagonale = oubli sur les tâches précédentes
diag = [mat[i, i] for i in range(n_tasks)]
print(f"\nAccuracy diagonale (performance instantanée) : {[f'{v:.3f}' for v in diag]}")
```

### Cellule 4 — Analyse du forgetting par tâche

```python
# Cellule 4 — Courbes forgetting par tâche
fig, axes = plt.subplots(1, n_tasks, figsize=(4 * n_tasks, 4), sharey=True)

task_names = ["T1 — Saine", "T2 — Usure", "T3 — Pré-panne"]

for task_j in range(n_tasks):
    # Évolution de l'accuracy sur tâche j au fur et à mesure des tâches suivantes
    history = []
    for i in range(task_j, n_tasks):
        if j < len(acc_matrix_raw[i]):
            history.append((i + 1, acc_matrix_raw[i][task_j]))

    if not history:
        continue

    steps, accs = zip(*history)
    ax = axes[task_j] if n_tasks > 1 else axes
    ax.plot(steps, accs, "o-", color="steelblue", linewidth=2, markersize=7)
    ax.axhline(y=accs[0], color="coral", linestyle="--", linewidth=1.5,
               label=f"Pic : {accs[0]:.3f}")
    ax.set_title(f"Tâche {task_j + 1}\n({task_names[task_j]})")
    ax.set_xlabel("Tâche d'entraînement")
    ax.set_xticks(list(steps))
    ax.set_xticklabels([f"T{s}" for s in steps])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

axes[0].set_ylabel("Accuracy")
fig.suptitle("TinyOL — Évolution de l'accuracy par tâche (forgetting)", fontsize=13)
plt.tight_layout()
plt.show()

# Calcul forgetting par tâche (delta pic → fin)
print("\nForgetting par tâche :")
for j in range(n_tasks - 1):
    peak_acc = acc_matrix_raw[j][j]
    final_acc = acc_matrix_raw[-1][j]
    forgetting = peak_acc - final_acc
    print(f"  T{j+1} : pic={peak_acc:.4f}, fin={final_acc:.4f}, oubli={forgetting:+.4f}")
```

### Cellule 5 — Comparaison TinyOL vs baselines

```python
# Cellule 5 — Tableau comparatif modèles
# Note : remplir les colonnes EWC et Fine-tuning depuis exp_001 si disponible

comparison_data = {
    "Modèle": ["TinyOL (M1)", "EWC + MLP (M2)*", "Fine-tuning naïf*"],
    "Dataset": ["Dataset 1\n(Pump)", "Dataset 2\n(Monitoring)", "Dataset 2\n(Monitoring)"],
    "AA": [
        results["acc_final"],
        0.9824,   # exp_001
        0.9811,   # exp_001
    ],
    "AF": [
        results["avg_forgetting"],
        0.0010,
        0.0000,
    ],
    "BWT": [
        results["backward_transfer"],
        0.0000,
        0.0010,
    ],
    "RAM update (B)": [
        results["ram_peak_bytes"],
        6837,
        "—",
    ],
    "Latence (ms)": [
        f"{results['inference_latency_ms']:.3f}",
        "0.036",
        "—",
    ],
}

df = pd.DataFrame(comparison_data)
df = df.set_index("Modèle")

print("Tableau comparatif — TinyOL vs baselines\n")
print(df.to_string())
print("\n* Résultats sur Dataset 2 (différent de Dataset 1) — comparaison indicative uniquement")
print("  Voir S4-04 pour la comparaison homogène sur les deux datasets.")
```

### Cellule 6 — Tableau récapitulatif pour `roadmap.md`

```python
# Cellule 6 — Export tableau roadmap
print("=" * 60)
print("Copier-coller dans roadmap.md — section 'Résultats M1 TinyOL'")
print("=" * 60)
print()
print("### Résultats M1 TinyOL — exp_003 (seed=42, cpu, Dataset 1 — 3 tâches)\n")
print("| Métrique | TinyOL |")
print("|----------|:------:|")
print(f"| AA | **{results['acc_final']:.4f}** |")
print(f"| AF | **{results['avg_forgetting']:.4f}** |")
print(f"| BWT | **{results['backward_transfer']:+.4f}** |")
print(f"| RAM peak update | **{results['ram_peak_bytes']} B** |")
print(f"| Latence inférence | **{results['inference_latency_ms']:.3f} ms** |")
print(f"| Params OtO | **{results['n_params_oto']}** (40 B @ FP32) |")
print(f"| Params encodeur | **{results['n_params_encoder']}** (~5,8 Ko @ FP32) |")
print(f"| Budget 64 Ko | **{results['ram_peak_bytes'] / 65536 * 100:.1f}%** |")
```

---

## Critères d'acceptation

- [ ] `notebooks/03_cl_evaluation.ipynb` créé dans `notebooks/`
- [ ] "Run All" s'exécute sans erreur à partir de `experiments/exp_003_tinyol_dataset1/results/metrics.json`
- [ ] Heatmap matrice accuracy générée et lisible
- [ ] Courbes forgetting par tâche générées (1 subplot par tâche)
- [ ] Tableau comparatif TinyOL vs baselines affiché (valeurs numériques)
- [ ] Cellule 6 produit le bloc Markdown prêt à copier-coller dans `roadmap.md`
- [ ] Pas de code modèle dans le notebook — uniquement lecture de `metrics.json` + visualisation
- [ ] Le notebook ne modifie aucun fichier source (lecture seule)

---

## Commande complète

```bash
# Prérequis : S3-06 terminé (exp_003 exécutée)
pip install -e ".[dev]"
pip install jupyterlab  # si non installé

# Lancer le notebook
jupyter lab notebooks/03_cl_evaluation.ipynb

# Ou exécuter en mode non-interactif (CI)
jupyter nbconvert --to notebook --execute notebooks/03_cl_evaluation.ipynb \
    --output notebooks/03_cl_evaluation_executed.ipynb
```

---

## Questions ouvertes

- `TODO(arnaud)` : faut-il ajouter une comparaison avec la baseline fine-tuning naïf sur Dataset 1 dans ce notebook, ou reporter au Sprint 4 (S4-04, tableau comparatif final) ?
- `TODO(arnaud)` : les 3 tâches du Dataset 1 correspondent-elles à des phases de dégradation interprétables (T1=saine, T2=usure, T3=pré-panne) ? À préciser dans les titres des subplots si oui.
- `FIXME(gap1)` : si l'AF est quasi-nul comme sur Dataset 2 (domaines trop similaires), documenter explicitement cette limitation dans une cellule Markdown du notebook et dans le manuscrit.
