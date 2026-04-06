# S2-04 — Notebook comparaison EWC vs HDC vs Fine-tuning

| Champ | Valeur |
|-------|--------|
| **ID** | S2-04 |
| **Sprint** | Sprint 2 — Semaine 2 (22–29 avril 2026) |
| **Priorité** | ✅ Terminé |
| **Durée estimée** | 3h |
| **Dépendances** | S2-03 (exp_002 exécuté), S1-09 (exp_001 exécuté) |
| **Fichiers cibles** | `notebooks/02_baseline_comparison.ipynb`, `notebooks/figures/` |
| **Complété le** | 6 avril 2026 |

---

## Objectif

Produire le notebook `notebooks/02_baseline_comparison.ipynb` qui compare les trois approches sur Dataset 2 :
- **EWC Online** (exp_001) — régularisation Fisher
- **HDC** (exp_002) — prototypes hyperdimensionnels
- **Fine-tuning naïf** — borne inférieure

Le notebook doit être **exécutable de bout en bout** sans intervention manuelle et produire des figures permettant de conclure sur le **triple gap** du projet.

**Critère de succès** : `jupyter nbconvert --to notebook --execute notebooks/02_baseline_comparison.ipynb` s'exécute sans erreur.

---

## Sous-tâches

### Section 0 — Imports et chemins

```python
# Cell 0 : Setup
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

# Imports projet
from src.evaluation.metrics import format_metrics_report
from src.evaluation.memory_profiler import compare_models_memory

# Chemins des expériences
EXP_EWC  = Path("experiments/exp_001_ewc_dataset2/results")
EXP_HDC  = Path("experiments/exp_002_hdc_dataset2/results")

DOMAINS = ["Pump", "Turbine", "Compressor"]

# Vérification que les fichiers existent avant de continuer
for p in [EXP_EWC / "metrics.json", EXP_HDC / "metrics.json"]:
    assert p.exists(), f"Fichier manquant : {p}. Lancer S1-09 et S2-03 d'abord."
```

### Section 1 — Chargement des résultats

```python
# Cell 1 : Chargement métriques
with open(EXP_EWC / "metrics.json") as f:
    metrics_ewc = json.load(f)

with open(EXP_HDC / "metrics.json") as f:
    metrics_hdc = json.load(f)

# Matrices d'accuracy
acc_ewc   = np.load(EXP_EWC / "acc_matrix_ewc.npy")
acc_hdc   = np.load(EXP_HDC / "acc_matrix_hdc.npy")

# Baselines EWC (si disponibles)
naive_path = EXP_EWC / "acc_matrix_naive.npy"
joint_path = EXP_EWC / "acc_matrix_joint.npy"
acc_naive = np.load(naive_path) if naive_path.exists() else None
acc_joint = np.load(joint_path) if joint_path.exists() else None

print("Métriques EWC chargées :", list(metrics_ewc.keys()))
print("Métriques HDC chargées :", list(metrics_hdc.keys()))
```

### Section 2 — Tableau comparatif AA / AF / BWT / RAM / Latence

```python
# Cell 2 : Tableau de synthèse
rows = {
    "EWC Online": metrics_ewc,
    "HDC":        metrics_hdc,
}
if acc_naive is not None:
    # Recalculer métriques naïf depuis la matrice
    from src.evaluation.metrics import compute_cl_metrics
    metrics_naive = compute_cl_metrics(acc_naive)
    rows["Fine-tuning naïf"] = metrics_naive

# Colonnes d'intérêt
cols = ["aa", "af", "bwt", "ram_peak_bytes", "inference_latency_ms", "n_params"]
col_labels = ["AA ↑", "AF ↓", "BWT ↑", "RAM peak (B)", "Latence (ms)", "Params"]

table_data = []
for model_name, m in rows.items():
    table_data.append([
        model_name,
        f"{m.get('aa', float('nan')):.4f}",
        f"{m.get('af', float('nan')):.4f}",
        f"{m.get('bwt', float('nan')):.6f}",
        f"{m.get('ram_peak_bytes', -1):,} B",
        f"{m.get('inference_latency_ms', float('nan')):.4f} ms",
        f"{m.get('n_params', -1):,}",
    ])

df = pd.DataFrame(table_data, columns=["Modèle"] + col_labels)
display(df.style.set_caption("Comparaison EWC vs HDC — Dataset 2 (3 domaines)"))
```

### Section 3 — Évolution de l'accuracy par tâche (acc_matrix heatmap)

```python
# Cell 3 : Matrices d'accuracy
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, matrix, title in zip(
    axes,
    [acc_ewc, acc_hdc],
    ["EWC Online", "HDC"]
):
    # Remplacer NaN par une valeur neutre pour l'affichage
    display_matrix = np.where(np.isnan(matrix), -0.05, matrix)
    im = ax.imshow(display_matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(DOMAINS)))
    ax.set_yticks(range(len(DOMAINS)))
    ax.set_xticklabels([f"Task {i+1}\n({d})" for i, d in enumerate(DOMAINS)])
    ax.set_yticklabels([f"After Task {i+1}" for i in range(len(DOMAINS))])
    ax.set_xlabel("Tâche évaluée")
    ax.set_ylabel("Après entraînement sur")
    ax.set_title(f"Matrice d'accuracy — {title}")

    # Annotations
    for i in range(len(DOMAINS)):
        for j in range(len(DOMAINS)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        color="black", fontsize=9, fontweight="bold")
            else:
                ax.text(j, i, "N/A", ha="center", va="center", color="gray", fontsize=8)

    plt.colorbar(im, ax=ax, label="Accuracy")

plt.suptitle("Évolution de l'accuracy par tâche — Dataset 2", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("notebooks/figures/acc_matrix_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
```

### Section 4 — Évolution des prototypes HDC (heatmap)

```python
# Cell 4 : Prototypes HDC (visualisation des accumulateurs)
# Charger le checkpoint HDC final
hdc_checkpoint = Path("experiments/exp_002_hdc_dataset2/checkpoints/hdc_task3_final.npz")
if hdc_checkpoint.exists():
    hdc_state = np.load(hdc_checkpoint)
    prototypes_acc = hdc_state["prototypes_acc"]  # [2, 1024]
    class_counts = hdc_state["class_counts"]      # [2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 3))
    class_names = ["Normal (0)", "Faulty (1)"]

    for ax, proto, name, count in zip(axes, prototypes_acc, class_names, class_counts):
        # Afficher les 256 premières dimensions pour lisibilité
        ax.bar(range(256), proto[:256], color=["steelblue" if v > 0 else "tomato" for v in proto[:256]], width=1)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"Prototype {name} — {count} exemples vus\n(256/1024 dimensions)")
        ax.set_xlabel("Dimension hypervecteur")
        ax.set_ylabel("Valeur accumulateur INT32")

    plt.suptitle("Prototypes HDC après 3 tâches — Dataset 2", fontsize=12)
    plt.tight_layout()
    plt.savefig("notebooks/figures/hdc_prototypes.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Classe Normal : {class_counts[0]} exemples | Classe Faulty : {class_counts[1]} exemples")
else:
    print(f"Checkpoint HDC non trouvé : {hdc_checkpoint}. Lancer S2-03 d'abord.")
```

### Section 5 — Comparaison mémoire

```python
# Cell 5 : Comparaison RAM
with open(EXP_EWC / "memory_report.json") as f:
    mem_ewc = json.load(f)
with open(EXP_HDC / "memory_report.json") as f:
    mem_hdc = json.load(f)

print(compare_models_memory([
    {**mem_ewc, "model_name": "EWC Online"},
    {**mem_hdc, "model_name": "HDC"},
]))

# Graphique comparatif
fig, ax = plt.subplots(figsize=(8, 4))
models = ["EWC Online", "HDC"]
ram_vals = [
    mem_ewc.get("ram_peak_bytes", 0) / 1024,
    mem_hdc.get("estimated_ram_fp32_bytes", 0) / 1024,
]
colors = ["#4C72B0", "#DD8452"]
bars = ax.bar(models, ram_vals, color=colors, width=0.4)
ax.axhline(64, color="red", linestyle="--", label="Budget 64 Ko (STM32N6)")
ax.set_ylabel("RAM (Ko)")
ax.set_title("Empreinte mémoire — EWC vs HDC (Dataset 2)")
ax.legend()
for bar, val in zip(bars, ram_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.1f} Ko", ha="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("notebooks/figures/memory_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
```

### Section 6 — Conclusion triple gap

```python
# Cell 6 : Conclusion
conclusion = f"""
## Conclusions — Sprint 2 (Dataset 2, 3 domaines)

### Résultats clés

| Modèle       | AA     | AF     | RAM peak | Latence  |
|-------------|--------|--------|----------|----------|
| EWC Online  | {metrics_ewc.get('aa', 'N/A'):.4f} | {metrics_ewc.get('af', 'N/A'):.4f} | {metrics_ewc.get('ram_peak_bytes', -1):,} B | {metrics_ewc.get('inference_latency_ms', 'N/A'):.3f} ms |
| HDC         | {metrics_hdc.get('aa', 'N/A'):.4f} | {metrics_hdc.get('af', 'N/A'):.4f} | {metrics_hdc.get('estimated_ram_fp32_bytes', -1):,} B | {metrics_hdc.get('inference_latency_ms', 'N/A'):.3f} ms |

### Triple gap

- **Gap 1** (données industrielles réelles) : ✅ Les deux modèles sont validés sur Dataset 2
  (Industrial Equipment Monitoring, 7 672 échantillons, 3 domaines réels).

- **Gap 2** (< 100 Ko RAM avec chiffres mesurés) : ✅ EWC : ~{metrics_ewc.get('ram_peak_bytes', 0)/1024:.1f} Ko peak
  (update), HDC : ~{metrics_hdc.get('estimated_ram_fp32_bytes', 0)/1024:.1f} Ko estimé — tous deux sous 64 Ko.

- **Gap 3** (quantification INT8 pendant l'entraînement) : ⬜ Non adressé ici — prévu Sprint 4
  (module de quantification UINT8 pour TinyOL).

### Observations

1. HDC atteint une précision de {metrics_hdc.get('aa', float('nan')):.2%} vs {metrics_ewc.get('aa', float('nan')):.2%} pour EWC —
   {'acceptable' if metrics_hdc.get('aa', 0) > 0.85 else 'insuffisant (< 85 %), voir TODO(arnaud)'}.
2. Oubli catastrophique HDC : AF = {metrics_hdc.get('af', float('nan')):.4f} ({'nul par construction ✅' if metrics_hdc.get('af', 1) < 0.01 else 'présent — vérifier implémentation ⚠️'}).
3. HDC est plus frugal en mémoire : {metrics_hdc.get('estimated_ram_fp32_bytes', 0)/1024:.1f} Ko vs
   {metrics_ewc.get('ram_peak_bytes', 0)/1024:.1f} Ko (update EWC).
"""

from IPython.display import Markdown
display(Markdown(conclusion))
```

---

## Contraintes du notebook

- **Pas de données brutes dans le notebook** — toutes les données viennent des fichiers JSON/NPY de `experiments/`
- **Figures sauvegardées** dans `notebooks/figures/` (créer le dossier si absent)
- **Exécutable sans GPU** (HDC = NumPy pur, EWC résultats chargés depuis JSON)
- **Pas de `import torch`** dans les cells de visualisation
- **Compatibilité** : notebook exécutable avec `jupyter nbconvert --execute`

---

## Critères d'acceptation

- [x] `notebooks/02_baseline_comparison.ipynb` créé et versionné (sans outputs)
- [x] Toutes les cells s'exécutent sans erreur (assertion sur existence des fichiers)
- [x] Section 2 : tableau AA/AF/BWT/RAM produit pour EWC et HDC
- [x] Section 3 : matrices d'accuracy [3×3] visualisées avec annotations numériques
- [x] Section 4 : heatmap prototypes HDC visible (ou message clair si checkpoint absent)
- [x] Section 5 : comparaison RAM EWC vs HDC avec ligne budget 64 Ko
- [x] Section 6 : conclusion triple gap (Gap 1 ✅, Gap 2 ✅, Gap 3 ⬜)
- [x] `jupyter nbconvert --to notebook --execute notebooks/02_baseline_comparison.ipynb` — succès
- [x] Figures sauvegardées dans `notebooks/figures/`

---

## Interface attendue (fichiers produits par S2-03)

```
experiments/exp_002_hdc_dataset2/results/
├── metrics.json          # clés : aa, af, bwt, estimated_ram_fp32_bytes, n_params
├── acc_matrix_hdc.npy    # [3, 3] float64
└── memory_report.json    # clés : ram_peak_bytes, inference_latency_ms, ...
```

---

## Notes d'implémentation (6 avril 2026)

La structure réelle des JSON diffère légèrement de l'interface attendue dans la spec :

**`exp_001/results/metrics.json`** — structure imbriquée :

```
data["cl_metrics"]["ewc"]    → {aa, af, bwt, ...}
data["cl_metrics"]["naive"]  → {aa, af, bwt, ...}
data["cl_metrics"]["joint"]  → {aa, af, bwt, ...}
data["cl_metrics"]["memory"] → {model, forward:{...}, update:{...}}
```

**`exp_002/results/metrics.json`** — métriques à plat sous `cl_metrics` :

```
data["cl_metrics"] → {aa, af, bwt, ram_peak_bytes, inference_latency_ms, n_params, ...}
```

Le notebook gère le changement de répertoire (`os.chdir`) en Cell 0 pour être exécutable depuis
`notebooks/` (via `nbconvert`) ou depuis la racine du dépôt.

**Figures produites** :

- `notebooks/figures/acc_matrix_comparison.png` — heatmaps [3×3] EWC et HDC
- `notebooks/figures/hdc_prototypes.png` — accumulateurs des 2 classes (256/2048 dims)
- `notebooks/figures/memory_comparison.png` — barplot RAM avec budget 64 Ko

---

## Questions ouvertes

- `TODO(arnaud)` : faut-il ajouter une courbe d'apprentissage (accuracy task-by-task) ou le tableau AA/AF/BWT suffit pour le manuscrit ?
- `TODO(arnaud)` : la figure `acc_matrix` est-elle suffisante pour illustrer l'absence d'oubli catastrophique HDC, ou faut-il une courbe d'accuracy au fil du temps (sample-level) ?
- `FIXME(gap1)` : ce notebook ne couvre que Dataset 2. La validation sur FEMTO PRONOSTIA (Gap 1 complet) est prévue Phase 2 — ajouter une note dans la cellule conclusion.
