# S5-09 — Notebook comparatif supervisé vs non supervisé (5 modèles)

| Champ | Valeur |
|-------|--------|
| **ID** | S5-09 |
| **Sprint** | Sprint 5 — Semaine 5 (13–20 mai 2026) |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 2h |
| **Dépendances** | S5-06 (`exp_005` exécuté), S5-07 (`exp_006` exécuté), S1-09 (`exp_001` EWC), S2-03 (`exp_002` HDC), S3-06 (`exp_003` TinyOL) |
| **Fichiers cibles** | `notebooks/05_supervised_vs_unsupervised.ipynb` |
| **Complété le** | — |

---

## Objectif

Produire le notebook `notebooks/05_supervised_vs_unsupervised.ipynb` qui compare les **5 modèles du projet** sur les deux datasets, et conclut sur le triple gap de la recherche.

| Modèle | Type | Dataset | Expérience |
|--------|------|---------|------------|
| EWC Online | Supervisé, régularisation | Dataset 2 | exp_001 |
| HDC | Supervisé, architecture | Dataset 2 | exp_002 |
| TinyOL | Supervisé, architecture | Dataset 1 | exp_003 |
| K-Means + KNN | Non supervisé | Dataset 2 + 1 | exp_005, exp_006 |
| PCA | Non supervisé | Dataset 2 + 1 | exp_005, exp_006 |

**Critère de succès** : `jupyter nbconvert --to notebook --execute notebooks/05_supervised_vs_unsupervised.ipynb` s'exécute sans erreur et produit toutes les figures de comparaison.

---

## Structure du notebook (7 sections)

```
Section 0 — Setup & imports
Section 1 — Chargement de tous les résultats
Section 2 — Tableau comparatif global AA / AF / BWT / AUROC / RAM / Latence
Section 3 — Matrices d'accuracy CL (heatmaps)
Section 4 — Graphiques barres AA + AF
Section 5 — Profils mémoire et latence
Section 6 — Analyse triple gap + conclusions manuscrit
```

---

## Cellules du notebook

### Section 0 — Setup & imports

```python
# notebooks/05_supervised_vs_unsupervised.ipynb — Cellule 0
"""
Notebook : Comparaison supervisé vs non supervisé — 5 modèles
Sprint 5 — S5-09

Prérequis :
  - exp_001 (EWC) → experiments/exp_001_ewc_dataset2/results/metrics.json
  - exp_002 (HDC) → experiments/exp_002_hdc_dataset2/results/metrics.json
  - exp_003 (TinyOL) → experiments/exp_003_tinyol_dataset1/results/metrics.json
  - exp_005 (Unsup Dataset 2) → experiments/exp_005_unsupervised_dataset2/results/metrics_all.json
  - exp_006 (Unsup Dataset 1) → experiments/exp_006_unsupervised_dataset1/results/metrics_all.json

Références : DeLange2021Survey, Hurtado2023CLPdM, triple_gap.md
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

# Chemins
EXP_EWC      = Path("experiments/exp_001_ewc_dataset2/results")
EXP_HDC      = Path("experiments/exp_002_hdc_dataset2/results")
EXP_TINYOL   = Path("experiments/exp_003_tinyol_dataset1/results")
EXP_UNSUP_D2 = Path("experiments/exp_005_unsupervised_dataset2/results")
EXP_UNSUP_D1 = Path("experiments/exp_006_unsupervised_dataset1/results")

# Vérification des fichiers
required_files = [
    EXP_EWC      / "metrics.json",
    EXP_HDC      / "metrics.json",
    EXP_TINYOL   / "metrics.json",
    EXP_UNSUP_D2 / "metrics_all.json",
    EXP_UNSUP_D1 / "metrics_all.json",
]
for p in required_files:
    if not p.exists():
        print(f"⚠️  Fichier manquant : {p}")
    else:
        print(f"✓ {p}")
```

### Section 1 — Chargement des résultats

```python
# Cellule 1 — Chargement

# Modèles supervisés
with open(EXP_EWC  / "metrics.json") as f: metrics_ewc    = json.load(f)
with open(EXP_HDC  / "metrics.json") as f: metrics_hdc    = json.load(f)
with open(EXP_TINYOL / "metrics.json") as f: metrics_tinyol = json.load(f)

# Modèles non supervisés — Dataset 2
with open(EXP_UNSUP_D2 / "metrics_all.json") as f: metrics_unsup_d2 = json.load(f)

# Modèles non supervisés — Dataset 1
with open(EXP_UNSUP_D1 / "metrics_all.json") as f: metrics_unsup_d1 = json.load(f)

# Matrices d'accuracy
acc_ewc    = np.load(EXP_EWC  / "acc_matrix_ewc.npy")
acc_hdc    = np.load(EXP_HDC  / "acc_matrix_hdc.npy")
acc_km_d2  = np.load(EXP_UNSUP_D2 / "acc_matrix_kmeans.npy")
acc_knn_d2 = np.load(EXP_UNSUP_D2 / "acc_matrix_knn.npy")
acc_pca_d2 = np.load(EXP_UNSUP_D2 / "acc_matrix_pca.npy")

print(f"EWC metrics : {list(metrics_ewc.keys())[:6]}")
print(f"KMeans D2   : {list(metrics_unsup_d2['kmeans'].keys())[:6]}")
```

### Section 2 — Tableau comparatif global

```python
# Cellule 2 — Tableau comparatif

# Construire le DataFrame unifié
rows = []

# Supervisés — Dataset 2
for name, m in [("EWC (D2)", metrics_ewc), ("HDC (D2)", metrics_hdc)]:
    rows.append({
        "Modèle": name,
        "Type": "Supervisé",
        "Dataset": "D2",
        "AA": m.get("acc_final", float("nan")),
        "AF": m.get("avg_forgetting", float("nan")),
        "BWT": m.get("backward_transfer", float("nan")),
        "AUROC": float("nan"),   # Non reporté pour supervisés
        "RAM (Ko)": m.get("ram_peak_bytes", -1) / 1024,
        "Latence (ms)": m.get("inference_latency_ms", float("nan")),
    })

# TinyOL — Dataset 1
rows.append({
    "Modèle": "TinyOL (D1)",
    "Type": "Supervisé",
    "Dataset": "D1",
    "AA": metrics_tinyol.get("acc_final", float("nan")),
    "AF": metrics_tinyol.get("avg_forgetting", float("nan")),
    "BWT": metrics_tinyol.get("backward_transfer", float("nan")),
    "AUROC": float("nan"),
    "RAM (Ko)": metrics_tinyol.get("ram_peak_bytes", -1) / 1024,
    "Latence (ms)": metrics_tinyol.get("inference_latency_ms", float("nan")),
})

# Non supervisés — Dataset 2
for model_key, label in [("kmeans", "K-Means (D2)"), ("knn", "KNN (D2)"), ("pca", "PCA (D2)")]:
    m = metrics_unsup_d2[model_key]
    rows.append({
        "Modèle": label,
        "Type": "Non supervisé",
        "Dataset": "D2",
        "AA": m.get("acc_final", float("nan")),
        "AF": m.get("avg_forgetting", float("nan")),
        "BWT": m.get("backward_transfer", float("nan")),
        "AUROC": m.get("auroc_final", float("nan")),
        "RAM (Ko)": m.get("ram_peak_bytes", -1) / 1024,
        "Latence (ms)": m.get("inference_latency_ms", float("nan")),
    })

# Non supervisés — Dataset 1
for model_key, label in [("kmeans", "K-Means (D1)"), ("knn", "KNN (D1)"), ("pca", "PCA (D1)")]:
    m = metrics_unsup_d1[model_key]
    rows.append({
        "Modèle": label,
        "Type": "Non supervisé",
        "Dataset": "D1",
        "AA": m.get("acc_final", float("nan")),
        "AF": m.get("avg_forgetting", float("nan")),
        "BWT": m.get("backward_transfer", float("nan")),
        "AUROC": m.get("auroc_final", float("nan")),
        "RAM (Ko)": m.get("ram_peak_bytes", -1) / 1024,
        "Latence (ms)": m.get("inference_latency_ms", float("nan")),
    })

df = pd.DataFrame(rows).set_index("Modèle")
df.round(4)
```

### Section 3 — Heatmaps matrices d'accuracy

```python
# Cellule 3 — Heatmaps

DOMAINS_D2 = ["Pump", "Turbine", "Compressor"]
DOMAINS_D1 = ["Healthy", "Wearing", "Pre-failure"]


def plot_acc_matrix(ax, mat: np.ndarray, title: str, domains: list[str]) -> None:
    """Heatmap matrice d'accuracy CL [T, T]."""
    T = len(domains)
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(T)); ax.set_xticklabels(domains, fontsize=9)
    ax.set_yticks(range(T)); ax.set_yticklabels([f"After T{i+1}" for i in range(T)], fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for i in range(T):
        for j in range(T):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8)
    return im


fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Matrices d'accuracy CL — 5 modèles", fontsize=13, fontweight="bold")

matrices_d2 = [
    (acc_ewc,    "EWC Online (D2)",   DOMAINS_D2),
    (acc_hdc,    "HDC (D2)",          DOMAINS_D2),
    (acc_km_d2,  "K-Means (D2)",      DOMAINS_D2),
]
matrices_d1 = [
    (acc_knn_d2, "KNN (D2)",          DOMAINS_D2),
    (acc_pca_d2, "PCA (D2)",          DOMAINS_D2),
]

for ax, (mat, title, domains) in zip(axes[0], matrices_d2):
    plot_acc_matrix(ax, mat, title, domains)

for ax, (mat, title, domains) in zip(axes[1], matrices_d1):
    plot_acc_matrix(ax, mat, title, domains)

axes[1, 2].axis("off")  # Case vide (TinyOL D1 nécessite format différent)

plt.tight_layout()
plt.savefig("notebooks/figures/05_acc_matrices.png", dpi=150, bbox_inches="tight")
plt.show()
```

### Section 4 — Graphiques barres AA + AF

```python
# Cellule 4 — Bar charts comparatifs

fig, (ax_aa, ax_af) = plt.subplots(1, 2, figsize=(13, 5))

colors = {
    "Supervisé":     "#2196F3",   # bleu
    "Non supervisé": "#FF9800",   # orange
}

models   = df.index.tolist()
aa_vals  = df["AA"].values
af_vals  = df["AF"].values
types    = df["Type"].values
bar_colors = [colors[t] for t in types]

# AA
bars = ax_aa.bar(models, aa_vals, color=bar_colors, edgecolor="white", linewidth=0.5)
ax_aa.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Aléatoire (0.5)")
ax_aa.set_ylabel("Average Accuracy (AA)")
ax_aa.set_title("AA — Toutes tâches après entraînement complet")
ax_aa.set_ylim(0, 1.05)
ax_aa.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
ax_aa.legend()

# AF
bars = ax_af.bar(models, af_vals, color=bar_colors, edgecolor="white", linewidth=0.5)
ax_af.axhline(0, color="black", linestyle="-", linewidth=0.5)
ax_af.set_ylabel("Average Forgetting (AF)")
ax_af.set_title("AF — Oubli catastrophique (↓ = meilleur)")
ax_af.set_xticklabels(models, rotation=35, ha="right", fontsize=8)

# Légende commune
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=t) for t, c in colors.items()]
fig.legend(handles=legend_elements, loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig("notebooks/figures/05_aa_af_bars.png", dpi=150, bbox_inches="tight")
plt.show()
```

### Section 5 — Profils mémoire et latence

```python
# Cellule 5 — Scatter RAM vs Latence

fig, ax = plt.subplots(figsize=(9, 6))

for _, row in df.iterrows():
    color = colors[row["Type"]]
    ax.scatter(
        row["Latence (ms)"],
        row["RAM (Ko)"],
        s=120,
        color=color,
        edgecolors="black",
        linewidths=0.5,
        zorder=3,
    )
    ax.annotate(
        row.name,
        (row["Latence (ms)"], row["RAM (Ko)"]),
        textcoords="offset points",
        xytext=(5, 3),
        fontsize=7,
    )

ax.axhline(64, color="red", linestyle="--", linewidth=1.5, label="Budget 64 Ko (STM32N6)")
ax.set_xlabel("Latence inférence (ms)")
ax.set_ylabel("RAM peak (Ko)")
ax.set_title("Profils mémoire et latence — comparaison 5 modèles")
ax.legend()
plt.tight_layout()
plt.savefig("notebooks/figures/05_ram_latence_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
```

### Section 6 — Analyse triple gap

```python
# Cellule 6 — Conclusions triple gap

print("=" * 70)
print("ANALYSE TRIPLE GAP")
print("=" * 70)

# Gap 1 — Données industrielles réelles
print("""
Gap 1 — Validation sur données industrielles réelles
  Dataset 1 (Pump Kaggle) : données simulées, drift artificiel
  Dataset 2 (Monitoring)  : données tabulaires sans drift prononcé
  → Limitation documentée dans docs/triple_gap.md
""")

# Gap 2 — RAM < 100 Ko avec chiffres précis
print("Gap 2 — RAM peak mesurée (tracemalloc) :")
for model_name, m in {**metrics_unsup_d2, **metrics_unsup_d1}.items():
    ram_ko = m.get("ram_peak_bytes", -1) / 1024
    status = "✓ OK" if ram_ko <= 64 else "✗ DÉPASSÉ"
    print(f"  {model_name:<20} : {ram_ko:6.1f} Ko  {status}")

# Gap 3 — Quantification INT8 pendant l'entraînement
print("""
Gap 3 — Quantification INT8 pendant l'entraînement incrémental
  Modèles non supervisés : opèrent en FP32 (sklearn)
  → Non adressé dans cette phase. Voir backlog Phase 2.
""")

print("=" * 70)
print("CONCLUSION : Supervisé vs Non supervisé")
print("-" * 70)
gap_aa = df.loc[df["Type"] == "Supervisé", "AA"].mean() - df.loc[df["Type"] == "Non supervisé", "AA"].mean()
print(f"  Delta AA moyen (supervisé - non supervisé) : {gap_aa:+.4f}")
print(f"  → Coût de l'absence de supervision : {gap_aa*100:.1f} points d'accuracy")
```

---

## Figures produites

| Fichier | Description |
|---------|-------------|
| `notebooks/figures/05_acc_matrices.png` | Heatmaps matrices accuracy — 5 modèles |
| `notebooks/figures/05_aa_af_bars.png` | Bar chart AA et AF comparatifs |
| `notebooks/figures/05_ram_latence_scatter.png` | Scatter RAM vs latence + budget 64 Ko |

> Créer le répertoire `notebooks/figures/` si inexistant avant d'exécuter le notebook.

---

## Critères d'acceptation

- [ ] `jupyter nbconvert --to notebook --execute notebooks/05_supervised_vs_unsupervised.ipynb` s'exécute sans erreur
- [ ] Toutes les figures sont générées dans `notebooks/figures/`
- [ ] Le DataFrame comparatif (Section 2) contient les 9 lignes (3 supervisés + 6 non supervisés)
- [ ] Section 6 affiche les chiffres RAM mesurés et la comparaison AA supervisé vs non supervisé
- [ ] Le notebook est exécutable depuis `Run All` sans intervention manuelle
- [ ] Les fichiers d'expérience manquants affichent un warning (pas une exception) grâce aux checks Section 0

---

## Questions ouvertes

- `TODO(arnaud)` : faut-il inclure une 4e expérience EWC sur Dataset 1 pour comparer directement TinyOL vs EWC sur le même dataset ? Ou conserver la séparation D1/D2 ?
- `TODO(arnaud)` : les figures de ce notebook sont-elles destinées directement au manuscrit (format A4) ou seulement à l'exploration ? Si manuscrit, adapter `figsize` et `fontsize` en conséquence.
- `TODO(fred)` : quels modèles présenter en priorité dans le rapport Edge Spectrum — le meilleur en AA (EWC) ou le meilleur compromis RAM/latence/précision ?
- `FIXME(gap1)` : si `auroc_final ≈ 0.50` pour tous les non supervisés sur Dataset 2, documenter explicitement cette limitation dans la Section 6 et référencer `docs/triple_gap.md`.
- `FIXME(gap3)` : ajouter une note dans Section 6 sur l'absence de quantification INT8 pour les modèles non supervisés sklearn, et la différence avec les approches TinyML de Ren2021TinyOL.
