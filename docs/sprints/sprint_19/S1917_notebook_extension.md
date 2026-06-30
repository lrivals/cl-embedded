# S1917 — Extension notebook sprint19_plots.ipynb

| Champ | Valeur |
|-------|--------|
| **Sprint** | 19 |
| **Priorité** | 🟢 Nice-to-have |
| **Statut** | ✅ Terminé — §9–§13 ajoutés (32 cellules total) |
| **Durée estimée** | 3h |
| **Dépendances** | S1914–S1916 (experiments board disponibles), S1918–S1922 (EWC/HDC multi-dataset) |
| **Fichiers cibles** | `notebooks/sprint19_plots.ipynb` |

---

## Contexte

Le notebook `notebooks/sprint19_plots.ipynb` couvre actuellement les sections §1–§3 (EWC λ sweep, forgetting, scatter board vs dry-run). Cette tâche étend le notebook pour couvrir l'ensemble de la matrice modèle × dataset produite par S1914–S1925, et ajouter des visualisations comparatives.

---

## Sections à ajouter

### §4 — Mahalanobis : comparaison cross-dataset

**Type** : Bar chart groupé — `acc_final` par dataset (CWRU, Pronostia, Monitoring)

**Données** : `exp_S19_01`, `exp_S19_03`, `exp_S19_04`

```python
import json, matplotlib.pyplot as plt, numpy as np

datasets = ["CWRU", "Pronostia", "Monitoring"]
exp_ids  = ["exp_S19_01", "exp_S19_03", "exp_S19_04"]

accs = [json.load(open(f"experiments/{e}/results.json"))["acc_final"] for e in exp_ids]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(datasets, accs, color=["#4C72B0", "#DD8452", "#55A868"])
ax.set_ylim(0, 1)
ax.set_ylabel("acc_final")
ax.set_title("Mahalanobis C — Accuracy finale par dataset (NUCLEO-F439ZI)")
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, acc + 0.02, f"{acc:.2f}", ha="center")
plt.tight_layout()
plt.savefig("notebooks/figures/s19_mahal_cross_dataset.png", dpi=150)
```

---

### §5 — M4 baselines : heatmap AUROC

**Type** : Heatmap — modèles (KMeans, DBSCAN, KNN, PCA) × datasets (CWRU, Pronostia, Monitoring)

**Données** : `exp_S19_13` à `exp_S19_24`

```python
import seaborn as sns

models   = ["KMeans", "DBSCAN", "KNN", "PCA"]
datasets = ["CWRU", "Monitoring", "Pronostia"]
exp_map  = {
    ("KMeans",  "CWRU"):       "exp_S19_13",
    ("KMeans",  "Monitoring"): "exp_S19_14",
    ("KMeans",  "Pronostia"):  "exp_S19_15",
    ("DBSCAN",  "CWRU"):       "exp_S19_16",
    ("DBSCAN",  "Monitoring"): "exp_S19_17",
    ("DBSCAN",  "Pronostia"):  "exp_S19_18",
    ("KNN",     "CWRU"):       "exp_S19_19",
    ("KNN",     "Monitoring"): "exp_S19_20",
    ("KNN",     "Pronostia"):  "exp_S19_21",
    ("PCA",     "CWRU"):       "exp_S19_22",
    ("PCA",     "Monitoring"): "exp_S19_23",
    ("PCA",     "Pronostia"):  "exp_S19_24",
}

matrix = np.array([
    [json.load(open(f"experiments/{exp_map[(m, d)]}/results.json")).get("auroc", np.nan)
     for d in datasets]
    for m in models
])

fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(matrix, annot=True, fmt=".2f", xticklabels=datasets, yticklabels=models,
            vmin=0.5, vmax=1.0, cmap="YlOrRd", ax=ax)
ax.set_title("M4 Baselines — AUROC par modèle × dataset (PC Python)")
plt.tight_layout()
plt.savefig("notebooks/figures/s19_m4_auroc_heatmap.png", dpi=150)
```

---

### §6 — Latence board par modèle

**Type** : Scatter latency vs accuracy — 1 point par (modèle, dataset)

**Données** : `exp_S19_01` à `exp_S19_12` (modèles C board uniquement)

```python
# Tous les experiments board C
board_exps = {
    ("Mahalanobis", "CWRU"):       "exp_S19_01",
    ("EWC",         "Monitoring"): "exp_S19_02",
    ("Mahalanobis", "Pronostia"):  "exp_S19_03",
    ("Mahalanobis", "Monitoring"): "exp_S19_04",
    ("EWC",         "CWRU"):       "exp_S19_05",
    ("EWC",         "Pronostia"):  "exp_S19_06",
    ("HDC",         "CWRU"):       "exp_S19_07",
    ("HDC",         "Monitoring"): "exp_S19_08",
    ("HDC",         "Pronostia"):  "exp_S19_09",
    ("TinyOL",      "CWRU"):       "exp_S19_10",
    ("TinyOL",      "Monitoring"): "exp_S19_11",
    ("TinyOL",      "Pronostia"):  "exp_S19_12",
}

colors = {"Mahalanobis": "#4C72B0", "EWC": "#DD8452", "HDC": "#55A868", "TinyOL": "#C44E52"}
markers = {"CWRU": "o", "Monitoring": "s", "Pronostia": "^"}

fig, ax = plt.subplots(figsize=(8, 5))
for (model, dataset), exp_id in board_exps.items():
    try:
        r = json.load(open(f"experiments/{exp_id}/results.json"))
        ax.scatter(r["inference_latency_ms"], r["acc_final"],
                   c=colors[model], marker=markers[dataset], s=120,
                   label=f"{model}/{dataset}")
    except FileNotFoundError:
        pass  # expérience pas encore lancée

ax.set_xlabel("Latence inférence (ms)")
ax.set_ylabel("acc_final")
ax.set_title("Board NUCLEO-F439ZI — Latence vs Accuracy par modèle × dataset")
ax.axvline(100, color="red", linestyle="--", label="Contrainte 100 ms (Gap 2)")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("notebooks/figures/s19_latency_scatter.png", dpi=150)
```

---

### §7 — RAM statique par modèle

**Type** : Bar chart horizontal — `ram_peak_bytes` par modèle vs budget 256 Ko

```python
model_ram = {
    "Mahalanobis": 1768,   # mesuré S1911
    "EWC":         None,   # à mesurer S1918
    "HDC":         28364,  # estimé
    "TinyOL":      7040,   # estimé (S1903)
}

fig, ax = plt.subplots(figsize=(7, 4))
names = list(model_ram.keys())
vals  = [v if v else 0 for v in model_ram.values()]

ax.barh(names, [v / 1024 for v in vals], color="#4C72B0")
ax.axvline(256, color="red", linestyle="--", label="Budget 256 Ko (NUCLEO)")
ax.axvline(64, color="orange", linestyle="--", label="Budget 64 Ko (STM32N6 cible)")
ax.set_xlabel("RAM statique (Ko)")
ax.set_title("RAM firmware par modèle — NUCLEO-F439ZI")
ax.legend()
plt.tight_layout()
plt.savefig("notebooks/figures/s19_ram_static.png", dpi=150)
```

---

### §8 — Tableau récapitulatif cross-modèle × cross-dataset

**Type** : Tableau Markdown/HTML généré programmatiquement

```python
import pandas as pd

rows = []
for (model, dataset), exp_id in board_exps.items():
    try:
        r = json.load(open(f"experiments/{exp_id}/results.json"))
        rows.append({
            "Modèle": model, "Dataset": dataset,
            "acc_final": f"{r['acc_final']:.3f}" if r.get("acc_final") else "—",
            "avg_forgetting": f"{r['avg_forgetting']:.3f}" if r.get("avg_forgetting") else "—",
            "lat (ms)": f"{r['inference_latency_ms']:.3f}" if r.get("inference_latency_ms") else "—",
            "RAM (B)": r.get("ram_peak_bytes", "—"),
        })
    except FileNotFoundError:
        rows.append({"Modèle": model, "Dataset": dataset,
                     "acc_final": "⬜", "avg_forgetting": "⬜", "lat (ms)": "⬜", "RAM (B)": "⬜"})

df = pd.DataFrame(rows).sort_values(["Modèle", "Dataset"])
display(df.to_html(index=False))
```

---

## Fichiers cibles

| Fichier | Action |
|---------|--------|
| `notebooks/sprint19_plots.ipynb` | Ajouter cellules §4–§8 |
| `notebooks/figures/s19_mahal_cross_dataset.png` | Généré par §4 |
| `notebooks/figures/s19_m4_auroc_heatmap.png` | Généré par §5 |
| `notebooks/figures/s19_latency_scatter.png` | Généré par §6 |
| `notebooks/figures/s19_ram_static.png` | Généré par §7 |

---

## Points de vigilance

### Expériences manquantes = `FileNotFoundError`

Les cellules §6 et §8 utilisent `try/except FileNotFoundError` pour afficher `⬜` quand une expérience n'est pas encore lancée (TinyOL bloqué par S1903). Le notebook reste exécutable à tout moment.

### Figures Pronostia

Si S1914, S1919, S1922, S1925 ne sont pas encore lancées, les points Pronostia seront absents du scatter §6. C'est attendu.

---

## Vérification

- [ ] Sections §4–§8 ajoutées sans écraser §1–§3 existants
- [ ] Toutes les cellules s'exécutent sans erreur (même si certains JSONs manquent)
- [ ] Figures exportées dans `notebooks/figures/`
- [ ] `try/except FileNotFoundError` sur tous les chargements JSON d'expériences non garanties
