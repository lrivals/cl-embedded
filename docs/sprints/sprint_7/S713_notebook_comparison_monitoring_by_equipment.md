# S7-13 — Notebook Comparaison — monitoring_by_equipment

| Champ | Valeur |
|-------|--------|
| **ID** | S7-13 |
| **Sprint** | Sprint 7 — Phase 1 Extension |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 3h |
| **Dépendances** | S7-01 (TinyOL), S7-02 (EWC), S7-03 (HDC), S7-04 (KMeans), S7-05 (Mahalanobis), S7-06 (DBSCAN) |
| **Fichier cible** | `notebooks/cl_eval/monitoring_by_equipment/comparison.ipynb` |
| **Statut** | ⬜ À faire |

---

## Objectif

Créer le notebook de **comparaison des 6 modèles** sur le scénario **monitoring_by_equipment** (Pump → Turbine → Compressor, 3 tâches). Ce notebook est le support de présentation principal pour les réunions d'encadrement.

**Critère de succès** : notebook exécutable du début à la fin sans erreur, avec toutes les figures comparatives sauvegardées dans `notebooks/figures/cl_evaluation/comparison/monitoring/by_equipment/`, et un tableau récapitulatif complet AA/AF/BWT/AUROC/RAM/latence pour les 6 modèles.

---

## Structure de sortie attendue

```
notebooks/cl_eval/monitoring_by_equipment/
└── comparison.ipynb                    ← notebook créé

notebooks/figures/cl_evaluation/comparison/monitoring/by_equipment/
├── radar_comparison.png
├── barplot_aa_comparison.png
└── acc_matrix_grid.png
```

---

## Contenu du notebook (6 sections)

### Section 0 — En-tête (cellule Markdown)

```markdown
# Comparaison 6 modèles — Dataset 2 Equipment Monitoring — by_equipment

| Champ | Valeur |
|-------|--------|
| **Scénario** | by_equipment : Pump → Turbine → Compressor (3 tâches) |
| **Modèles** | TinyOL · EWC · HDC · KMeans · Mahalanobis · DBSCAN |
| **Dataset** | equipment_anomaly_data.csv — 7 672 échantillons |
| **Date** | {date d'exécution} |

Ce notebook agrège les résultats des expériences exp_011, exp_001, exp_002, exp_005, exp_007, exp_008.
Il constitue le support de présentation pour les réunions d'encadrement.
```

### Section 1 — Setup & imports + chargement de tous les résultats

```python
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path("../..").resolve()))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.plots import (
    plot_accuracy_matrix, plot_model_radar, plot_metrics_comparison, save_figure
)

FIGURES_DIR = Path("../../notebooks/figures/cl_evaluation/comparison/monitoring/by_equipment")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TASK_NAMES = ["Pump", "Turbine", "Compressor"]

# Mapping modèle → (répertoire expérience, fichier métriques)
MODEL_EXP_MAP = {
    "TinyOL":      ("exp_011_tinyol_dataset2",     "metrics.json"),
    "EWC":         ("exp_001_ewc_dataset2",         "metrics.json"),
    "HDC":         ("exp_002_hdc_dataset2",         "metrics.json"),
    "KMeans":      ("exp_005_unsupervised_dataset2","metrics_kmeans.json"),
    "Mahalanobis": ("exp_007_mahalanobis",          "metrics_mahalanobis_dataset2.json"),
    "DBSCAN":      ("exp_008_dbscan",               "metrics_dbscan_dataset2.json"),
}

BASE = Path("../../experiments")

results = {}
acc_matrices = {}

for model, (exp_dir, metrics_file) in MODEL_EXP_MAP.items():
    m_path = BASE / exp_dir / "results" / metrics_file
    a_path = BASE / exp_dir / "results" / "acc_matrix.npy"
    results[model]     = json.loads(m_path.read_text())
    # Adapter la clé AA/AF/BWT selon le format du fichier (cl_metrics ou directement)
    acc_matrices[model] = np.load(a_path, allow_pickle=True)
```

### Section 2 — Radar multi-critères (6 modèles)

```python
# 5 axes normalisés : AA, Stabilité (1−AF), BWT neutre (1−|BWT|),
#                     RAM (1 − RAM/64Ko, clampé à 0 si dépassement),
#                     Vitesse (1 − latence/100ms)
RAM_LIMIT = 64 * 1024  # bytes

radar_data = {}
for model, m in results.items():
    aa  = m.get("aa",  m.get("acc_final", 0))
    af  = m.get("af",  m.get("avg_forgetting", 0))
    bwt = m.get("bwt", m.get("backward_transfer", 0))
    ram = m.get("ram_peak_bytes", 0)
    lat = m.get("inference_latency_ms", 0)
    radar_data[model] = {
        "AA":          aa,
        "Stabilité":   max(0, 1 - af),
        "BWT neutre":  max(0, 1 - abs(bwt)),
        "RAM":         max(0, 1 - ram / RAM_LIMIT),
        "Vitesse":     max(0, 1 - lat / 100.0),
    }

fig = plot_model_radar(radar_data, title="Comparaison 6 modèles — monitoring/by_equipment")
save_figure(fig, FIGURES_DIR / "radar_comparison.png")
plt.show()
```

### Section 3 — Barplot comparaison AA / AF / BWT

```python
# Barplot groupé : AA, AF, BWT pour les 6 modèles
fig = plot_metrics_comparison(results, metrics=["aa", "af", "bwt"],
                               task_names=TASK_NAMES)
save_figure(fig, FIGURES_DIR / "barplot_aa_comparison.png")
plt.show()
```

### Section 4 — Grille de matrices d'accuracy (6 modèles, 2×3)

```python
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, (model, acc_mat) in zip(axes.flat, acc_matrices.items()):
    # Heatmap acc_matrix pour chaque modèle
    im = ax.imshow(acc_mat, vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_title(model)
    ax.set_xticks(range(len(TASK_NAMES)))
    ax.set_yticks(range(len(TASK_NAMES)))
    ax.set_xticklabels(TASK_NAMES, rotation=45)
    ax.set_yticklabels(TASK_NAMES)
    for i in range(len(TASK_NAMES)):
        for j in range(len(TASK_NAMES)):
            if i < acc_mat.shape[0] and j < acc_mat.shape[1]:
                ax.text(j, i, f"{acc_mat[i,j]:.2f}", ha="center", va="center", fontsize=8)
plt.colorbar(im, ax=axes.ravel().tolist())
plt.suptitle("Matrices d'accuracy — monitoring/by_equipment", fontsize=14)
plt.tight_layout()
save_figure(fig, FIGURES_DIR / "acc_matrix_grid.png")
plt.show()
```

### Section 5 — Tableau comparatif complet

```python
RAM_LIMIT = 64 * 1024

print("| Modèle | AA | AF | BWT | AUROC | RAM | Latence | n_params |")
print("|--------|:--:|:--:|:---:|:-----:|:---:|:-------:|:--------:|")

for model, m in results.items():
    aa    = m.get("aa",  m.get("acc_final", float("nan")))
    af    = m.get("af",  m.get("avg_forgetting", float("nan")))
    bwt   = m.get("bwt", m.get("backward_transfer", float("nan")))
    auroc = m.get("auroc_avg", "—")
    ram   = m.get("ram_peak_bytes", 0)
    lat   = m.get("inference_latency_ms", float("nan"))
    n     = m.get("n_params", "—")
    ram_str = f"{ram/1024:.1f} Ko {'⚠️' if ram > RAM_LIMIT else ''}"
    auroc_str = f"{auroc:.4f}" if isinstance(auroc, float) else auroc
    print(f"| {model} | {aa:.4f} | {af:.4f} | {bwt:.4f} | {auroc_str} | {ram_str} | {lat:.5f} ms | {n} |")
```

**Valeurs de référence attendues** :

| Modèle | AA | AF | BWT | AUROC | RAM | Latence | n_params |
|--------|:--:|:--:|:---:|:-----:|:---:|:-------:|:--------:|
| TinyOL | 0.9123 | 0.0079 | −0.0029 | — | 4.3 Ko | 0.00975 ms | 194 |
| EWC | 0.9824 | 0.0010 | 0.0000 | — | 1.1 Ko | 0.03580 ms | 705 |
| HDC | 0.8698 | 0.0000 | +0.0019 | — | 14.2 Ko | 0.04758 ms | 2 048 |
| KMeans | 0.9433 | 0.0049 | −0.0040 | 0.9621 | 5.2 Ko | 0.39870 ms | — |
| Mahalanobis | 0.9524 | 0.0010 | −0.0010 | 0.9718 | 1.5 Ko | 0.01801 ms | — |
| DBSCAN | 0.9557 | 0.0000 | +0.0010 | 0.9786 | 71.9 Ko ⚠️ | 0.42320 ms | — |

### Section 6 — Discussion qualitative (cellule Markdown)

```markdown
## Discussion — monitoring_by_equipment

### Quel modèle a la meilleure accuracy sur ce scénario ?
EWC obtient le meilleur AA (0.9824), suivi de Mahalanobis (0.9524) et DBSCAN (0.9557).
HDC a l'AA le plus faible (0.8698) malgré un AF=0.0.

### Quel modèle a le meilleur ratio performance/RAM ?
Mahalanobis : AA=0.9524, AUROC=0.9718, RAM=1.5 Ko, latence=0.018 ms.
EWC : AA=0.9824, RAM=1.1 Ko, latence=0.036 ms.
Ces deux modèles sont les candidats les plus solides pour STM32N6 (Gap 2).

### Y a-t-il de l'oubli catastrophique ?
Non — tous les modèles ont AF < 0.01 sur ce scénario à 3 tâches.
HDC, DBSCAN et KMeans atteignent AF=0.0 par construction.

### Contrainte RAM STM32N6 (64 Ko)
- ✅ TinyOL : 4.3 Ko
- ✅ EWC : 1.1 Ko
- ✅ HDC : 14.2 Ko
- ✅ KMeans : 5.2 Ko
- ✅ Mahalanobis : 1.5 Ko
- ❌ DBSCAN : 71.9 Ko (dépasse la limite) — FIXME(gap2)

### Questions scientifiques ouvertes
- FIXME(gap1) : Ces résultats sur données industrielles réelles comblent-ils Gap 1 ?
- FIXME(gap2) : DBSCAN dépasse 64 Ko — piste buffer borné ou quantification INT8 des points noyaux.
- FIXME(gap3) : Les modèles non-supervisés ne nécessitent pas de labels — avantage industriel fort.
```

---

## Métriques attendues (résumé)

Voir tableau Section 5 ci-dessus avec les 6 modèles.

---

## Critères d'acceptation

- [ ] `notebooks/cl_eval/monitoring_by_equipment/comparison.ipynb` créé
- [ ] Notebook exécutable sans erreur (Restart Kernel + Run All)
- [ ] `radar_comparison.png` sauvegardé — radar à 5 axes pour 6 modèles
- [ ] `barplot_aa_comparison.png` sauvegardé — barplot AA/AF/BWT
- [ ] `acc_matrix_grid.png` sauvegardé — grille 2×3 des matrices d'accuracy
- [ ] Tableau récapitulatif complet avec les 6 modèles et toutes les métriques
- [ ] Avertissement DBSCAN RAM > 64 Ko présent dans le notebook
- [ ] Section discussion avec les 3 FIXME(gap1/2/3) documentés

---

## Vérification post-création

```bash
python -c "
import nbformat
from pathlib import Path
nb_path = Path('notebooks/cl_eval/monitoring_by_equipment/comparison.ipynb')
nb = nbformat.read(nb_path, as_version=4)
print(f'Nombre de cellules : {len(nb.cells)}')
figs = list(Path('notebooks/figures/cl_evaluation/comparison/monitoring/by_equipment').glob('*.png'))
print(f'Figures générées : {[f.name for f in figs]}')
assert len(figs) == 3, f'Attendu 3 figures, obtenu {len(figs)}'
print('OK — 3 figures comparatives présentes')
"
```

---

## Questions ouvertes

- `TODO(arnaud)` : La section discussion doit-elle inclure des recommandations explicites de déploiement (ex. EWC ou Mahalanobis recommandé pour STM32N6) ? À valider avant la réunion d'encadrement.
- `TODO(arnaud)` : Inclure les modèles KNN et PCA (aussi dans exp_005) dans le radar, ou garder 6 modèles seulement ?
- `FIXME(gap3)` : Les modèles non-supervisés (KMeans, Mahalanobis, DBSCAN) n'ont pas besoin de labels — noter explicitement cet avantage dans la discussion (pertinent pour l'application industrielle Edge Spectrum).
