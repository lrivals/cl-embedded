# S7-14 — Notebook Comparaison — monitoring_by_location

| Champ | Valeur |
|-------|--------|
| **ID** | S7-14 |
| **Sprint** | Sprint 7 — Phase 1 Extension |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 3h |
| **Dépendances** | S7-07 (TinyOL), S7-08 (EWC), S7-09 (HDC), S7-10 (KMeans), S7-11 (Mahalanobis), S7-12 (DBSCAN) |
| **Fichier cible** | `notebooks/cl_eval/monitoring_by_location/comparison.ipynb` |
| **Statut** | ⬜ À faire |

---

## Objectif

Créer le notebook de **comparaison des 6 modèles** sur le scénario **monitoring_by_location** (Atlanta → Chicago → Houston → New York → San Francisco, 5 tâches). Ce notebook est le support de présentation principal pour les réunions d'encadrement.

**Critère de succès** : notebook exécutable du début à la fin sans erreur, avec toutes les figures comparatives sauvegardées dans `notebooks/figures/cl_evaluation/comparison/monitoring/by_location/`, et un tableau récapitulatif complet AA/AF/BWT/AUROC/RAM/latence pour les 6 modèles.

---

## Structure de sortie attendue

```
notebooks/cl_eval/monitoring_by_location/
└── comparison.ipynb                    ← notebook créé

notebooks/figures/cl_evaluation/comparison/monitoring/by_location/
├── radar_comparison.png
├── barplot_aa_comparison.png
├── acc_matrix_grid.png
├── scatter_ram_vs_accuracy.png          ← cohérence baseline (Gap 2)
├── scatter_flops_vs_accuracy.png        ← cohérence baseline (coût portable)
└── scatter_latency_vs_accuracy.png      ← cohérence baseline (latence PC)
```

---

## Figures requises (cohérence avec baseline)

En plus des trois figures comparatives classiques (radar, barplot, matrice), ce notebook **doit** produire les trois scatters `accuracy vs <ressource>` **au même format que `monitoring_single_task.ipynb`** (cellules `cell-6` RAM, `cell-6b` FLOPs, `cell-6c` latence) pour permettre la comparaison directe avec la baseline hors-CL :

- `scatter_ram_vs_accuracy.png` — trade-off RAM (Gap 2, budget STM32 ≤ 64 Ko)
- `scatter_flops_vs_accuracy.png` — coût de calcul portable (MACs via `src.evaluation.compute_macs`, indépendant de la machine)
- `scatter_latency_vs_accuracy.png` — latence PC (budget 100 ms, ⚠ non transférable au MCU)

Les trois scatters utilisent le même dict `SCATTER_MARKERS` que la baseline pour que les 6 modèles soient identifiables d'un notebook à l'autre.

---

## Contenu du notebook (6 sections)

### Section 0 — En-tête (cellule Markdown)

```markdown
# Comparaison 6 modèles — Dataset 2 Equipment Monitoring — by_location

| Champ | Valeur |
|-------|--------|
| **Scénario** | by_location : Atlanta → Chicago → Houston → New York → San Francisco (5 tâches) |
| **Modèles** | TinyOL · EWC · HDC · KMeans · Mahalanobis · DBSCAN |
| **Dataset** | equipment_anomaly_data.csv — 7 672 échantillons |
| **Date** | {date d'exécution} |

Ce notebook agrège les résultats des expériences exp_018, exp_016, exp_017, exp_022, exp_019, exp_023.
Il constitue le support de présentation pour les réunions d'encadrement (scénario by_location).
Voir `notebooks/cl_eval/monitoring_by_equipment/comparison.ipynb` pour le scénario by_equipment.
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

FIGURES_DIR = Path("../../notebooks/figures/cl_evaluation/comparison/monitoring/by_location")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TASK_NAMES = ["Atlanta", "Chicago", "Houston", "New York", "San Francisco"]

# Mapping modèle → (répertoire expérience, fichier métriques)
MODEL_EXP_MAP = {
    "TinyOL":      ("exp_018_tinyol_monitoring_by_location",      "metrics.json"),
    "EWC":         ("exp_016_ewc_monitoring_by_location",          "metrics.json"),
    "HDC":         ("exp_017_hdc_monitoring_by_location",          "metrics.json"),
    "KMeans":      ("exp_022_kmeans_monitoring_by_location",       "metrics_kmeans.json"),
    "Mahalanobis": ("exp_019_mahalanobis_monitoring_by_location",  "metrics_mahalanobis_dataset2.json"),
    "DBSCAN":      ("exp_023_dbscan_monitoring_by_location",       "metrics_dbscan_dataset2.json"),
}

BASE = Path("../../experiments")
RAM_LIMIT = 64 * 1024  # bytes — contrainte STM32N6

results = {}
acc_matrices = {}

for model, (exp_dir, metrics_file) in MODEL_EXP_MAP.items():
    m_path = BASE / exp_dir / "results" / metrics_file
    a_path = BASE / exp_dir / "results" / "acc_matrix.npy"
    results[model]      = json.loads(m_path.read_text())
    acc_matrices[model] = np.load(a_path, allow_pickle=True)
```

### Section 2 — Radar multi-critères (6 modèles)

```python
# 5 axes normalisés : AA, Stabilité (1−AF), BWT neutre (1−|BWT|),
#                     RAM (1 − RAM/64Ko, clampé à 0 si dépassement),
#                     Vitesse (1 − latence/100ms)

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

fig = plot_model_radar(radar_data, title="Comparaison 6 modèles — monitoring/by_location")
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
# Matrices 5×5 (5 tâches by_location) — 2×3 subplots
fig, axes = plt.subplots(2, 3, figsize=(21, 12))
for ax, (model, acc_mat) in zip(axes.flat, acc_matrices.items()):
    im = ax.imshow(acc_mat, vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_title(model)
    ax.set_xticks(range(len(TASK_NAMES)))
    ax.set_yticks(range(len(TASK_NAMES)))
    ax.set_xticklabels(TASK_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(TASK_NAMES)
    for i in range(len(TASK_NAMES)):
        for j in range(len(TASK_NAMES)):
            if i < acc_mat.shape[0] and j < acc_mat.shape[1]:
                ax.text(j, i, f"{acc_mat[i,j]:.2f}", ha="center", va="center", fontsize=7)
plt.colorbar(im, ax=axes.ravel().tolist())
plt.suptitle("Matrices d'accuracy — monitoring/by_location (5 tâches)", fontsize=14)
plt.tight_layout()
save_figure(fig, FIGURES_DIR / "acc_matrix_grid.png")
plt.show()
```

### Section 5 — Tableau comparatif complet

```python
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
    ram_str   = f"{ram/1024:.1f} Ko {'⚠️' if ram > RAM_LIMIT else ''}"
    auroc_str = f"{auroc:.4f}" if isinstance(auroc, float) else auroc
    print(f"| {model} | {aa:.4f} | {af:.4f} | {bwt:.4f} | {auroc_str} | {ram_str} | {lat:.5f} ms | {n} |")
```

**Valeurs de référence attendues** (à mettre à jour après exécution des expériences) :

| Modèle | AA | AF | BWT | AUROC | RAM | Latence | n_params |
|--------|:--:|:--:|:---:|:-----:|:---:|:-------:|:--------:|
| TinyOL | À déterminer | À déterminer | À déterminer | — | À déterminer | À déterminer | 194 |
| EWC | À déterminer | À déterminer | À déterminer | — | À déterminer | À déterminer | 705 |
| HDC | À déterminer | 0.0000 | À déterminer | — | À déterminer | À déterminer | 2 048 |
| KMeans | À déterminer | À déterminer | À déterminer | À déterminer | À déterminer | À déterminer | — |
| Mahalanobis | À déterminer | À déterminer | À déterminer | À déterminer | ~0.7 Ko (720 B) | À déterminer | — |
| DBSCAN | À déterminer | 0.0000 | À déterminer | À déterminer | >64 Ko ⚠️ | À déterminer | — |

### Section 6 — Discussion qualitative (cellule Markdown)

```markdown
## Discussion — monitoring_by_location (5 tâches)

### Quel modèle a la meilleure accuracy sur ce scénario ?
À compléter après exécution des expériences exp_016–023.
Référence by_equipment : EWC AA=0.9824 > Mahalanobis AA=0.9524 > DBSCAN AA=0.9557.

### Quel modèle a le meilleur ratio performance/RAM ?
Mahalanobis est le candidat principal :
- RAM théorique 5 tâches = 720 B (0.7 Ko) — 90× sous la contrainte 64 Ko
- AUROC by_equipment = 0.9718, latence = 0.018 ms
EWC : RAM ≈ 1.1 Ko, AA potentiellement élevé (si cohérent avec by_equipment).

### Y a-t-il de l'oubli catastrophique ?
HDC et DBSCAN : AF=0.0 par construction.
Les autres modèles : AF attendu < 0.01 d'après les résultats by_equipment.
Le scénario by_location (5 tâches) est plus exigeant que by_equipment (3 tâches).

### Contrainte RAM STM32N6 (64 Ko)
À compléter après exécution — référence by_equipment :
- ✅ TinyOL : 4.3 Ko
- ✅ EWC : 1.1 Ko
- ✅ HDC : 14.2 Ko
- ✅ KMeans : 5.2 Ko
- ✅ Mahalanobis : 1.5 Ko (720 B théorique pour 5 tâches)
- ❌ DBSCAN : 71.9 Ko by_equipment → probablement plus élevé sur by_location — FIXME(gap2)

### Comparaison by_equipment vs by_location
Comparer les métriques AA/AF/BWT/RAM entre les deux scénarios pour identifier
si le drift géographique (by_location) est plus ou moins difficile que le drift
par type d'équipement (by_equipment). Cette comparaison est un résultat original (Gap 1).

### Questions scientifiques ouvertes
- FIXME(gap1) : Ces résultats sur données industrielles réelles (5 locations géographiques)
  constituent une contribution originale — documenter dans le manuscrit.
- FIXME(gap2) : DBSCAN dépasse 64 Ko — piste buffer borné ou quantification INT8.
- FIXME(gap3) : Les modèles non-supervisés (KMeans, Mahalanobis, DBSCAN) n'ont pas
  besoin de labels — avantage industriel fort pour le déploiement multi-sites.
```

---

## Métriques attendues (résumé)

Voir tableau Section 5 ci-dessus (à compléter après exécution des expériences exp_016–023).

---

## Critères d'acceptation

- [ ] `notebooks/cl_eval/monitoring_by_location/comparison.ipynb` créé
- [ ] Notebook exécutable sans erreur (Restart Kernel + Run All)
- [ ] `radar_comparison.png` sauvegardé — radar à 5 axes pour 6 modèles
- [ ] `barplot_aa_comparison.png` sauvegardé — barplot AA/AF/BWT
- [ ] `acc_matrix_grid.png` sauvegardé — grille 2×3 des matrices 5×5
- [ ] Tableau récapitulatif complet avec les 6 modèles et toutes les métriques
- [ ] Avertissement DBSCAN RAM > 64 Ko présent dans le notebook
- [ ] Section discussion avec les 3 FIXME(gap1/2/3) documentés
- [ ] Comparaison by_location vs by_equipment mentionnée dans la discussion

---

## Vérification post-création

```bash
python -c "
import nbformat
from pathlib import Path
nb_path = Path('notebooks/cl_eval/monitoring_by_location/comparison.ipynb')
nb = nbformat.read(nb_path, as_version=4)
print(f'Nombre de cellules : {len(nb.cells)}')
figs = list(Path('notebooks/figures/cl_evaluation/comparison/monitoring/by_location').glob('*.png'))
print(f'Figures générées : {[f.name for f in figs]}')
assert len(figs) == 3, f'Attendu 3 figures, obtenu {len(figs)}'
print('OK — 3 figures comparatives présentes')
"
```

---

## Questions ouvertes

- `TODO(arnaud)` : La section discussion doit-elle inclure une comparaison explicite by_equipment vs by_location (quel scénario est le plus difficile pour chaque modèle) ? Cette comparaison cross-scénario serait une contribution originale pour le manuscrit.
- `TODO(arnaud)` : Inclure les modèles KNN et PCA (potentiellement dans exp_022) dans le radar, ou garder 6 modèles seulement ?
- `FIXME(gap1)` : La comparaison des 6 modèles sur 5 locations géographiques réelles (données industrielles) contribue directement à combler Gap 1 — documenter explicitement dans le manuscrit.
- `FIXME(gap3)` : Les modèles non-supervisés (KMeans, Mahalanobis, DBSCAN) n'ont pas besoin de labels à l'entraînement — noter explicitement cet avantage dans la section discussion (pertinent pour le déploiement multi-sites industriels Edge Spectrum).

---

> **⚠️ Après l'implémentation de ce sprint** : exécuter tous les notebooks via "Restart Kernel & Run All Cells" et vérifier l'absence d'erreurs. Contrôler que chaque sous-dossier de figures est bien créé. Mettre à jour `docs/roadmap_phase1.md` en marquant S7-07 à S7-14 comme ✅.
