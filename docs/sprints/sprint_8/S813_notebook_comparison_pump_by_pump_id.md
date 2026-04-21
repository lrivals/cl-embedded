# S8-13 — Notebook Comparaison — pump_by_pump_id

| Champ | Valeur |
|-------|--------|
| **ID** | S8-13 |
| **Sprint** | Sprint 8 — Phase 1 Extension |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 3h |
| **Dépendances** | S8-01 (TinyOL), S8-02 (EWC), S8-03 (HDC), S8-04 (KMeans), S8-05 (Mahalanobis), S8-06 (DBSCAN) |
| **Fichier cible** | `notebooks/cl_eval/pump_by_pump_id/comparison.ipynb` |
| **Statut** | ✅ Terminé |

---

## Objectif

Créer le notebook de **comparaison des 6 modèles** sur le scénario **pump_by_pump_id**
(Pump 1 → Pump 2 → Pump 3 → Pump 4 → Pump 5, 5 tâches). Ce notebook est le support
de présentation principal pour les réunions d'encadrement sur Dataset 1.

**Critère de succès** : notebook exécutable du début à la fin sans erreur, avec toutes
les figures comparatives sauvegardées dans
`notebooks/figures/cl_evaluation/comparison/pump/by_pump_id/`, et un tableau récapitulatif
complet AA/AF/BWT/AUROC/RAM/latence pour les 6 modèles.

---

## Structure de sortie attendue

```
notebooks/cl_eval/pump_by_pump_id/
└── comparison.ipynb                    ← notebook créé

notebooks/figures/cl_evaluation/comparison/pump/by_pump_id/
├── radar_comparison.png
├── barplot_aa_comparison.png
├── acc_matrix_grid.png
├── performance_by_pump_id_bar.png       ← spécifique by_pump_id
├── scatter_ram_vs_accuracy.png          ← cohérence baseline (Gap 2)
├── scatter_flops_vs_accuracy.png        ← cohérence baseline (coût portable)
└── scatter_latency_vs_accuracy.png      ← cohérence baseline (latence PC)
```

---

## Figures requises (cohérence avec baseline)

En plus des figures comparatives classiques (radar, barplot, matrice), ce notebook **doit**
produire les trois scatters `accuracy vs <ressource>` **au même format que
`pump_single_task.ipynb`** (cellules `cell-6` RAM, `cell-6b` FLOPs, `cell-6c` latence)
pour permettre la comparaison directe avec la baseline hors-CL :

- `scatter_ram_vs_accuracy.png` — trade-off RAM (Gap 2, budget STM32 ≤ 64 Ko)
- `scatter_flops_vs_accuracy.png` — coût de calcul portable (MACs via `src.evaluation.compute_macs`, indépendant de la machine)
- `scatter_latency_vs_accuracy.png` — latence PC (budget 100 ms, ⚠ non transférable au MCU)

Les trois scatters utilisent le même dict `SCATTER_MARKERS` que la baseline pour que les
6 modèles soient identifiables d'un notebook à l'autre.

---

## Contenu du notebook (7 sections)

### Section 0 — En-tête (cellule Markdown)

```markdown
# Comparaison 6 modèles — Dataset 1 Pump Maintenance — by_pump_id

| Champ | Valeur |
|-------|--------|
| **Scénario** | by_pump_id : Pump 1 → Pump 2 → Pump 3 → Pump 4 → Pump 5 (5 tâches) |
| **Modèles** | TinyOL · EWC · HDC · KMeans · Mahalanobis · DBSCAN |
| **Dataset** | Large_Industrial_Pump_Maintenance_Dataset.csv |
| **Date** | {date d'exécution} |

Ce notebook agrège les résultats des expériences exp_012, exp_013, exp_014, exp_020, exp_015, exp_021.
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
    plot_accuracy_matrix, plot_model_radar, plot_metrics_comparison,
    plot_performance_by_pump_id_bar, save_figure
)

FIGURES_DIR = Path("../../notebooks/figures/cl_evaluation/comparison/pump/by_pump_id")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TASK_NAMES = ["Pump 1", "Pump 2", "Pump 3", "Pump 4", "Pump 5"]

# Mapping modèle → (répertoire expérience, fichier métriques)
MODEL_EXP_MAP = {
    "TinyOL":      ("exp_012_tinyol_pump_by_pump_id",      "metrics.json"),
    "EWC":         ("exp_013_ewc_pump_by_pump_id",          "metrics.json"),
    "HDC":         ("exp_014_hdc_pump_by_pump_id",          "metrics.json"),
    "KMeans":      ("exp_020_kmeans_pump_by_pump_id",        "metrics_kmeans.json"),
    "Mahalanobis": ("exp_015_mahalanobis_pump_by_pump_id",  "metrics_mahalanobis.json"),
    "DBSCAN":      ("exp_021_dbscan_pump_by_pump_id",        "metrics_dbscan.json"),
}

BASE = Path("../../experiments")
RAM_LIMIT = 64 * 1024  # bytes — contrainte STM32N6

results = {}
acc_matrices = {}

for model, (exp_dir, metrics_file) in MODEL_EXP_MAP.items():
    m_path = BASE / exp_dir / "results" / metrics_file
    a_path = BASE / exp_dir / "results" / "acc_matrix.npy"
    if m_path.exists():
        results[model]      = json.loads(m_path.read_text())
        acc_matrices[model] = np.load(a_path, allow_pickle=True)
    else:
        # Fallback mock pour expérience non encore exécutée
        print(f"[MOCK] {model} : {exp_dir} non trouvé — valeurs mock utilisées")
        results[model] = {"aa": 0.5, "af": 0.0, "bwt": 0.0,
                          "ram_peak_bytes": 0, "inference_latency_ms": 0.0, "n_params": 0}
        acc_matrices[model] = np.full((5, 5), 0.5)
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

fig = plot_model_radar(radar_data, title="Comparaison 6 modèles — pump/by_pump_id")
save_figure(fig, FIGURES_DIR / "radar_comparison.png")
plt.show()
```

### Section 3 — Barplot comparaison AA / AF / BWT

```python
fig = plot_metrics_comparison(results, metrics=["aa", "af", "bwt"],
                               task_names=TASK_NAMES)
save_figure(fig, FIGURES_DIR / "barplot_aa_comparison.png")
plt.show()
```

### Section 4 — Grille de matrices d'accuracy (6 modèles, 2×3)

```python
# Matrices 5×5 (5 tâches by_pump_id) — 2×3 subplots
fig, axes = plt.subplots(2, 3, figsize=(21, 12))
for ax, (model, acc_mat) in zip(axes.flat, acc_matrices.items()):
    im = ax.imshow(acc_mat, vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_title(model)
    ax.set_xticks(range(len(TASK_NAMES)))
    ax.set_yticks(range(len(TASK_NAMES)))
    ax.set_xticklabels(TASK_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(TASK_NAMES)
    for i in range(acc_mat.shape[0]):
        for j in range(acc_mat.shape[1]):
            ax.text(j, i, f"{acc_mat[i,j]:.2f}", ha="center", va="center", fontsize=7)
plt.colorbar(im, ax=axes.ravel().tolist())
plt.suptitle("Matrices d'accuracy — pump/by_pump_id (5 tâches)", fontsize=14)
plt.tight_layout()
save_figure(fig, FIGURES_DIR / "acc_matrix_grid.png")
plt.show()
```

### Section 5 — Performance par Pump_ID (barplot comparaison inter-modèles)

```python
# Comparaison inter-modèles : quel modèle retient le mieux chaque Pump_ID ?
results_pump = {
    model: {i+1: float(mat[-1, i]) for i in range(5)}
    for model, mat in acc_matrices.items()
}
fig = plot_performance_by_pump_id_bar(
    results_pump,
    pump_ids=[1, 2, 3, 4, 5],
    title="Accuracy finale par Pump_ID — comparaison 6 modèles"
)
save_figure(fig, FIGURES_DIR / "performance_by_pump_id_bar.png")
plt.show()
```

### Section 6 — Tableau comparatif complet

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
| Mahalanobis | À déterminer | À déterminer | À déterminer | À déterminer | ~0.9 Ko (5 pompes) | À déterminer | — |
| DBSCAN | À déterminer | 0.0000 | À déterminer | À déterminer | >64 Ko ⚠️ probable | À déterminer | — |

> ⚠️ Contexte Dataset 1 : les expériences sur le scénario chronologique (3 tâches)
> donnaient AA ≈ 0.50–0.56 pour tous les modèles. Ces valeurs proches du hasard sont
> attendues et doivent être expliquées, pas ignorées.

### Section 7 — Discussion qualitative (cellule Markdown)

```markdown
## Discussion — pump/by_pump_id (5 tâches)

### Pourquoi les performances approchent-elles du hasard (~0.50) ?
Le Dataset 1 (Large Industrial Pump Maintenance) présente un faible drift inter-Pump_ID :
les signatures de défaut sont similaires pour toutes les pompes (même type d'équipement,
mêmes capteurs). Les 25 features statistiques extraites par fenêtrage ne discriminent pas
suffisamment les états de pompes distinctes.

### Les scénarios granulaires (5 Pump_IDs) sont-ils plus difficiles que le chronologique (3 tâches) ?
À compléter après exécution. Hypothèse : difficulté similaire car le facteur limitant
est la qualité du feature engineering, pas le nombre de tâches.

### Quelles pompes sont structurellement plus difficiles à classifier ?
Voir le barplot `performance_by_pump_id_bar.png` — certains Pump_IDs peuvent avoir des
profils plus séparables si leurs conditions opératoires sont plus distinctes.

### Quel modèle a le meilleur ratio performance/RAM ?
À compléter après exécution. Référence by_equipment (Dataset 2) :
- EWC : AA=0.9824, RAM=1.1 Ko
- Mahalanobis : AA=0.9524, RAM=1.5 Ko

### Contrainte RAM STM32N6 (64 Ko)
À compléter après exécution — DBSCAN probablement > 64 Ko (FIXME gap2).

### Questions scientifiques ouvertes
- FIXME(gap1) : Les AA ≈ 0.50 sont-elles un résultat négatif intéressant (difficulté
  intrinsèque du dataset) ou une limite du feature engineering ? Contraster avec
  FEMTO PRONOSTIA (données plus diversifiées).
- FIXME(gap2) : DBSCAN dépasse probablement 64 Ko — piste buffer borné ou INT8.
- TODO(arnaud) : Les AA proches du hasard sont-elles à mettre en avant comme résultat
  négatif, ou à minimiser dans la présentation ?
- TODO(fred) : Une accuracy de 0.50 sur 5 Pump_IDs est-elle acceptable si AF ≈ 0.01 ?
  Le client préfère-t-il la stabilité à la performance brute ?
```

---

## Métriques attendues (résumé)

Voir tableau Section 6 ci-dessus (à compléter après exécution des expériences exp_012–021).

---

## Critères d'acceptation

- [x] `notebooks/cl_eval/pump_by_pump_id/comparison.ipynb` créé
- [x] Notebook exécutable sans erreur (Restart Kernel + Run All)
- [x] `radar_comparison.png` sauvegardé — radar à 5 axes pour 6 modèles
- [x] `barplot_aa_comparison.png` sauvegardé — barplot AA/AF/BWT
- [x] `acc_matrix_grid.png` sauvegardé — grille 2×3 des matrices 5×5
- [x] `performance_by_pump_id_bar.png` sauvegardé — comparaison inter-modèles par Pump_ID
- [x] `scatter_ram_vs_accuracy.png` sauvegardé — au même format que `pump_single_task.ipynb`
- [x] `scatter_flops_vs_accuracy.png` sauvegardé — MACs via `src.evaluation.compute_macs`
- [x] `scatter_latency_vs_accuracy.png` sauvegardé — latence PC
- [x] Tableau récapitulatif complet avec les 6 modèles et toutes les métriques
- [x] Fallback mock fonctionnel pour les expériences non encore exécutées
- [x] Section discussion avec FIXME(gap1/2) et TODO(arnaud/fred) documentés

---

## Vérification post-création

```bash
python -c "
import nbformat
from pathlib import Path
nb_path = Path('notebooks/cl_eval/pump_by_pump_id/comparison.ipynb')
nb = nbformat.read(nb_path, as_version=4)
print(f'Nombre de cellules : {len(nb.cells)}')
figs = list(Path('notebooks/figures/cl_evaluation/comparison/pump/by_pump_id').glob('*.png'))
print(f'Figures générées : {[f.name for f in figs]}')
assert len(figs) == 7, f'Attendu 7 figures, obtenu {len(figs)}'
print('OK — 7 figures présentes (dont performance_by_pump_id_bar)')
"
```

---

## Questions ouvertes

- `TODO(arnaud)` : Les AA ≈ 0.50 sur Dataset 1 sont-elles à mettre en avant comme un résultat négatif intéressant (difficulté intrinsèque), ou à minimiser dans la présentation ?
- `TODO(arnaud)` : La comparaison chronologique (3 tâches, exp_003/009/010) vs granulaire (5 tâches by_pump_id) est-elle à inclure dans ce notebook ou dans un notebook séparé ?
- `TODO(fred)` : Dans le contexte industriel, une accuracy de 0.50 sur 5 Pump_IDs est-elle acceptable si l'oubli catastrophique est faible (AF ≈ 0.01) ?
- `FIXME(gap1)` : Contraster les résultats Dataset 1 (Kaggle, pump) avec FEMTO PRONOSTIA (INSA, roulements) — les données industrielles réelles sont-elles plus discriminantes ?

---

> **⚠️ Après l'implémentation** : exécuter via "Restart Kernel & Run All Cells". Vérifier
> que les 7 figures sont créées dans `notebooks/figures/cl_evaluation/comparison/pump/by_pump_id/`.
> Mettre à jour `docs/roadmap_phase1.md` en marquant S8-13 comme ✅.
