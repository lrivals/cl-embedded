# S8-14 — Notebook Comparaison — pump_by_temporal_window

| Champ | Valeur |
|-------|--------|
| **ID** | S8-14 |
| **Sprint** | Sprint 8 — Phase 1 Extension |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 3h |
| **Dépendances** | S8-07 (TinyOL), S8-08 (EWC), S8-09 (HDC), S8-10 (KMeans), S8-11 (Mahalanobis), S8-12 (DBSCAN) |
| **Fichier cible** | `notebooks/cl_eval/pump_by_temporal_window/comparison.ipynb` |
| **Statut** | ✅ Terminé |

---

## Objectif

Créer le notebook de **comparaison des 6 modèles** sur le scénario **pump_by_temporal_window**
(Q1 0–5k h → Q2 5k–10k h → Q3 10k–15k h → Q4 15k–20k h, 4 tâches). Ce notebook
documente l'évolution des performances au fil du temps opérationnel de la pompe.

**Critère de succès** : notebook exécutable du début à la fin sans erreur, avec toutes
les figures comparatives sauvegardées dans
`notebooks/figures/cl_evaluation/comparison/pump/by_temporal_window/`, et un tableau
récapitulatif complet AA/AF/BWT/AUROC/RAM/latence pour les 6 modèles.

---

## Structure de sortie attendue

```
notebooks/cl_eval/pump_by_temporal_window/
└── comparison.ipynb                    ← notebook créé

notebooks/figures/cl_evaluation/comparison/pump/by_temporal_window/
├── radar_comparison.png
├── barplot_aa_comparison.png
├── acc_matrix_grid.png
├── scatter_ram_vs_accuracy.png          ← cohérence baseline (Gap 2)
├── scatter_flops_vs_accuracy.png        ← cohérence baseline (coût portable)
└── scatter_latency_vs_accuracy.png      ← cohérence baseline (latence PC)
```

---

## Figures requises (cohérence avec baseline)

En plus des trois figures comparatives classiques (radar, barplot, matrice), ce notebook **doit**
produire les trois scatters `accuracy vs <ressource>` **au même format que
`pump_single_task.ipynb`** pour permettre la comparaison directe avec la baseline hors-CL :

- `scatter_ram_vs_accuracy.png` — trade-off RAM (Gap 2, budget STM32 ≤ 64 Ko)
- `scatter_flops_vs_accuracy.png` — coût de calcul portable (MACs via `src.evaluation.compute_macs`)
- `scatter_latency_vs_accuracy.png` — latence PC (budget 100 ms, ⚠ non transférable au MCU)

Les trois scatters utilisent le même dict `SCATTER_MARKERS` que la baseline.

---

## Contenu du notebook (7 sections)

### Section 0 — En-tête (cellule Markdown)

```markdown
# Comparaison 6 modèles — Dataset 1 Pump Maintenance — by_temporal_window

| Champ | Valeur |
|-------|--------|
| **Scénario** | by_temporal_window : Q1 (0–5k h) → Q2 (5k–10k h) → Q3 (10k–15k h) → Q4 (15k–20k h) — 4 tâches |
| **Modèles** | TinyOL · EWC · HDC · KMeans · Mahalanobis · DBSCAN |
| **Dataset** | Large_Industrial_Pump_Maintenance_Dataset.csv |
| **Date** | {date d'exécution} |

Ce notebook agrège les résultats des expériences exp_024, exp_025, exp_026, exp_028, exp_027, exp_029.
Voir `notebooks/cl_eval/pump_by_pump_id/comparison.ipynb` pour le scénario by_pump_id.
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

FIGURES_DIR = Path("../../notebooks/figures/cl_evaluation/comparison/pump/by_temporal_window")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TASK_NAMES = [
    "Q1 (0–5k h)",
    "Q2 (5k–10k h)",
    "Q3 (10k–15k h)",
    "Q4 (15k–20k h)",
]

# Mapping modèle → (répertoire expérience, fichier métriques)
MODEL_EXP_MAP = {
    "TinyOL":      ("exp_024_tinyol_pump_by_temporal_window",      "metrics.json"),
    "EWC":         ("exp_025_ewc_pump_by_temporal_window",          "metrics.json"),
    "HDC":         ("exp_026_hdc_pump_by_temporal_window",          "metrics.json"),
    "KMeans":      ("exp_028_kmeans_pump_by_temporal_window",        "metrics_kmeans.json"),
    "Mahalanobis": ("exp_027_mahalanobis_pump_by_temporal_window",  "metrics_mahalanobis.json"),
    "DBSCAN":      ("exp_029_dbscan_pump_by_temporal_window",        "metrics_dbscan.json"),
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
        acc_matrices[model] = np.full((4, 4), 0.5)
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

fig = plot_model_radar(radar_data, title="Comparaison 6 modèles — pump/by_temporal_window")
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
# Matrices 4×4 (4 tâches by_temporal_window) — 2×3 subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, (model, acc_mat) in zip(axes.flat, acc_matrices.items()):
    im = ax.imshow(acc_mat, vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_title(model)
    ax.set_xticks(range(len(TASK_NAMES)))
    ax.set_yticks(range(len(TASK_NAMES)))
    ax.set_xticklabels(TASK_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(TASK_NAMES)
    for i in range(acc_mat.shape[0]):
        for j in range(acc_mat.shape[1]):
            ax.text(j, i, f"{acc_mat[i,j]:.2f}", ha="center", va="center", fontsize=8)
plt.colorbar(im, ax=axes.ravel().tolist())
plt.suptitle("Matrices d'accuracy — pump/by_temporal_window (4 tâches)", fontsize=14)
plt.tight_layout()
save_figure(fig, FIGURES_DIR / "acc_matrix_grid.png")
plt.show()
```

### Section 5 — Évolution de l'oubli sur T1 au fil du temps (comparaison inter-modèles)

```python
# Visualiser si les modèles "oublient" Q1 au fur et à mesure qu'ils apprennent Q2, Q3, Q4
fig, ax = plt.subplots(figsize=(10, 6))
for model, acc_mat in acc_matrices.items():
    t1_over_time = acc_mat[:, 0]  # accuracy sur T1 après chaque tâche d'entraînement
    ax.plot(range(len(t1_over_time)), t1_over_time, marker='o', label=model)
ax.set_xlabel("Tâche d'entraînement courante")
ax.set_ylabel("Accuracy sur T1 (Q1 : 0–5k heures)")
ax.set_xticks(range(len(TASK_NAMES)))
ax.set_xticklabels(TASK_NAMES, rotation=30, ha="right")
ax.set_title("Oubli catastrophique T1 au fil du temps — comparaison 6 modèles")
ax.legend()
ax.grid(True, alpha=0.3)
# Note : figure illustrative — pas sauvegardée dans les figures de comparaison principales
plt.tight_layout()
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
| Mahalanobis | À déterminer | À déterminer | À déterminer | À déterminer | ~0.7 Ko (4 fenêtres) | À déterminer | — |
| DBSCAN | À déterminer | 0.0000 | À déterminer | À déterminer | >64 Ko ⚠️ probable | À déterminer | — |

> ⚠️ Contexte Dataset 1 : AA ≈ 0.50–0.56 attendu (cohérent avec le scénario chronologique
> 3 tâches et le scénario by_pump_id). Le scénario temporel peut présenter un drift plus
> progressif — à vérifier si cela aide ou non les modèles CL.

### Section 7 — Discussion qualitative (cellule Markdown)

```markdown
## Discussion — pump/by_temporal_window (4 tâches)

### Pourquoi les performances approchent-elles du hasard (~0.50) ?
Même cause que pour by_pump_id : le feature engineering (25 stats) ne discrimine pas
suffisamment les états de maintenance sur ce dataset. Le drift temporel (dégradation
progressive) pourrait être trop subtil pour les features statistiques actuelles.

### Le scénario temporel est-il plus difficile que by_pump_id (5 tâches) ?
À compléter après exécution. Hypothèse : le drift temporel progressif (dégradation de la
pompe) est plus continu que le drift inter-Pump_ID, rendant les frontières de tâches floues.

### Comparaison by_pump_id vs by_temporal_window
Comparer les métriques AA/AF/BWT entre les deux scénarios pour identifier si le drift
par Pump_ID ou le drift temporel est plus difficile pour les modèles CL.
Cette comparaison est un résultat original (Gap 1).

### Y a-t-il de l'oubli catastrophique sur T1 (Q1) ?
Voir la visualisation Section 5. Pour un scénario de maintenance prédictive, oublier les
patterns de défaut de la phase initiale (Q1, pompe récente) peut être problématique si
les défauts précoces sont différents des défauts tardifs.

### Contrainte RAM STM32N6 (64 Ko)
À compléter après exécution — DBSCAN probablement > 64 Ko (FIXME gap2).

### Questions scientifiques ouvertes
- FIXME(gap1) : Le drift temporel progressif (dégradation d'une pompe sur 20k heures)
  est-il représentatif des conditions réelles de maintenance ? Contraster avec FEMTO
  PRONOSTIA (INSA, roulements à billes sous charge contrôlée).
- FIXME(gap2) : DBSCAN dépasse probablement 64 Ko — piste buffer borné ou INT8.
- TODO(arnaud) : Les AA ≈ 0.50 sur le scénario temporel sont-elles plus faciles à
  expliquer scientifiquement (drift subtil) que les AA ≈ 0.50 sur by_pump_id ?
- TODO(fred) : Dans un contexte industriel, les patterns de défaut varient-ils
  significativement entre le début de vie (Q1) et la fin de vie (Q4) d'une pompe ?
```

---

## Métriques attendues (résumé)

Voir tableau Section 6 ci-dessus (à compléter après exécution des expériences exp_024–029).

---

## Critères d'acceptation

- [x] `notebooks/cl_eval/pump_by_temporal_window/comparison.ipynb` créé
- [x] Notebook exécutable sans erreur (Restart Kernel + Run All)
- [x] `radar_comparison.png` sauvegardé — radar à 5 axes pour 6 modèles
- [x] `barplot_aa_comparison.png` sauvegardé — barplot AA/AF/BWT
- [x] `acc_matrix_grid.png` sauvegardé — grille 2×3 des matrices 4×4
- [x] `scatter_ram_vs_accuracy.png` sauvegardé — au même format que `pump_single_task.ipynb`
- [x] `scatter_flops_vs_accuracy.png` sauvegardé — MACs via `src.evaluation.compute_macs`
- [x] `scatter_latency_vs_accuracy.png` sauvegardé — latence PC
- [x] Tableau récapitulatif complet avec les 6 modèles et toutes les métriques
- [x] Visualisation évolution T1 au fil du temps présente (Section 5)
- [x] Fallback mock fonctionnel pour les expériences non encore exécutées
- [x] Section discussion avec FIXME(gap1/2) et TODO(arnaud/fred) documentés
- [x] Comparaison by_pump_id vs by_temporal_window mentionnée dans la discussion

---

## Vérification post-création

```bash
python -c "
import nbformat
from pathlib import Path
nb_path = Path('notebooks/cl_eval/pump_by_temporal_window/comparison.ipynb')
nb = nbformat.read(nb_path, as_version=4)
print(f'Nombre de cellules : {len(nb.cells)}')
figs = list(Path('notebooks/figures/cl_evaluation/comparison/pump/by_temporal_window').glob('*.png'))
print(f'Figures générées : {[f.name for f in figs]}')
assert len(figs) == 6, f'Attendu 6 figures, obtenu {len(figs)}'
print('OK — 6 figures présentes')
"
```

---

## Questions ouvertes

- `TODO(arnaud)` : Le drift temporel progressif (dégradation 0–20k heures) est-il à documenter comme un type de drift différent du drift par type d'équipement (Dataset 2) ?
- `TODO(arnaud)` : La visualisation de l'oubli sur T1 (Section 5) est-elle pertinente à inclure dans le manuscrit, ou uniquement à titre exploratoire ?
- `TODO(fred)` : Les patterns de défaut changent-ils qualitativement entre début de vie (Q1) et fin de vie (Q4) pour les pompes industrielles suivies par Edge Spectrum ?
- `FIXME(gap1)` : Contraster les résultats Dataset 1 temporel (Kaggle) avec FEMTO PRONOSTIA (INSA) — les données de dégradation réelle sont-elles plus discriminantes ?

---

> **⚠️ Après l'implémentation** : exécuter via "Restart Kernel & Run All Cells". Vérifier
> que les 6 figures sont créées dans `notebooks/figures/cl_evaluation/comparison/pump/by_temporal_window/`.
> Mettre à jour `docs/roadmap_phase1.md` en marquant S8-14 comme ✅.
