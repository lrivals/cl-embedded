# S4-04 — Tableau comparatif final 3 modèles (+ baselines)

| Champ | Valeur |
|-------|--------|
| **ID** | S4-04 |
| **Sprint** | Sprint 4 — Semaine 4 (6–13 mai 2026) |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 3h |
| **Dépendances** | exp_001 (EWC), exp_002 (HDC), exp_003 (TinyOL), exp_004 (TinyOL UINT8) — tous disponibles |
| **Fichiers cibles** | `notebooks/04_final_comparison.ipynb` |
| **Statut** | ✅ Terminé |

---

## Objectif

Produire le notebook de comparaison finale des 3 modèles CL (EWC, HDC, TinyOL) contre les baselines (fine-tuning naïf, joint training) sur le Dataset 2, avec chiffres mesurés et visualisations pour le manuscrit.

Ce notebook est le **livrable principal de la Phase 1** pour la présentation aux encadrants.

**Critère de succès** : notebook exécutable end-to-end (`jupyter nbconvert --to notebook --execute`) sans erreur, avec tableau de résultats complet.

---

## Structure du notebook

### Cellule 0 — En-tête et imports

```python
# notebooks/04_final_comparison.ipynb
# Sprint 4 — Comparaison finale Phase 1
# Dépendances : exp_001, exp_002, exp_003, exp_004
# Auteur : Léonard Rivals — ISAE-SUPAERO (DISC)
# Date : mai 2026

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

EXP_DIR = Path("../experiments")
FIGURE_DIR = Path("../notebooks/figures/sprint4")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
```

### Cellule 1 — Chargement des résultats

```python
EXPERIMENTS = {
    "EWC Online":      EXP_DIR / "exp_001_ewc_dataset2",
    "HDC Online":      EXP_DIR / "exp_002_hdc_dataset2",
    "TinyOL FP32":     EXP_DIR / "exp_003_tinyol_dataset1",
    "TinyOL UINT8":    EXP_DIR / "exp_004_tinyol_uint8",
}

results = {}
for name, path in EXPERIMENTS.items():
    metrics_path = path / "results" / "metrics.json"
    memory_path = path / "results" / "memory_report.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            results[name] = json.load(f)
        with open(memory_path) as f:
            results[name]["memory"] = json.load(f)
```

### Cellule 2 — Tableau comparatif principal

```python
# Tableau AA / AF / BWT / RAM / Latence
table_data = []
for model_name, data in results.items():
    table_data.append({
        "Modèle": model_name,
        "AA": data.get("aa", data.get("uint8_buffer", {}).get("aa", "—")),
        "AF": data.get("af", "—"),
        "BWT": data.get("bwt", "—"),
        "RAM peak (B)": data.get("memory", {}).get("ram_peak_bytes", "—"),
        "Latence inf. (ms)": data.get("inference_latency_ms", "—"),
        "N params": data.get("n_params", "—"),
        "Budget 256Ko": "✅" if data.get("memory", {}).get("within_budget_256ko", False) else "❌",
    })

df = pd.DataFrame(table_data).set_index("Modèle")
display(df.style.highlight_max(subset=["AA"], color="lightgreen")
                .highlight_min(subset=["AF", "RAM peak (B)"], color="lightblue"))
```

### Cellule 3 — Heatmaps des matrices d'accuracy

```python
from src.evaluation.plots import plot_accuracy_matrix

for model_name, exp_path in EXPERIMENTS.items():
    for matrix_file in (exp_path / "results").glob("acc_matrix_*.npy"):
        acc_matrix = np.load(matrix_file)
        fig = plot_accuracy_matrix(acc_matrix, title=f"{model_name} — Matrice accuracy")
        fig.savefig(FIGURE_DIR / f"acc_matrix_{model_name.replace(' ', '_')}.png", dpi=150)
        plt.show()
```

### Cellule 4 — Bar chart RAM comparatif

```python
models = list(results.keys())
ram_bytes = [results[m].get("memory", {}).get("ram_peak_bytes", 0) for m in models]

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(models, [r / 1024 for r in ram_bytes], color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"])
ax.axhline(256, color="red", linestyle="--", label="Budget 256 Ko (NUCLEO-F439ZI)")
ax.axhline(64, color="orange", linestyle="--", label="Budget 64 Ko (STM32N6 cible)")
ax.set_ylabel("RAM peak (Ko)")
ax.set_title("Comparaison RAM — 3 modèles CL")
ax.legend()
fig.savefig(FIGURE_DIR / "ram_comparison.png", dpi=150)
plt.show()
```

### Cellule 5 — Synthèse Triple Gap

```python
# Tableau bilan Gap 1 / Gap 2 / Gap 3
gap_summary = pd.DataFrame({
    "Gap": ["Gap 1 — Données industrielles réelles", "Gap 2 — RAM < 100 Ko mesurée", "Gap 3 — Quantif. INT8 en CL"],
    "Statut": ["⚠️ Partiellement (Dataset 2 peu discriminant)", "✅ EWC : 1.1 Ko · HDC : 14.2 Ko · TinyOL : 6.3 Ko", "🔄 En cours (buffer UINT8 TinyOL — exp_004)"],
    "Exp. clé": ["exp_001, exp_002, exp_003", "exp_001 (RAM 10.4% budget)", "exp_004"],
})
display(gap_summary)
```

### Cellule 6 — Conclusion et prochaines étapes

```markdown
## Conclusion Phase 1

- **M2 EWC** : meilleure précision (AA=0.98), RAM minimale (1.1 Ko update), oubli quasi-nul
- **M3 HDC** : AF=0 par construction, plus simple à porter, AA=0.87
- **M1 TinyOL** : AA=0.56 sur Dataset 1 (backbone peu discriminant sur données anormales)

**Recommandation Phase 2** : porter M2 (EWC) en priorité — meilleur compromis précision/RAM.
M3 (HDC) en parallèle pour sa propriété AF=0. M1 (TinyOL) après si temps disponible.

**Prochaine étape** : Sprint 10+ — portage C sur NUCLEO-F439ZI, mesures DWT, validation MCU.
```

---

## Critères d'acceptation

- [ ] Notebook exécutable sans erreur (`nbconvert --execute`)
- [ ] Tableau comparatif complet (AA, AF, BWT, RAM, latence) pour les 3 modèles disponibles
- [ ] Au moins 3 visualisations : matrice accuracy, bar chart RAM, synthèse Gap
- [ ] Figures sauvegardées dans `notebooks/figures/sprint4/`
- [ ] Cellule de conclusion avec recommandation Phase 2 documentée
- [ ] Gestion propre des expériences manquantes (`.get()` avec valeur par défaut)

---

## Questions ouvertes

- `TODO(arnaud)` : Doit-on inclure le tableau de comparaison Dataset 1 vs Dataset 2 dans ce notebook, ou le réserver pour le manuscrit ?
- `TODO(fred)` : Quels critères Edge Spectrum prioritise pour choisir le modèle à déployer (précision vs RAM vs latence) ?
