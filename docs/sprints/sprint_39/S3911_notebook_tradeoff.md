# S3911 — Notebook trade-off + heatmaps ablation

| Champ | Valeur |
|-------|--------|
| **Sprint** | 39 |
| **Priorité** | 🟡 Important — synthèse visuelle pour le manuscrit |
| **Statut** | ✅ Implémenté (1er juillet 2026) — nbconvert OK, 5 PNG produits |
| **Durée estimée** | 2h |
| **Dépendances** | S3906 ✅ (`exp_S39_quant_sweep/`) · S3904 ✅ (`exp_S39_ablation/`) |
| **Fichier cible** | `notebooks/cl_eval/int8_intermediate/comparison.ipynb` |
| **Références** | `src/evaluation/plots.py` (helpers) · notebooks `sprint29_int8_board.ipynb`, `sprint36` (patrons heatmaps) |

---

## Contexte

Rassembler les résultats PC (ablation S3904 + sweep S3906) en figures lisibles répondant aux questions du
sprint : *où est la perte ?* (ablation) et *quel schéma équilibre latence/RAM/accuracy ?* (trade-off).

## Sections du notebook

1. **Ablation par facteur** — barres empilées Δ F1 le long de `ABLATION_LADDER` (par dataset) → identifie
   le facteur dominant.
2. **Trade-off scatter** — axe X = RAM (×vs FP32), axe Y = métrique, taille/couleur = proxy latence ;
   un point par schéma. Front de Pareto annoté.
3. **Heatmaps 4×5** — métrique par (modèle × dataset) pour chaque schéma (legacy / per-channel / q15),
   symétriques aux heatmaps Sprint 35/36 (N/A en gris pour HDC exact / cas mono-classe).
4. **Récap recommandation** — table : par modèle, schéma recommandé (ex. EWC → q15 si RAM ×2 acceptable).

## Sortie

PNG dans `docs/figures/sprint39_int8_intermediate/` : `ablation_factors.png`, `tradeoff_pareto.png`,
`heatmap_{scheme}.png`. nbconvert exécuté (pas de valeur en dur — tout depuis les JSON).

## Vérification

```bash
jupyter nbconvert --execute --to notebook --inplace \
    notebooks/cl_eval/int8_intermediate/comparison.ipynb
ls docs/figures/sprint39_int8_intermediate/
```
