# S2914 — Notebook grille board 4×5 + synthèse 20/20

| Champ | Valeur |
|-------|--------|
| **Sprint** | 29 (extension O8 — board 4×5 complet) |
| **Priorité** | 🟡 |
| **Statut** | ✅ Implémenté (28 juin 2026) — notebook Section 2 généralisé 20 cellules + heatmaps board 4×5 (2.5) + Section 4 mise à jour ; nbconvert OK, `board_gap3_heatmaps.png` généré |
| **Durée estimée** | 2h |
| **Dépendances** | S2913 ✅ (20 JSON board) |
| **Fichiers cibles** | `notebooks/sprint29_int8_board.ipynb` (Section 2 + Section 4) · `experiments/figures/sprint29/*.png` |

---

## Contexte

Avec les 20 cellules board produites (S2913), le notebook `sprint29_int8_board.ipynb` doit
présenter une **grille board 4 modèles × 5 datasets** symétrique à la grille PC (Section 1),
pour une comparaison PC↔board directe.

---

## Spécification

### Section 2 — chargement généralisé + heatmaps board 4×5

- Généraliser le chargement du DataFrame `df` (cellule d'entête) aux **20 JSON**
  `exp_S29_board_int8/results_*.json` (au lieu de 5), en tolérant `metric_value: null`
  (cellules N/A).
- Ajouter une **heatmap board 4×5** (modèle × dataset) pour : métrique INT8, ratio RAM,
  ratio latence — sur le modèle de la heatmap PC (`pc_pivot_delta` / `pc_pivot_ram`,
  Section 1), avec **N/A annoté en gris** (même style que la heatmap PC Paderborn).
- Conserver les barplots existants (2.1–2.4) comme vue détaillée par couple.

### Section 4 — synthèse PC ∩ board 20/20

- Mettre à jour la synthèse pour refléter **4 modèles sur board** (Mahalanobis inclus) et
  les cellules N/A.
- Contextualiser honnêtement : Mahalanobis INT8 dégradé (`sigma_inv_` grande dynamique) →
  **Q15 = fallback board recommandé** (Sprint 34) ; latence INT8 négative sur Cortex-M4 FPU
  confirmée sur les 4 familles.

---

## Critères d'acceptation

- `jupyter nbconvert --to notebook --execute --inplace notebooks/sprint29_int8_board.ipynb`
  → exécution complète sans erreur.
- Heatmap board 4×5 rendue (20 cellules, N/A en gris), comparable visuellement à la PC.
- Figures régénérées dans `experiments/figures/sprint29/`.
- Aucun chiffre en dur : tout dérive des JSON via `df`.

---

## Suivi

- Mettre à jour `docs/roadmap_phase2.md` (Sprint 29 → O8 ✅) et le statut Sprint dans
  `CLAUDE.md`.
- Invoquer `graphify_sprint_update` après finalisation.
