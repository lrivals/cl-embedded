# S3511 — Intégration des heatmaps à la présentation

| Champ | Valeur |
|-------|--------|
| **Sprint** | 35 |
| **Priorité** | 🟡 Important — finalité demandée (« ajouter ces plots sur la présentation ») |
| **Statut** | ⬜ À démarrer |
| **Durée estimée** | 2h |
| **Dépendances** | S3510 (12 PNG) |
| **Fichiers cibles** | `docs/presentation_seminaire_juin2026/presentation_plots.ipynb`, `docs/presentation_seminaire_juin2026/01_structure.md`, `docs/presentation_seminaire_juin2026/02_script.md` |
| **Références** | `presentation_plots.ipynb` Slide 6 (`show("gap1_gap2_heatmap_acc.png", ...)`), `01_structure.md:93` (Slide 6) |

---

## Contexte

La Slide 6 montre une seule heatmap accuracy 5-feat. Le sprint ajoute la comparaison F1 vs acc
et l'impact du nombre de features (5feat/all/best), board + PC.

## Spec

- Ajouter dans `presentation_plots.ipynb` les `show(...)` des nouvelles figures (réutiliser le
  helper `show()` existant). Organisation suggérée :
  - **Slide 6** (PC) : `gap1_heatmap_{f1,acc}_{best}_pc.png` (best = vision optimale par modèle).
  - **Nouvelle slide 6bis** (impact features) : panel 5feat vs all vs best, F1 + acc, board.
- Mettre à jour `01_structure.md` (entrée slide) et `02_script.md` (texte d'accompagnement :
  message « accuracy trompeuse → F1 », « 5-feat board vs optimal »).
- Mentionner explicitement le footnote « board = 5 features » et la correction HDC×monitoring.

**Règle** : ne pas dupliquer la logique de génération (les PNG viennent de S3510) ; la présentation
ne fait qu'afficher.

## Vérification

```bash
jupyter nbconvert --to notebook --execute docs/presentation_seminaire_juin2026/presentation_plots.ipynb
grep -c "gap1_heatmap_" docs/presentation_seminaire_juin2026/presentation_plots.ipynb   # > 0
```
