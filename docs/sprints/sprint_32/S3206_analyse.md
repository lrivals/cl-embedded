# S3206 — Analyse comparative (notebook)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 32 |
| **Priorité** | 🟡 Moyen |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 4h |
| **Résultat** | `notebooks/cl_eval/threshold_impact/comparison.ipynb` : positive_ratio vs seuil, perf (acc_final/AF/BWT) vs seuil par modèle/dataset, heatmaps modèle×seuil, **invariance HW board** (latence/`.bss` vs seuil) + tables/parité PC↔board. AUROC/F1 non recalculés (non persistés S3204) → documenté honnêtement. Exécuté de bout en bout via nbconvert. |
| **Dépendances** | S3204 (perf/HW PC), S3205 (board) |
| **Fichiers cibles** | `notebooks/cl_eval/threshold_impact/comparison.ipynb` |
| **Références** | `src/evaluation/plots.py`, `experiments/exp_S32_*/results/` |

---

## Contexte

Synthétiser le balayage en une réponse exploitable au `TODO(arnaud)` : quel seuil par dataset, et comment chaque modèle réagit au seuil.

---

## Spec

Contenu du notebook :
- **Courbes perf vs seuil** : F1, AUROC, précision, rappel (une courbe par modèle, facette par dataset).
- **Ratio de positifs vs seuil** : explique la dérive des métriques.
- **Heatmaps** : modèle × seuil pour la métrique clé de chaque dataset.
- **Tables PC↔board** : latence/RAM par seuil + parité.
- **Analyse écrite** :
  - sensibilité de chaque modèle au seuil (lequel est le plus robuste) ;
  - seuil optimal par dataset (compromis F1/rappel) ;
  - trade-off restrictif (peu de positifs, détection tardive) vs permissif (plus de faux positifs) ;
  - **confirmation** que RAM/latence sont invariantes au seuil (preuve empirique S3205).

- Notebook rangé dans `notebooks/` (règle CLAUDE.md). Pas de chiffres hardcodés — tout chargé depuis `experiments/`.

---

## Vérification

```bash
jupyter nbconvert --execute --to notebook --inplace \
    notebooks/cl_eval/threshold_impact/comparison.ipynb
# exécution de bout en bout sans erreur ; figures perf-vs-seuil + tables PC/board produites
```
