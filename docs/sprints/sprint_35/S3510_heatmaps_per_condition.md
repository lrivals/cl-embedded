# S3510 — Heatmaps F1 + acc_final par condition (board + PC)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 35 |
| **Priorité** | 🔴 Critique — livrable visuel central du sprint |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 5h |
| **Dépendances** | S3503 (PC), S3508 (board), S3509 (fix HDC), S3504 (F1) |
| **Fichiers cibles** | `scripts/generate_comparison_sprint23.py`, `notebooks/board_benchmark_all_datasets.ipynb`, `docs/figures/gap1_heatmap_{metric}_{condition}_{platform}.png` |
| **Références** | `scripts/generate_comparison_sprint23.py` (`comparison_sprint23.json`), `notebooks/board_benchmark_all_datasets.ipynb:_heatmap_acc` (5×4, masque pending) |

---

## Contexte

La heatmap actuelle est unique : `acc_final` board 5-feat (`gap1_gap2_heatmap_acc.png`).
Le sprint produit **12 heatmaps** : `{F1, acc_final} × {5feat, all, best} × {board, pc}`.

## Spec

- Étendre `generate_comparison_sprint23.py` pour ingérer `exp_S35_PC_*` et `exp_S35_board_*`
  (loaders par condition), exposant `acc_final` **et** `f1_faulty` par cellule, indexés par condition.
- Généraliser `_heatmap_acc(platform, ...)` du notebook en `_heatmap(metric, condition, platform, ...)` :
  matrice **5 datasets × 4 modèles**, `annot=True`, masque pour cellules `pending`/`N/A`,
  même échelle `[0,1]`.
- Produire 12 PNG : `docs/figures/gap1_heatmap_{metric}_{condition}_{platform}.png`
  (`metric ∈ {f1, acc}`, `condition ∈ {5feat, all, best}`, `platform ∈ {board, pc}`).
- HDC×monitoring board affiche la valeur corrigée (S3509), pas 0.113.

**Règle** : aucune valeur en dur dans le notebook — tout vient de `comparison_sprint23.json` ;
cellules non mesurées masquées (grises), pas inventées.

## Implémentation (✅)

- **`generate_comparison_sprint23.py`** : `_load_s35_conditions()` ingère `exp_S35_PC_*`
  (acc←`acc_final`) et `exp_S35_board_*` (acc←`online_accuracy`) → nouvelle clé
  `comparison_sprint23.json["results_by_condition"][condition][dataset][model][platform]`
  exposant `acc_final` **et** `f1_faulty` (None si « à mesurer »). `_apply_s3509_override`
  corrige la cellule legacy monitoring/hdc/board. `results` legacy conservé (rétro-compat).
- **Notebook `board_benchmark_all_datasets.ipynb`** (section « 2bis ») : `_heatmap(metric,
  condition, platform)` (5 datasets × 4 modèles, `annot`, masque pending gris, échelle [0,1]),
  boucle `{f1,acc} × {5feat,all,best} × {board,pc}` → **12 PNG**
  `docs/figures/gap1_heatmap_{metric}_{condition}_{platform}.png` (générés, vérifiés).
- HDC×monitoring board = valeur corrigée S3509 (0.8788), plus 0.113.

## Vérification

```bash
python scripts/generate_comparison_sprint23.py    # régénère comparison_sprint23.json avec S35
jupyter nbconvert --to notebook --execute notebooks/board_benchmark_all_datasets.ipynb
ls docs/figures/gap1_heatmap_*.png   # 12 fichiers
```
