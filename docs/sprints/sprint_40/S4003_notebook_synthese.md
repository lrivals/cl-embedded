# S4003 — Notebook de synthèse unifié (figures de l'article)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 40 |
| **Priorité** | 🔴 Critique — source unique des figures/chiffres de l'article |
| **Statut** | 📝 Doc |
| **Durée estimée** | ~8h |
| **Dépendances** | exp_S36_* ✅ · exp_S39_ablation/ ✅ · exp_S39_quant_sweep/ ✅ · exp_S40_board_v2/ (S4002, différé) |
| **Fichiers cibles** | `notebooks/cl_eval/article_ewc_int8/synthesis.ipynb` → `docs/figures/sprint40_article/` |
| **Références** | `notebooks/cl_eval/pc_board_ewc/comparison.ipynb` + helpers `plots.py` (à réutiliser) |

## Contexte

Les données de l'article vivent dans trois campagnes JSON séparées. Ce notebook les **recharge et unifie**
en un jeu de figures cohérent, **source unique** des chiffres du texte (aucune valeur en dur ni dans le
notebook ni dans le `.tex` — tout dérive des JSON). Il réutilise les helpers du notebook Sprint 36 pour
garantir un style homogène.

## Spec

### Sources rechargées
- `experiments/exp_S36_summary.json` + `exp_S36_parity_*.json` (PC↔board FP32 + INT8 legacy, frozen/online).
- `experiments/exp_S39_ablation/{pronostia,monitoring}.json` (échelle d'ablation).
- `experiments/exp_S39_quant_sweep/ewc_{pronostia,monitoring}.json` (schémas + RAM + proxy latence).
- `experiments/exp_S40_board_v2/*.json` (récupération board v2 — **N/A → `"à mesurer"` si absent**).

### Figures (export PNG haute-déf → `docs/figures/sprint40_article/`)
1. **Parité FP32 PC↔board** — frozen (exact 1.000) vs online (0.96–0.99), Pronostia + Monitoring.
2. **Latences Gap 2** — inférence vs inférence+MAJ, PC vs board (barres, échelle µs, ligne 100 ms).
3. **Échelle d'ablation INT8** — `legacy_c → fix_acc32 → per_tensor_calib → per_channel → q15` (F1,
   Pronostia + Monitoring) : illustre le saut +0.88 F1 au scale calibré.
4. **INT8 vs FP32 board** — avant (legacy, effondrement) / après (v2, récupération) : F1 + accord INT8↔FP32
   + RAM ÷4 (Q15 ÷2). Cellules v2 en `"à mesurer"` tant que S4002 non exécuté.
5. **Pareto RAM × F1 × latence** — nuage {FP32, INT8 legacy, per-channel, Q15} annoté.

### Contraintes
- **Aucune valeur en dur** — chaque nombre provient d'un `json.load`. Cellules board absentes → masquées /
  `"à mesurer"` (règle « aucun chiffre inventé »).
- Distinction visuelle **« mesuré board »** (plein) vs **« émulé PC »** (hachuré/gris) sur les figures INT8.
- nbconvert `--execute` doit passer même sans `exp_S40_board_v2/` (dégradation gracieuse).

## Vérification

```bash
jupyter nbconvert --to notebook --execute \
  notebooks/cl_eval/article_ewc_int8/synthesis.ipynb --output synthesis.ipynb
ls docs/figures/sprint40_article/    # PNG générés
```
