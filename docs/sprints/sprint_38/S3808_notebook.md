# S3808 — Notebook de synthèse

| Champ | Valeur |
|-------|--------|
| **Sprint** | 38 |
| **Priorité** | 🟠 Haute — visualisation de l'arbitrage et du diagnostic drift↔faute. |
| **Statut** | ✅ Implémenté — `notebooks/cl_eval/autonomous_ewc/comparison.ipynb` (nbconvert OK) + 4 PNG. |
| **Durée estimée** | 4h |
| **Dépendances** | S3807 ✅ (`exp_S38_summary.json`) · `src/evaluation/plots.py` ✅ (helpers réutilisés) |
| **Fichiers cibles** | `notebooks/cl_eval/autonomous_ewc/comparison.ipynb`, `docs/figures/sprint38_autonomous_ewc/*.png` |
| **Références** | S3607 (notebook Sprint 36 comme modèle ; nbconvert) |

---

## Contexte

Le notebook lit `exp_S38_summary.json` (lecture seule) et matérialise l'arbitrage et le diagnostic
drift↔faute. Aucune valeur en dur ; tout vient du summary. Exécuté via nbconvert.

## Spec — sections

1. **Économie vs précision** : nuage (latence économisée vs ΔF1) par politique × dataset ; barres
   `update_rate` ; coût RAM du gate (`bss_delta_vs_default`).
2. **Update_rate vs F1** : montre que `gated_*` approche `always` en F1 avec bien moins de mises à jour.
3. **Confusion drift↔faute** : matrice verdict × vérité sur **Monitoring** (drift inter-équipements) et
   **Pronostia** (temporel / première faute) — illustre la désambiguïsation.
4. **Parité board↔PC** : `prediction_parity_rate` + `verdict_parity_rate` par cellule.
5. **Table récap** : 4 politiques × 2 datasets × {pc, board} (précision + économie + Gap 2).

Figures → `docs/figures/sprint38_autonomous_ewc/`.

## Vérification

```bash
jupyter nbconvert --to notebook --execute notebooks/cl_eval/autonomous_ewc/comparison.ipynb
```
- nbconvert sans erreur ; PNG générés ; les cellules `null` (non exécuté) sont masquées proprement.
