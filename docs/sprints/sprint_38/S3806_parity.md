# S3806 — Parité board ↔ PC (verdicts + prédictions)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 38 |
| **Priorité** | 🟠 Haute — atteste que le gate embarqué reproduit la décision PC (honnêteté board). |
| **Statut** | ✅ Implémenté — `scripts/board_pc_parity38.py` ; 16 fichiers `exp_S38_parity_*.json`. |
| **Durée estimée** | 3h |
| **Dépendances** | S3802 ✅ (dump PC : pred + verdict par échantillon) · S3804/S3805 ✅ (dumps board) · `scripts/board_pc_parity.py` ✅ (modèle) |
| **Fichiers cibles** | `scripts/board_pc_parity38.py`, `experiments/exp_S38_parity_{policy}_{dataset}.json` |
| **Références** | S3605 (parité par échantillon Sprint 36 comme modèle) |

---

## Contexte

Deux niveaux de parité (lecture seule, aucune métrique recalculée à partir de zéro) :
- **prédiction** : `pred_board` vs `pred_pc` (exacte en frozen ; approchée sinon — float32 vs float64).
- **verdict du gate** : `verdict_board` vs `verdict_pc` ∈ {NORMAL, DRIFT, FAULT} — spécifique au Sprint 38,
  atteste que la **décision d'update** est identique des deux côtés (mêmes seuils exportés).

## Spec

Pour chaque `(policy, dataset)` :
1. Charger les `samples` PC (S3802) et board (S3804/S3805).
2. Table par échantillon : `[idx, true, pred_pc, pred_board, verdict_pc, verdict_board, pred_match, verdict_match]`.
3. Agrégats : `prediction_parity_rate`, `verdict_parity_rate`, listes `pred_mismatches` / `verdict_mismatches`
   (idx + valeurs), matrice de confusion verdict_pc × verdict_board.
4. → `experiments/exp_S38_parity_{policy}_{dataset}.json`.

**Règles** : frozen → `prediction_parity_rate == 1.0` attendu ; gated → parité verdicts élevée, les
mismatches se concentrent aux frontières de seuil (arrondi float). Aucun chiffre inventé.

## Vérification

```bash
python scripts/board_pc_parity38.py            # → tous les exp_S38_parity_*.json
```
- frozen : prédiction 1.000.
- gated_* : `verdict_parity_rate` proche de 1 ; mismatches documentés (frontières de décision).
