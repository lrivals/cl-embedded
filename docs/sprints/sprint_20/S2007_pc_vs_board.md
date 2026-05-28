# S2007 — Comparaison PC vs board : Mahalanobis + EWC delta ≤ 1e-4

| Champ | Valeur |
|-------|--------|
| **Sprint** | 20 |
| **Priorité** | 🟡 Important |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 3h |
| **Dépendances** | S2002 (protocol v3), S2005 (exp EWC complète) |
| **Fichiers cibles** | `scripts/compare_mahalanobis_pc_vs_board.py` |
| **Référence** | `docs/sprints/sprint_19/S1901_mahalanobis_validation.md` |

---

## Contexte

`compare_mahalanobis_pc_vs_board.py` existe en skeleton (Sprint 19).
Il faut le compléter pour comparer les scores PC (Python/NumPy FP64) vs board (C FP32) sur le même dataset, et étendre la comparaison à EWC (logits de sortie).

---

## Ce qu'il faut compléter

### Logique du script

```python
# 1. Charger 500 samples CWRU (3 tâches)
# 2. Forward Python : MahalanobisDetector.score(x) → scores_pc[]
# 3. Envoyer les 500 frames via sensor_stream.py (protocol v2/v3)
# 4. Récupérer scores_board[] depuis les réponses UART (ou dry-run)
# 5. Calculer delta = max|scores_pc - scores_board|
# 6. Asserter delta ≤ TOLERANCE (1e-4 FP32 strict, ou 1% pour FP32 vs FP64)
```

### Sortie

```json
{
  "model": "mahalanobis",
  "n_samples": 500,
  "max_abs_delta": 3.2e-6,
  "mean_abs_delta": 1.1e-7,
  "tolerance": 1e-4,
  "compliant": true,
  "platform": "dry_run"
}
```

### Extension EWC

Même structure pour comparer les **logits** (avant softmax) du forward EWC :
- Python : `ewc_mlp.forward(x).detach().numpy()`
- Board : parser les bytes `conf:f32` de la réponse v3

---

## Vérification

- [ ] Script s'exécute en mode `--dry-run` et produit `comparison_results.json`
- [ ] `compliant: true` pour Mahalanobis (delta ≤ 1e-4)
- [ ] `compliant: true` pour EWC logits (delta ≤ 1e-3 tolérable pour FP32 backprop)
- [ ] Résultats ajoutés dans `experiments/exp_S19_01/` et `experiments/exp_S19_02/`

---

## Résultats dry-run (2026-05-27)

| Modèle | max_abs_delta | Tolérance | compliant | platform |
| --- | --- | --- | --- | --- |
| mahalanobis | 8.35e-07 | 1e-4 | ✅ true | dry_run |
| ewc | 5.25e-08 | 1e-3 | ✅ true | dry_run |

- `experiments/exp_S19_01/comparison_results.json` — Mahalanobis FP64 vs FP32
- `experiments/exp_S19_02/comparison_results.json` — EWC sigmoid float64 vs float32

## Questions ouvertes

- `TODO(arnaud)` : Tolérance 1e-4 ou 1% ? (FP32 C vs FP64 Python peut introduire 1–2 ULP)
- `TODO(arnaud)` : Comparer aussi les **prédictions** (0/1) ou uniquement les scores flottants ?
