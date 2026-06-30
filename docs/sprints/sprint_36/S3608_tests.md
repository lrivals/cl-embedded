# S3608 — Tests

| Champ | Valeur |
|-------|--------|
| **Sprint** | 36 |
| **Priorité** | 🟢 Low — jamais bloquant (règle `sprint_generation.md`) mais jamais absent. |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 2h |
| **Dépendances** | S3606 ✅ (`exp_S36_summary.json`), S3605 ✅ (parité) · `src/evaluation/metrics.py` ✅ |
| **Fichiers cibles** | `tests/test_sprint36_comparison.py` |
| **Références** | `tests/test_threshold_sweep.py`, `tests/test_heatmap_builders.py` (gabarits structure/parité) |

---

## Contexte

Vérifier la **structure** et la **cohérence** des artefacts produits (pas les valeurs
mesurées, inconnues tant que la board n'a pas tourné), à l'image de `test_threshold_sweep.py`.
Le firmware EWC étant **inchangé**, Unity doit rester à 0 régression.

## Spec

`tests/test_sprint36_comparison.py` :

- `test_summary_structure` — `exp_S36_summary.json` indexé `[dataset][condition][platform]`, datasets = {pronostia, monitoring}, conditions = {5feat, all}.
- `test_parity_frozen_is_exact_when_present` — pour les fichiers `exp_S36_parity_*_frozen_*`, si `parity_rate` renseigné ⇒ `== 1.0` (parité exacte poids gelés).
- `test_parity_table_shape` — chaque `rows[i]` contient `{idx, true, pred_pc, pred_board, match}` ; `match == (pred_pc == pred_board)`.
- `test_metrics_keys_present` — chaque entrée PC expose `{acc_final, af, f1_faulty, roc_auc, ram_peak_bytes}` ; board expose `{latency_us_p50, parity_rate}`.
- `test_gap2_latencies` — toutes les latences renseignées < 100 000 µs (Gap 2).
- (tests robustes aux champs `null` : `skip`/`xfail` documenté tant que non mesuré).

**Note Unity firmware** : `make test` doit rester vert (103/105, 2 TinyOL préexistants hors périmètre) — **EWC non modifié** par ce sprint.

## Vérification

```bash
pytest tests/test_sprint36_comparison.py -v
cd firmware/stm32f4_blink && make test     # 0 régression attendue
```

## Implémentation (✅)

- [x] `tests/test_sprint36_comparison.py` : `test_summary_structure`, `test_metrics_keys_present`,
      `test_gap2_latencies`, `test_parity_frozen_is_exact_when_present`, `test_parity_table_shape`,
      `test_parity_online_is_approx_class` — **6/6 PASS**, robustes aux champs `null` (skip si artefact absent).
- [x] Unity `make test` : **112 tests, 2 échecs = les 2 TinyOL préexistants** (hors périmètre,
      `test_tinyol_predict_normal_zero_weights` + `test_tinyol_forward_delta`) ⇒ **0 régression** (EWC inchangé).
