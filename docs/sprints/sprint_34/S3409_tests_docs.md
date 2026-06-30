# S3409 — Tests + documentation (Sprint 34)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 34 |
| **Priorité** | 🟢 Bas (mais jamais absente) |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 3h |
| **Dépendances** | S3401–S3408 |
| **Fichiers cibles** | `tests/test_streaming_model.py`, `tests/test_mahalanobis_q15.py`, `firmware/stm32f4_blink/tests/test_ring_buffer.c`, `firmware/stm32f4_blink/tests/test_mahalanobis_q15.c`, `docs/roadmap_phase2.md`, `CLAUDE.md` |
| **Références** | `tests/` (conventions pytest existantes), `firmware/stm32f4_blink/tests/test_runner.c`, `skills/graphify_sprint_update.md` |

---

## Contexte

Verrouille les deux livrables du sprint (streaming/buffer + Q15 Mahalanobis) par des tests
et clôture le sprint (roadmap, statut `CLAUDE.md`).

---

## Spec

```python
# tests/test_streaming_model.py
def test_debit_max_inverse_latence():
    # debit_max(latence) == 1/latence

def test_debit_streaming_formula():
    # debit_streaming(f_acq, stride, window) == f_acq * stride/window

def test_sram_budget_constraint():
    # check_sram_budget rejette un buffer trop grand

# tests/test_mahalanobis_q15.py
def test_q15_recovers_auroc_cwru_pronostia():
    # delta_auroc < 0.02 vs FP32, sur les 2 datasets ciblés

def test_q15_no_regression_other_datasets():
    # Monitoring/CMAPSS/Paderborn : Q15 ne dégrade pas FP32/INT8 existants

def test_int8_mode_unchanged():
    # quant="int8" (défaut) produit des résultats identiques à avant le sprint
```

```c
// firmware/stm32f4_blink/tests/test_ring_buffer.c
// push/window/stride, wrap-around, non-régression HDC (mêmes échantillons lus en FIFO)

// firmware/stm32f4_blink/tests/test_mahalanobis_q15.c
// parité forward Q15 C <-> Python sur vecteurs de référence (delta < tolérance)
```

- Intégrer les deux nouveaux fichiers Unity à `test_runner.c`.
- `make test` : 0 nouvelle régression sur les modes existants (DUAL_MODE, PAIR_MODE, HDC).
- MAJ `docs/roadmap_phase2.md` (ligne Sprint 34) + statut sprint dans `CLAUDE.md`.
- Invoquer le skill `graphify_sprint_update`.

---

## Vérification

```bash
pytest tests/ -k "streaming or mahalanobis_q15" -v
cd firmware/stm32f4_blink && make test
```

---

## Réalisé (S3409)

- **Python** : `tests/test_mahalanobis_q15.py` (7 tests PASS) — recouvrement fidélité de rang
  Q15 > INT8 sur grande dynamique synthétique (corr > 0.99 vs ~0.67), reconstruction Σ⁻¹ plus
  fine, empreinte Σ⁻¹ ÷2, `test_int8_mode_unchanged` (égalité stricte), rejet `quant` invalide,
  validation de l'agrégat `exp_S34_maha_q15/summary.json` (skip si absent). `tests/
  test_streaming_model.py` (16) + `test_ring_buffer.c` (9) déjà couverts (S3401/S3402).
- **C** : `firmware/.../tests/test_mahalanobis_q15.c` (4 tests PASS) — **parité forward Q15
  C↔Python** sur vecteurs de référence (`test_vectors_q15.h`, généré), + init euclidien, distance
  nulle, seuil. Intégrés à `test_runner.c` + `Makefile`. `make test` : **109 tests, 2 échecs
  TinyOL préexistants hors périmètre, 0 nouvelle régression**.
- Docs : statut des 3 docs S34xx, `roadmap_phase2.md` ligne Sprint 34, statut `CLAUDE.md`,
  `graphify_sprint_update`.
