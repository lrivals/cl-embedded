# S3909 — Tests Unity host : parité C v2 ↔ émulateur Python

| Champ | Valeur |
|-------|--------|
| **Sprint** | 39 |
| **Priorité** | 🟡 Important — prouve le correctif sans board (`make test` x86) |
| **Statut** | ✅ Implémenté (1er juillet 2026) — `make test` 127 (2 TinyOL préexistants) + `make test-v2-q15` |
| **Durée estimée** | 2h |
| **Dépendances** | S3907 (kernel v2) · S3902 (émulateur, golden vectors) |
| **Fichier cible** | `firmware/stm32f4_blink/tests/test_ewc_int8_v2.c` |
| **Références** | `firmware/stm32f4_blink/tests/test_ewc_int8.c` (patron Unity) · `tests/test_int8_c_emulation.py` |

---

## Contexte

`make test` compile les tests Unity **sur host x86** (TEST_MODE=1) et les exécute — **aucune carte
requise**. C'est le levier clé pour valider le kernel v2 à la maison : on confronte le forward C v2 à des
**golden vectors** produits par l'émulateur Python (S3902), garantissant la parité numérique.

## Cas de test

| Test | Vérifie |
|------|---------|
| `test_v2_no_overflow` | sur un cas où v1 déborde (acc int16), v2 (int32) donne le résultat correct |
| `test_v2_parity_emulator` | logits C v2 ≈ `forward_quant(..., per_channel_int8)` (golden vectors, tol 1e-3) |
| `test_v2_q15_parity` | build `-DEWC_INT8_Q15` ≈ `forward_quant(..., q15)` |
| `test_v2_recovers_f1` | sur un mini-jeu étiqueté, accord v2↔FP32 ≥ accord v1↔FP32 |
| `test_v1_unchanged` | `ewc_head_int8.c` (v1) produit toujours les mêmes valeurs (0 régression A/B) |

## Golden vectors

Générés une fois par l'émulateur et figés dans le test (ou un header `test_vectors_v2.h`) :

```python
# scripts/export_weights_c.py --int8-v2-test-vectors  (réutilise la même calibration)
# émet : entrées X[k], logits FP32, logits per_channel_int8, logits q15
```

## Vérification

```bash
cd firmware/stm32f4_blink && make test
# Attendu : test_ewc_int8_v2 PASS (5/5) ; test_ewc_int8 (v1) inchangé ;
#           2 TinyOL préexistants hors périmètre (connus) ; 0 régression.
```
