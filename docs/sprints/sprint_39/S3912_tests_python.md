# S3912 — Tests Python (émulateur, ablation, schémas)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 39 |
| **Priorité** | 🟡 Important — jamais absent, jamais 🔴 (règle skill) |
| **Statut** | ✅ Implémenté (1er juillet 2026) — 11/11 PASS ; suite `-k "int8 or quant or emulation"` 40 PASS |
| **Durée estimée** | 2h |
| **Dépendances** | S3902 ✅ · S3904 · S3906 |
| **Fichier cible** | `tests/test_s39_quant.py` |
| **Références** | `tests/test_int8_c_emulation.py` (S3903) · `tests/test_int8_benchmark.py` (Sprint 28, patron) |

---

## Cas de test

| Test | Vérifie |
|------|---------|
| `test_ablation_ladder_monotonic` | F1 non décroissante le long de `ABLATION_LADDER` (chaque correctif ne nuit pas) |
| `test_legacy_degrades` | F1(legacy_c) << F1(fp32) sur données de grande dynamique |
| `test_q15_recovers` | F1(q15) ≥ F1(fp32) − 0.02 (critère Gap 3) |
| `test_sweep_structure` | `summary.json` : 4 modèles × 5 datasets × 5 schémas, clés cohérentes |
| `test_ram_ratios` | RAM int8 = ×4, q15 = ×2 vs fp32 (analytique) |
| `test_hdc_exact` | HDC int8 == fp32 (témoin, Δ=0) |
| `test_lat_proxy_flagged` | toute latence du sweep porte `lat_proxy: true` (pas de mesure inventée) |

## Vérification

```bash
pytest tests/test_s39_quant.py -v
pytest -k "int8 or quant or emulation" -v     # suite quantification complète
```
