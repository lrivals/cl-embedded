# exp_S27_01 — DUAL_MODE board (RUL + Faute simultanés)

**Board** : NUCLEO-F439ZI (Cortex-M4 @ 180 MHz, 256 Ko SRAM) · **Date** : 2026-06-12
**Mode** : DUAL_MODE (1 trame UART → `g_ewc_reg` + `g_ewc_mc` en séquence → réponse 25 B)
**Flags** : `0x71` (DUAL_MODE | UPDATE | PROFILING) · `--consolidate-at 50` · `n_samples=200`

## Résultats mesurés (board réelle)

| Métrique | Valeur board | Critère | Statut |
|----------|--------------|---------|--------|
| `rmse_rul_offline` (RUL, 200 samples) | **22.59 cycles** | < 24.3 (±15 % vs S26 RMSE=21.15) | ✅ |
| `rmse_rul` board (online, dernier sample) | 7.25 cycles | cohérent (online, post-convergence) | ✅ |
| `f1_fault_offline` (faute) | **0.072** | ≥ 0.50 attendu | ❌ — voir `FIXME(gap1)` |
| `f1_fault` board (online, dernier sample) | 0.041 | — | — |
| `lat_mean_us` | **639 µs** | < 1 000 µs (Gap 2 << 100 ms) | ✅ |
| `lat_p99_us` | **788 µs** | < 2 000 µs | ✅ |
| `.bss` | **66 748 B** (25.5 %) | < 256 Ko | ✅ |
| `forgetting` | 0.0 | — | — |

## Interprétation

- **RUL préservé** (RMSE 22.59 ≈ single-mode S26 21.15) : `g_ewc_reg` lit `features[0:5]` = top-5 CMAPSS FD001 purs, donc le forward RUL est intact en mode dual. ✅
- **Latence Gap 2 largement satisfaite** : 639 µs moyen / 788 µs P99, soit ~155× sous le budget 100 ms. Le coût dual ≈ somme des deux forwards séquentiels (RUL + MC), pas de surcoût structurel.
- **F1_faute = 0.072 ❌ — `FIXME(gap1)`, pas un bug de portage** : la trame DUAL_MODE mélange 5 features CMAPSS (RUL) + 4 features CWRU (faute) sur 9 slots. `g_ewc_mc` (entraîné sur 9 features CWRU pures) ne reçoit donc que 4/9 slots dans son domaine — les 5 premiers slots sont hors-distribution. La parité numérique board↔PC reste exacte (RMSE_off=22.59, F1_off=0.072 cohérents board/offline) : c'est une **limitation de construction du dataset mixte**, pas un défaut d'implémentation. Résolution prévue : dataset unifié Pronostia (Sprint 28).

## Fichiers

- `dual_results.json` — 200 samples + métriques board/offline
- `config_snapshot.yaml` — paramètres firmware + protocole
- `summary.md` — ce fichier
