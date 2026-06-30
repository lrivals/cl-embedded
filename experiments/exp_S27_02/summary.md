# exp_S27_02 — Comparaison latence : single vs dual

**Board** : NUCLEO-F439ZI (Cortex-M4 @ 180 MHz) · **Date** : 2026-06-12 · `n_samples=100` par mode, online learning (UPDATE) activé.

## Latences mesurées (board réelle)

| Mode | Modèle(s) | Flags | Latence moyenne µs | Latence P99 µs |
|------|-----------|-------|-------------------:|---------------:|
| Single RUL | EWC Reg | `0x51` | **234** | 234 |
| Single MC | EWC MC | `0x31` | **403** | 491 |
| **DUAL** | EWC Reg + MC | `0x71` | **637** | 637 |

- **Somme des single** : 234 + 403 = **637 µs**
- **Latence dual mesurée** : **637 µs**
- **Overhead dual** : **≈ 0 µs** (−0.0) — le mode DUAL est une exécution séquentielle pure des deux forwards+updates ; le coût des comparaisons de flags + métriques duales est négligeable.

## Validation Gap 2

| Critère | Valeur | Statut |
|---------|--------|--------|
| Latence combinée < 100 ms | 637 µs (~155× sous budget) | ✅ |
| Overhead vs single séquentiel | ~0 µs | ✅ |

## Non-régression Sprint 26

Les modes single restent corrects sur le firmware Sprint 27 (rétro-compatible) :

- **RUL single** : réponse 21 B, RMSE board = 22.22 (≈ S26 21.15) ✅
- **MC single** : réponse 21 B, F1-macro = 0.434 — conforme au comportement S26 (oubli catastrophique du modèle EWC, déjà documenté `FIXME(gap1)`, hors périmètre portage) ✅

## Fichiers

- `rul_single.json` / `mc_single.json` / `dual_results.json` — résultats bruts par mode
- `latency_comparison.json` — agrégat (modes + overhead + gap2)
- `config_snapshot.yaml` — flags & commandes
- `summary.md` — ce fichier
