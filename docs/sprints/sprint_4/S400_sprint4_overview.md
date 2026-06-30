# Sprint 4 — Vue d'ensemble : Extension UINT8 + Comparaison finale 3 modèles

> **Période** : Semaine 4 (6–13 mai 2026) — exécuté en Sprint 22 (juin 2026)  
> **Statut** : ✅ Terminé (sauf S4-08 repoussé)  
> **Dépendances** : Sprint 3 complet (TinyOL + `oto_head.py` + exp_003 disponibles)

---

## Objectif

Finaliser la Phase 1 Python avec deux livrables principaux :

1. **Extension buffer UINT8** sur TinyOL — quantifier les activations intermédiaires pour réduire l'empreinte RAM embarquée (Gap 3)
2. **Tableau comparatif final** des 3 modèles (EWC, HDC, TinyOL) + toutes les baselines — prêt pour le manuscrit et la présentation Phase 2

---

## Planning des tâches

| ID | Tâche | Priorité | Durée est. | Dépendances | Statut |
|----|-------|:--------:|:----------:|-------------|--------|
| S4-01 | `quantization.py` (UINT8 encoder/decoder) | 🔴 | 3h | S3-05 (OtOHead) | ✅ |
| S4-02 | Extension buffer UINT8 sur TinyOL | 🔴 | 3h | S4-01 | ✅ |
| S4-03 | Exp exp_004 : UINT8 vs FP32 delta précision | 🔴 | 2h | S4-02 | ✅ |
| S4-04 | Notebook comparaison finale 3 modèles | 🔴 | 3h | exp_001, exp_002, exp_003 | ✅ |
| S4-05 | Export ONNX des 3 modèles | 🟡 | 3h | S4-03 | ✅ |
| S4-06 | Profiling mémoire systématique (3 modèles) | 🔴 | 2h | S4-03 | ✅ |
| S4-07 | Refactoring final + docstrings `src/` | 🟡 | 4h | — | ✅ exécuté Sprint 22 (01 juin 2026) |
| S4-08 | `CONTRIBUTING.md` + `LICENSE` | 🟢 | 1h | — | 🔁 Repoussé (à faire avant publication GitHub) |

**Total estimé** : 21h

---

## Critère de succès du sprint

- [x] `src/utils/quantization.py` : quantize/dequantize UINT8 validés par tests ✅
- [x] `OtOHead` avec buffer UINT8 : RAM UINT8 < RAM FP32 (mesurée par `profile_memory.py`) ✅
- [x] `experiments/exp_004_tinyol_uint8/` : métriques AA/AF/RAM FP32 vs UINT8 exportées ✅
- [x] `notebooks/04_final_comparison.ipynb` : tableau 3 modèles avec chiffres réels ✅
- [x] `scripts/export_onnx.py` : 3 fichiers `.onnx` valides (`experiments/onnx_sprint4/`) ✅
- [x] `scripts/profile_memory.py` : rapport systématique → `experiments/sprint4_memory_report.json` ✅
- [x] `ruff check src/` et `black --check src/` passent sans erreur ✅ (01 juin 2026 — 0 erreur, 427 tests)
- [x] `pytest tests/ -v` : 0 régression ✅ (427 passed, 12 skipped)

---

## Ordre recommandé d'exécution

```
S4-01 → S4-02 → S4-03 → S4-06 → S4-04
                                   ↓
                               S4-05 (si temps)
S4-07 (en parallèle, itératif)
S4-08 (dernier)
```

---

## Contexte Gap 3

Sprint 4 est le **premier point d'entrée vers Gap 3** (quantification INT8 pendant l'entraînement incrémental). L'extension buffer UINT8 (S4-01/02) est une preuve de concept orientée mémoire — pas encore un entraînement INT8 complet. La backprop en INT8 reste un problème ouvert (`TODO(dorra)`).

Les mesures de réduction RAM (FP32 → UINT8 sur les activations) doivent être reportées dans `experiments/exp_004_tinyol_uint8/results/memory_report.json` avec les clés :

```json
{
  "fp32_activations_bytes": ...,
  "uint8_activations_bytes": ...,
  "compression_ratio": ...,
  "aa_fp32": ...,
  "aa_uint8": ...,
  "delta_aa": ...
}
```

---

## Livrable sprint 4

Tableau comparatif complet 3 modèles, chiffres RAM mesurés FP32 et UINT8, export ONNX validé. Prêt pour portage MCU (Phase 2 — Sprint 10+).

---

## Questions ouvertes avant démarrage

- `TODO(dorra)` : La backprop peut-elle rester FP32 si seules les activations forward sont stockées en UINT8 ? Quel est le coût en précision attendu (delta AA cible ≤ 0.005) ?
- `TODO(arnaud)` : L'export ONNX doit-il inclure le backbone TinyOL (2 560 activations) ou uniquement la tête OtO (40 B) ?
- `FIXME(gap3)` : Documenter dans exp_004 si le delta AA est acceptable pour justifier UINT8 dans le manuscrit.
