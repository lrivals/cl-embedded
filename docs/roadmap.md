# Roadmap — CL-Embedded

> Version : 2.2 | Mise à jour : 10 avril 2026  
> Horizon : Phase 1 (PC Python) = avril–mai 2026

---

## Vue macro — Phases du stage

```
Phase 0 : Revue de littérature    [16 mars → 15 avril 2026]
  └── Manuscrit préliminaire (deadline : 15 avril)

Phase 1 : Implémentation Python   [15 avril → 20 mai 2026]
  └── Ce dépôt — 3 modèles CL + 3 baselines non supervisées (dont Mahalanobis)

Phase 2 : Portage MCU             [15 mai → 15 juin 2026]
  └── STM32N6 — profiling mémoire + latence

Phase 3 : Expériences + rédaction [15 juin → 6 août 2026]
  └── Rapport final + code GitHub public
```

---

## Détail par phase

- [Phase 1 — Implémentation Python](roadmap_phase1.md) — Sprints 1–9 (dont extension notebooks granulaires), résultats expériences
- [Phase 2 — Portage MCU](roadmap_phase2.md) — Sprint 10, Backlog

---

## Triple Gap — Statut de contribution

| Gap | Critère | Statut |
|-----|---------|--------|
| **Gap 1** | Validation sur données industrielles réelles | ✅ exp_050–055 PRONOSTIA (Sprint 10, 24 avril 2026) |
| **Gap 2** | CL complet sous 100 Ko RAM avec chiffres précis | 🔄 Partiellement — Mahalanobis 80 B ✅, DBSCAN 118 Ko ❌ |
| **Gap 3** | Quantification INT8 pendant entraînement incrémental | ⬜ Non adressé (Phase 2+) |

---

## Indicateurs de progression

> Légende colonnes : **Impl.** = code livré · **Testé** = tests unitaires verts · **Exp.** = expérience exécutée · **ONNX** = export validé · **RAM** = mesurée

| Modèle | Impl. | Testé | Exp. | ONNX | RAM |
|--------|:-----:|:-----:|:----:|:----:|:---:|
| M2 EWC + MLP | ✅ | ✅ | ✅ | ⬜ | ✅ |
| M3 HDC | ✅ | ✅ | ✅ | ⬜ | ✅ |
| M1 TinyOL | ✅ | ✅ | ✅ | ⬜ | ✅ |
| M1 + buffer UINT8 | ⬜ | ⬜ | ⬜ | N/A | ⬜ |
| M4a K-Means (K dynamique) | ✅ | ✅ | ✅ | N/A | ✅ |
| M4b KNN anomaly detection | ✅ | ✅ | ✅ | N/A | ✅ |
| M5 PCA reconstruction | ✅ | ✅ | ✅ | N/A | ✅ |
| M6 Mahalanobis | ✅ | ✅ | ✅ | N/A | ✅ |
| M7 DBSCAN | ✅ | ⬜ | ✅ | N/A | ⬜ |
