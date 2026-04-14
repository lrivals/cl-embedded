# Roadmap Phase 2 — Portage MCU

> Mise à jour : 14 avril 2026  
> Horizon : 15 mai – 15 juin 2026  
> ← [Index roadmap](roadmap.md)

> **Note de numérotation** : Les Sprints 6–9 ont été alloués à la Phase 1 Extension (notebooks d'évaluation granulaires). Le premier sprint Phase 2 est donc le **Sprint 10**.

---

## Sprint 10 — Semaine 1 Phase 2 (20–27 mai 2026)

**Objectif** : Setup environnement embarqué STM32 (NUCLEO-F439ZI + VS Code)

> **Contexte** : La NUCLEO-F439ZI (Cortex-M4, 256 Ko RAM, pas de NPU) est une board de développement intermédiaire — la cible finale reste le STM32N6 (Cortex-M55, NPU). Ce sprint valide la chaîne compile → flash → debug avant d'avoir accès au hardware cible.

| ID | Tâche | Impl. | Doc | Exec | Fichier cible | Durée est. |
|----|-------|:-----:|:---:|:----:|---------------|------------|
| S10-01 | Setup toolchain ARM GCC + OpenOCD + VS Code + Cortex-Debug + projet blink | ⬜ | ✅ | ⬜ | `docs/sprints/sprint_6/S601_stm32_env_setup.md` | 3h |

**Livrable sprint 10** : LED clignote sur NUCLEO-F439ZI, breakpoint VS Code opérationnel, `launch.json` documenté et reproductible.

---

## Backlog (Phase 2 — portage MCU)

| Tâche | Priorité | Notes |
|-------|:--------:|-------|
| Validation sur FEMTO PRONOSTIA | 🔴 | Gap 1 scientifique |
| Quantification PTQ + export TFLite Micro | 🔴 | Via STM32Cube.AI |
| Profiling RAM/latence sur STM32N6 réel | 🔴 | Gap 2 — mesures précises |
| Exploration backprop INT8 (MLP minimal) | 🟡 | Gap 3 |
| Benchmark sur équipement Edge Spectrum | 🟡 | Contexte industriel Frédéric |
| HMM (Hidden Markov Model) — analyse offline Dataset 1 | 🟢 | PC-only, hors contrainte 64 Ko. Baum-Welch incompatible online learning. Utile pour RUL offline uniquement. |
