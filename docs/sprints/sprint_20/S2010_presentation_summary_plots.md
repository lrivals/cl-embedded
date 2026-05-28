# S2010 — Résumé de présentation + Figures visuelles (Sprints 16–20)

| Champ | Valeur |
|-------|--------|
| **Tâche** | S2010 |
| **Sprint** | Sprint 20 |
| **Priorité** | 🟡 Important — support pour présentation et manuscrit |
| **Statut** | ✅ Terminé |
| **Durée estimée** | 2h |
| **Dépendances** | S2005 (expériences EWC finales), S2006 (profiling RAM) |
| **Fichiers produits** | `docs/presentation_board_sprint16_20.md` (déjà créé), `scripts/generate_presentation_plots.py`, `docs/figures/presentation_board/*.png` |

---

## Objectif

Produire un **document de référence exhaustif** et un **jeu de figures** couvrant l'intégralité du travail embarqué réalisé sur la NUCLEO-F439ZI (Sprints 16–20), exploitables pour :
- Une présentation orale / démo de la carte
- Le chapitre 4 (résultats hardware) du manuscrit
- Un support visuel lors des réunions avec Arnaud, Dorra, Fred

---

## Résumé du document `presentation_board_sprint16_20.md`

> 10 sections, ~800 lignes, tous les chiffres tirés des expériences réelles.

### Ce qui a été fait (Sprints 16–20)

**Sprint 16 — Toolchain**  
Mise en place complète : arm-none-eabi-gcc, OpenOCD, CMake, Unity, CI GitHub Actions. Premier portage C : Mahalanobis (128 B RAM, 3 µs latence). IDCODE = 0x20036413, SYSCLK = 180 MHz mesuré.

**Sprint 17 — HAL + Périphériques**  
GPIO LED (PA5), UART printf retargeté, TIM3 PWM. Renode v1.16.1 opérationnel pour CI sans hardware. 24/24 Unity tests PASS.

**Sprint 18 — Pipeline UART v2 + Profiling**  
Protocole binaire 32 B request / 14 B response, CRC8. Scripts Python `sensor_stream.py`, `board_experiment_recorder.py`. Expérience E18-01 sur carte : **3.7 µs latence, 1 000 B RAM, 34 235 ips, Gap 2 ✅**.

**Sprint 19 — 3 modèles CL en C**  
Mahalanobis ✅, EWC Head MLP ✅ (poids + Fisher + consolidation), TinyOL forward pass ✅. Protocole v3 21 B avec métriques CL temps réel (acc, auroc, forgetting). Expériences E19-01 (Mahalanobis, acc=0.63 ✅) et E19-02 (EWC, bug réinit → acc=8% ⚠️ corrigé en S20).

**Sprint 20 — Finalisation**  
Fix bug EWC, poids TinyOL exportés, Gap 2 formel (3 modèles simultanés < 64 Ko), comparaison PC vs board (delta ≤ 1e-4).

### Résultats clés (chiffres à retenir)

| Métrique | Valeur | Contexte |
|---------|--------|---------|
| Latence inférence | **3–4 µs** | @ 180 MHz, Cortex-M4 + FPU |
| RAM 3 modèles simultanés | **~11 Ko** | Sur budget 64 Ko (Gap 2) |
| Throughput | **34 235 ips** | vs capteur industriel typique 1 kHz |
| Forgetting EWC λ=0 | **0.308** | Catastrophic forgetting |
| Forgetting EWC λ=400 | **0.009** | Réduction ×34 avec EWC |
| Unity tests | **28/28 PASS** | Host x86 + board |
| Gap 2 compliant | **✅ True** | RAM + latence |

---

## Figures générées

Le script `scripts/generate_presentation_plots.py` produit 12 figures PNG dans `docs/figures/presentation_board/` :

| Fichier | Contenu | Section du doc |
|---------|---------|---------------|
| `01_ram_budget.png` | RAM par modèle vs budget 64 Ko | §1 Hardware |
| `02_latency.png` | Latence inférence vs budget 100 ms | §6 Profiling |
| `03_latency_log.png` | Latence en échelle log (board vs dry-run) | §6 Profiling |
| `04_ewc_lambda_impact.png` | Forgetting + accuracy vs λ (0 / 100 / 400) | §5 Modèles / §8 Résultats |
| `05_all_experiments_comparison.png` | Toutes les expériences S19–20 comparées | §8 Résultats |
| `06_memory_breakdown.png` | Flash vs SRAM par modèle (stacked bar) | §5 Modèles |
| `07_sprint_timeline.png` | Gantt Sprints 16–20 avec statuts | §2 Toolchain |
| `08_firmware_architecture.png` | Schéma firmware (PC → UART → pipeline → modèles) | §3 Firmware |
| `09_uart_protocol.png` | Trames request + réponse v2 + v3 (byte par byte) | §4 Protocole |
| `10_forgetting_intuition.png` | Courbes accuracy par tâche : sans EWC vs avec EWC | §5 Modèles |
| `11_gap2_compliance.png` | Tableau de conformité Gap 2 (tous modèles) | §6 Profiling |
| `12_throughput.png` | Throughput ips vs fréquence capteur industriel | §8 Résultats |

### Commande d'exécution

```bash
# Générer toutes les figures (sauvegarde dans docs/figures/presentation_board/)
python scripts/generate_presentation_plots.py

# Afficher sans sauvegarder (exploration rapide)
python scripts/generate_presentation_plots.py --show --no-save

# Répertoire de sortie alternatif
python scripts/generate_presentation_plots.py --output experiments/figures/
```

---

## Critère de succès

- [ ] `python scripts/generate_presentation_plots.py` → 12 fichiers PNG générés sans erreur
- [ ] `docs/presentation_board_sprint16_20.md` : document complet, tous les chiffres cohérents avec `experiments/comparison_sprint19_20.json`
- [ ] Figures lisibles et auto-suffisantes (titre + axes + légendes)

---

## Questions ouvertes

- `TODO(arnaud)` : Quelles figures inclure en priorité dans le chapitre 4 du manuscrit ?
- `TODO(arnaud)` : La figure `10_forgetting_intuition.png` est-elle assez pédagogique pour une audience non-ML ?
