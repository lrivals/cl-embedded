# S2011 — Génération des figures + Notebook de présentation (Sprints 16–20)

| Champ | Valeur |
|-------|--------|
| **Tâche** | S2011 |
| **Sprint** | Sprint 20 |
| **Priorité** | 🟡 Important — support visuel pour présentation et manuscrit |
| **Statut** | ✅ Terminé |
| **Durée estimée** | 2h |
| **Dépendances** | S2010 (`generate_presentation_plots.py` existant) |
| **Fichiers produits** | `docs/figures/presentation_board/*.png` (12 figures), `notebooks/presentation_board_sprint16_20.ipynb` |

---

## Objectif

1. **Générer les 12 figures PNG** via le script `scripts/generate_presentation_plots.py`
2. **Créer un notebook Jupyter** `notebooks/presentation_board_sprint16_20.ipynb` qui affiche chaque figure dans l'ordre pédagogique avec commentaires, utilisable comme support de présentation interactif

---

## Commandes d'exécution

```bash
# 1. Générer les figures
python scripts/generate_presentation_plots.py
# → docs/figures/presentation_board/*.png (12 fichiers)

# 2. Ouvrir le notebook
jupyter lab notebooks/presentation_board_sprint16_20.ipynb
# ou
jupyter notebook notebooks/presentation_board_sprint16_20.ipynb

# 3. Exécuter toutes les cellules : Kernel → Restart & Run All
```

---

## Ordre pédagogique des figures dans le notebook

| # | Figure | Section | Contenu |
|---|--------|---------|---------|
| 1 | `07_sprint_timeline` | Vue d'ensemble | Gantt Sprints 16–20 avec statuts |
| 2 | `08_firmware_architecture` | Firmware | PC → UART → pipeline → modèles |
| 3 | `09_uart_protocol` | Protocole | Trames request + réponse v2/v3 |
| 4 | `01_ram_budget` | Mémoire | RAM par modèle vs budget 64 Ko |
| 5 | `06_memory_breakdown` | Mémoire | Flash vs SRAM (stacked bar) |
| 6 | `11_gap2_compliance` | Gap 2 | Tableau de conformité |
| 7 | `02_latency` | Performance | Latence vs 100 ms |
| 8 | `03_latency_log` | Performance | Latence log (board vs dry-run) |
| 9 | `12_throughput` | Performance | 34 235 ips vs capteur 1 kHz |
| 10 | `10_forgetting_intuition` | EWC | Courbes accuracy par tâche |
| 11 | `04_ewc_lambda_impact` | EWC | Forgetting + accuracy vs λ |
| 12 | `05_all_experiments_comparison` | Résultats | Toutes les expériences S19–20 |

---

## Critère de succès

- [ ] `python scripts/generate_presentation_plots.py` → 12 PNG dans `docs/figures/presentation_board/`
- [ ] Notebook exécutable sans erreur (`Kernel → Restart & Run All`)
- [ ] Chaque figure accompagnée d'un commentaire Markdown auto-suffisant
