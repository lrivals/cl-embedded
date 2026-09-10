# Présentation encadrants — Sprints 44 → 53 (juillet → septembre 2026)

**Projet CL-Embedded** — Léonard Rivals (ISAE-SUPAERO / ENAC / Edge Spectrum)
Carte de travail : **NUCLEO-F439ZI** (Cortex-M4 @ 180 MHz, 256 Ko SRAM, pas de NPU).

**Public** : Arnaud Dion (ISAE-SUPAERO), Dorra Ben Khalifa (quantification / matériel),
Frédéric Zbierski (Edge Spectrum).
**Durée visée** : 35–40 min + questions, plus une **annexe énergie optionnelle** de
~12 min. **Format** : 28 slides en 6 blocs + 10 slides d'annexe.

## Ordre de lecture

| Fichier | Contenu |
|---|---|
| [`01_slides.md`](01_slides.md) | Le support : une section `#` = une slide, figures en ligne |
| [`02_script_oral.md`](02_script_oral.md) | Notes orateur slide par slide, chiffres à connaître, questions anticipées |

## Règle de traçabilité

Aucune valeur de ce support n'est saisie à la main. Les 14 figures propres à cette
présentation sont générées par le catalogue `src/figures/catalogs/seminaire_s44_s53.py`,
qui lit `experiments/` via `src.figures.loaders.load_experiment` :

```bash
python scripts/generate_figures.py --catalog seminaire_s44_s53 --style slide
python scripts/generate_figures.py --catalog energy_pedagogy --style slide
```

Une cellule non mesurée est tracée **en gris avec sa mention** — le sentinel littéral
« à mesurer » du projet, ou sa raison (`na_reason`) — et **jamais comme un zéro**
(garde AST « 0 chiffre en dur » : `tests/test_figures_library.py::test_no_hardcoded_results`).

## Traçabilité — slide → figure → source JSON

| Slide | Figure | Source de données |
|---|---|---|
| 1 | `seminaire_s44_s53/t1_timeline_sprints.png` | schéma (aucune donnée) |
| 2 | `seminaire_s44_s53/t2_carte_axes_gaps.png` | schéma (aucune donnée) |
| 4 | `drift_datasets/comparatif_datasets.png` | `exp_S43_drift_char/*/characterization.json` |
| 5 | `drift_detection_pc/f1_heatmap.png` · `cost_bars.png` | `exp_S44_PC_*/results.json` |
| 6 | `seminaire_s44_s53/d1_drift_proxy_vs_board.png` | `exp_S45_summary.json` |
| 7 | `seminaire_s44_s53/d2_drift_portabilite.png` | `exp_S45_summary.json` |
| 9 | `seminaire_s44_s53/q1_carte_quantification.png` | schéma (aucune donnée) |
| 10 | `seminaire_s44_s53/q3_moment_bilan.png` | `exp_S46_board/*.json`, `exp_S40_board_v2/` |
| 11 | `quantization_moment/M3_calibration_effect.png` | `exp_S46_ewc/*.json` |
| 12 | `quantization_depth/auroc_vs_bits.png` · `symmetry_gain.png` | `exp_S47_depth/`, `exp_S47_symmetry/` |
| 13 | `seminaire_s44_s53/q2_profondeur_pc_board.png` | `exp_S48_summary.json` |
| 14 | `quant_depth_board/latency_vs_bits.png` | `exp_S48_summary.json` |
| 16 | `seminaire_s44_s53/r1_ram_totale_recap.png` | `exp_S49_ram/summary.json` |
| 17 | `ram_full/historique_pic_pile.png` · `ratio_int8_fp32.png` | `exp_S49_ram/summary.json` |
| 19 | `seminaire_s44_s53/e1_latence_int8_breakdown.png` | `exp_S50_int8_latency/ewc.json` |
| 20 | `energy_real/e6_courant_moyen_mesure.png` | `exp_S50_energy/*.json` |
| 22 | `seminaire_s44_s53/e2_s53_wfi_repos.png` | `exp_S53_wfi/idle_reference.json` |
| 23 | `seminaire_s44_s53/e3_s53_uj_par_inference.png` | `exp_S53_wfi/batch_sweep.json` |
| 24 | `seminaire_s44_s53/e4_s53_gap3_energie.png` | `exp_S53_rate_sweep/summary.json` |
| 25 | `seminaire_s44_s53/e5_s53_sysclk.png` | `exp_S53_freq_sweep/summary.json` |
| 27 | `seminaire_s44_s53/e6_s53_statut_mesures.png` | `exp_S53_{wfi,rate_sweep,phase_profile,build_isolation}/`, `exp_S50_energy/` |

Slides sans figure (texte et tableau uniquement) : 3 (plan), 8, 15, 18, 21, 26, 28.

### Annexe A — pédagogie de la mesure d'énergie

Dix slides optionnelles, à dérouler si la question « d'où viennent ces µJ ? » est posée.
Toutes les figures viennent du catalogue `energy_pedagogy`
(`src/figures/catalogs/energy_pedagogy.py`).

| Slide | Figure | Source de données |
|---|---|---|
| A1 | `energy_pedagogy/a1_chaine_de_mesure.png` | schéma (aucune donnée) — câblage : `docs/context/lpm01a_setup.md` |
| A2 | `energy_pedagogy/a2_statique_vs_dynamique.png` | `exp_S53_freq_sweep/summary.json` (`acqmode_dyn_by_mhz`) |
| A3 | `energy_pedagogy/b1_probleme_isoler.png` | `exp_S53_rate_sweep/ewc_fp32.json` |
| A4 | `energy_pedagogy/b2_trois_estimateurs.png` | `exp_S53_wfi/delta_recovery.json`, `exp_S53_rate_sweep/summary.json`, `exp_S53_wfi/batch_sweep.json` |
| A5 | `energy_pedagogy/c1_methode_delta.png` | `exp_S53_wfi/delta_recovery.json` |
| A6 | `energy_pedagogy/c2_methode_regression.png` | `exp_S53_rate_sweep/ewc_fp32.json` |
| A7 | `energy_pedagogy/c3_methode_lot.png` | `exp_S53_wfi/batch_sweep.json` |
| A8 | `energy_pedagogy/d1_pieges_du_banc.png` | `exp_S53_wfi/idle_reference.json`, `exp_S53_counterbalance/counterbalance.json`, `exp_S53_rate_sweep/ewc_fp32.json`, `exp_S53_wfi/batch_sweep.json` |
| A9 | `energy_pedagogy/d2_repos_nest_pas_repos.png` | `exp_S53_counterbalance/idle_states_power_cycle.json`, `exp_S53_wfi/idle_reference.json` |
| A10 | `energy_pedagogy/e1_de_la_mesure_a_lautonomie.png` | `exp_S53_wfi/autonomy_delta.json` |

**Règle propre à cette annexe** : les trois estimateurs de µJ par inférence (delta,
régression de cadence, régression par lot) sont affichés **côte à côte et jamais
moyennés** — leur écart mesure le coût de la trame UART et sert de contrôle de validité
à la campagne.

## Sources rédactionnelles

- `docs/roadmap_phase2.md` (§ Sprint 44 → Sprint 53) — récit sprint par sprint
- `docs/triple_gap.md` — positionnement des résultats sur les trois gaps
- `docs/context/{drift_detectors,quantization_depth,int8_cost_benefit,ram_report}.md`
- `docs/sprints/sprint_{44..53}/` — spécifications et comptes rendus de tâche
