# Fiche de cadrage — Ch. 6 Gap 2 : RAM & latence mesurées + parité PC↔board (~4.5 p., cible md `06_gap2_ram_latence.md`)

## Messages clés

1. **Narratif RAM cohérent (point de vigilance n°2 du sprint)** : trois niveaux de claim à
   distinguer explicitement —
   - noyau minimal Sprint 20 : `.bss` ≈ 1 000 B (démonstration « <100 Ko possible ») ;
   - build multi-modèle par défaut : `.bss` = 105 036 B (41 % de 256 Ko) `[à confirmer — build S39/S40]` ;
   - pire cas mesuré (condition `all`, S35) : 183 936 B (70,2 %) — toujours dans le budget.
   Le claim Gap 2 = *tout le système multi-modèle tient dans 256 Ko, mesuré à l'octet près,
   méthodologie vérifiable* (protocole ch. 4).
2. **Latences mesurées DWT** : toutes ≪ 100 ms — inférence de 5 µs (Maha) à ~2 ms (HDC pire cas) ;
   séparation inférence vs inférence+MAJ CL (EWC : 48–65 µs gelé vs 239–340 µs online, S36).
3. **S36 — comparaison appariée PC↔board (étude avancée retenue)** :
   - gelé : **parité exacte 1.000** sur 4 cellules (7534–7672 échantillons) ;
   - online : parité approchée 0.963–0.989 (float32 board vs float64 PC), mismatches concentrés
     aux frontières de décision ; Δacc_final PC↔board ≤ 0.007.
   C'est la **validation scientifique du portage** : le board fait bien ce que le PC fait.

## Sources de chiffres (chemins vérifiés)

| Donnée | Source |
|---|---|
| `.bss` par build + contiguïté | `docs/context/ram_measurement.md` (tableau 4 sources) `[à confirmer]` |
| RAM/latences S36 frozen/online | `experiments/exp_S36_board_{frozen,online}_{5feat,all}_ewc_{monitoring,pronostia}/` |
| Parité par échantillon | `experiments/exp_S36_parity_{5feat,all}_{frozen,online}_{monitoring,pronostia}.json` (8 fichiers) |
| Synthèse indexée | `experiments/exp_S36_summary.json` |
| Latences autres modèles (Maha 5–6 µs, HDC ~585–2095 µs, TinyOL ~5–71 µs) | `experiments/exp_S29_board_int8/`, `exp_S32_board_sweep_summary.json`, exps S35 board |
| Pile (watermark) | `scripts/measure_stack_watermark.py` + mesures S39 (`experiments/exp_S39_ram/`) `[à confirmer]` |

## Figures prévues (S4109)

- 1 figure carte mémoire / décomposition `.bss` par modèle (base `docs/figures/ram_measurement/`
  `fig_monitoring_stacked.png` ou `fig_model_maps.png`).
- 1 figure latence inférence vs inférence+MAJ (base `docs/figures/sprint36_pc_board_ewc/board_model_ram_inf_vs_update_*.png`).
- 1 figure/table parité PC↔board (exacte vs approchée).

## Refs bib

`Kwon2023` (comparaison 212 Ko LifeLearner), `Ren2021` (overhead online), `Lin2024`.

## Glossaire touché

DWT, `.bss`, watermark, parité, P50/P99 (à créer S4104).

## Points ouverts

- **Dépendance S39/S40** : les mesures RAM en cours (`experiments/exp_S39_ram/`,
  `docs/figures/ram_measurement/` en évolution) peuvent affiner les chiffres → placeholders,
  résolution S4110.
- Confirmer le chiffre « 1 000 B Sprint 20 » et sa formulation exacte (noyau EWC seul) avant usage.
