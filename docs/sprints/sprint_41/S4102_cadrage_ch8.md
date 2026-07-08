# Fiche de cadrage — Ch. 8 Perspectives & conclusion (~2 p., cible md `08_perspectives_conclusion.md`) + abstracts/annexes (`09_abstracts_annexes.md`)

## Messages clés — Perspectives (chacune ancrée dans du travail réel, jamais spéculative)

1. **Mise à jour autonome par gate de nouveauté (S38)** : gate Mahalanobis + détecteur de drift
   décidant à bord quand updater — ~97 % des MAJ économisées, F1 préservé (pretrained Δ≤0.02),
   parité verdict board↔PC 1.000. Source : `experiments/exp_S38_summary.json`, exps
   `exp_S38_parity_*`. La contribution « système autonome » présentée comme extension naturelle.
2. **Q15 pour les grandes dynamiques (S34)** : corrélation de rang Q15 > INT8 sur 5 datasets,
   AUROC recouvrée (Pronostia −0.113 → +0.013) ; parité board exacte, 5 µs.
   Source : `experiments/exp_S34_maha_q15/`, `exp_S34_board_*`.
3. **Sélection de features (S35)** : impact 5feat/all/best sur F1 et coût (`.bss` 104 956 → 183 936 B),
   reco 5-feat par défaut. Source : `exp_S35_*`, `docs/sprints/sprint_35/S3512_analysis_update.md`.
4. **Énergie (S33 — décision : exclue du corps)** : au plus 1–2 phrases — chaîne de mesure
   prête (marqueurs GPIO, segmentation LPM01A) mais sonde non posée, aucune valeur µJ mesurée.
5. **SIMD/CMSIS-NN** pour un chemin entier INT8 (gain latence, bloqué S2908) ; QAT exporté vers
   le board ; données industrielles partenaire (Edge Spectrum).

## Messages clés — Conclusion

- Rappel de la triple lacune → ce qui est démontré, mesuré, et ce qui reste ouvert (miroir du ch. 3).
- Une phrase sur la portée : méthodologie de portage/parité réutilisable au-delà du cas d'étude.

## `09_abstracts_annexes.md`

- **Abstracts FR/EN réécrits au réalisé** (l'existant annonce du futur) — ~15 lignes chacun,
  mêmes chiffres phares que la conclusion (tous traçables).
- **Annexes** : A. tableau détaillé des 6 datasets ; B. grilles complètes 4 modèles × 5 datasets
  (PC + board, S28/S29/S35) ; C. figures supplémentaires ; D. protocole UART v3 (optionnel).
- Vérifier la consigne : annexes hors quota de pages.

## Figures

- Corps : aucune. Annexes : heatmaps 4×5 existantes (`docs/figures/gap1_heatmap_*`,
  `board_gap3_heatmaps.png`).

## Refs bib

Aucune nouvelle.

## Points ouverts

- Ordre des perspectives : mettre S38 en premier (le plus valorisant) — à confirmer à la rédaction.
- Si S40 aboutit avant la deadline, la perspective « QAT exporté/kernel v2 board » migre au ch. 7.
