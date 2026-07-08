# S4205 — Figures d'impact mesuré : effet des stratégies sur le fonctionnement du modèle

| Champ | Valeur |
|-------|--------|
| **Sprint** | 42 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | ~4h |
| **Statut** | ✅ Implémenté (7 juillet 2026) |
| **Dépendances** | S4201 (loaders) · JSON existants `exp_S28/S29/S34/S36/S39` (lecture seule) |
| **Fichier cible** | `src/figures/catalogs/quant_impact.py` → `docs/figures/quantization/impact/*.png` |

## Objectif

Les figures **résultats** : ce que chaque stratégie a réellement donné, chiffres chargés depuis les JSON
d'expériences existants (aucune relance de campagne — le sprint est lecture seule côté expériences).
Elles complètent S4203 (mécanisme) et S4204 (où) par le « et alors ? ».

## Figures spécifiées

| # | Fichier | Contenu | Source |
|---|---------|---------|--------|
| I1 | `metrique_par_strategie.png` | LA figure de synthèse : Δmétrique vs FP32 par stratégie (QAT PC / PTQ legacy board / v2 per-tensor / per-channel / Q15), barres groupées par dataset — visualise « quantifier ≠ quantifier » | `exp_S28_PC_*`, `exp_S36_board_*_int8_*`, `exp_S39_ablation/` |
| I2 | `ablation_perte_f1.png` | Escalier d'ablation Sprint 39 : F1 de legacy_c → +int32 → +scale calibré → per-channel → Q15, Pronostia + Monitoring — attribue la perte à chaque facteur | `exp_S39_ablation/` |
| I3 | `recuperation_q15_maha.png` | Cas Mahalanobis : corrélation de rang au FP32 et ΔAUROC, INT8 vs Q15, 5 datasets — la grande dynamique récupérée | `exp_S34_maha_q15/` |
| I4 | `ram_gap3.png` | Ratio RAM par stratégie et modèle (×2.33 HDC int16 → ×4 INT8), avec le budget 256 Ko en référence — réutilise les heatmaps/données Gap 3 existantes | `exp_S28_PC_*`, `exp_S29_board_int8/` |
| I5 | `paradoxe_latence.png` | Latence board FP32 vs INT8 (même modèle, même dataset) : l'INT8 n'accélère pas (déquant FPU), le gain est RAM — barres + annotation du mécanisme | `exp_S29_board_int8/`, `exp_S36_board_*` |
| I6 | `qat_vs_ptq_resultats.png` | Le contraste central du projet : même format INT8, QAT PC Δ≤0.006 ✅ vs PTQ board legacy F1 0.07–0.15 ❌ vs v2 calibré (émulé PC, board « à mesurer ») — pendant chiffré de la figure P3 | `exp_S28_PC_*`, `exp_S36_*`, `exp_S39_quant_sweep/`, `exp_S40_board_v2/` si présent |

## Règles d'honnêteté (strictes)

- **Toute valeur provient d'un `load_experiment`** — un test S4207 vérifie l'absence de littéraux numériques
  de résultat dans `quant_impact.py`.
- Chaque figure porte un **badge de plateforme** par série : `mesuré board` / `émulé PC (bit-exact)` /
  `PC natif` — jamais mélangés sans distinction (règle Sprint 40).
- Cellules absentes (board v2 non flashé, N/A mono-classe) : affichées **« à mesurer »** ou grisées avec
  `na_reason` — pas d'extrapolation, pas d'omission silencieuse.
- Les métriques gardent leur nom exact (F1, AUROC, corrélation de rang) — pas d'« accuracy » fourre-tout
  (leçon Sprint 35 : accuracy trompeuse).

## Critères d'acceptation

1. Les 6 PNG régénérés par `generate_figures.py --catalog quantization/impact`.
2. Chiffres identiques à ceux des notebooks sources (vérification croisée sur ≥3 valeurs par figure).
3. I6 affiche « à mesurer » pour le board v2 tant que `exp_S40_board_v2/` est absent, et se remplit
   automatiquement quand il apparaît (relance de la commande suffit).

## Réalisation (7 juillet 2026)

- `src/figures/catalogs/quant_impact.py` (catalogue `quantization/impact`) : I1–I6, **toute valeur chargée via `load_experiment` + `metric_or_na`** — 0 littéral de résultat (garde-fou AST `test_no_hardcoded_results`, S4207). Loaders dédiés : `_qat_delta` (S28), `_ablation` (S39), `_ptq_board_delta` (S36 summary), `_maha_variants` (S34), `_ram_ratio` (S28), `_board_latency` (S29), `_board_v2_f1` (S40), `_quant_sweep_metric` (S39).
- **Δmétrique calculée** (int8 − fp32) sur place pour homogénéiser le signe entre sources (les conventions de `delta_metric` diffèrent S28 vs S36). Badges de plateforme par série (`PC natif` / `mesuré board` / `émulé PC (bit-exact)`).
- I6 nourri par `exp_S40_board_v2/` désormais présent : **v2 board réel Pronostia** (frozen F1 ≈ 0.90, chargé), **Monitoring → « à mesurer »** (cellule non produite) — se remplira à la prochaine relance après flash.
- I3 conserve la nuance honnête CWRU (AUROC FP32 sub-aléatoire → Q15 non pertinent), I5 le paradoxe latence (INT8 ×1.84 vs FP32 sur FPU). 6 PNG sous `docs/figures/quantization/impact/`, 0 warning.
