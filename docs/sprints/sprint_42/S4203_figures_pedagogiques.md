# S4203 — Figures pédagogiques : ce que la quantification fait aux données

| Champ | Valeur |
|-------|--------|
| **Sprint** | 42 |
| **Priorité** | 🔴 Critique — cœur de la demande (expliquer les stratégies) |
| **Durée estimée** | ~6h |
| **Statut** | ✅ Implémenté (7 juillet 2026) |
| **Dépendances** | S4201 (infra) · checkpoints EWC/Maha existants · `src/utils/int8_c_emulation.py` (Sprint 39) |
| **Fichier cible** | `src/figures/catalogs/quant_pedagogy.py` → `docs/figures/quantization/pedagogy/*.png` |

## Objectif

Des figures **conceptuelles** qui expliquent le *mécanisme* de chaque stratégie — utilisables telles
quelles en slide devant un public non spécialiste quantification. Elles montrent la transformation des
données, pas les résultats (les résultats = S4205).

## Figures spécifiées

| # | Fichier | Contenu | Données |
|---|---------|---------|---------|
| P1 | `mapping_affine_int8.png` | Le mapping affine FP32→INT8 : axe réel continu → 256 niveaux, scale `s` et zero-point `z` annotés, valeurs clampées aux bornes visualisées | vrais poids d'un checkpoint EWC (histogramme) + grille superposée |
| P2 | `grilles_int8_vs_q15.png` | Comparaison des grilles : 256 niveaux INT8 vs 65 536 niveaux Q15 sur la même dynamique ; zoom sur une région montrant les valeurs écrasées en INT8 | tenseur `sigma_inv_` réel (Mahalanobis, grande dynamique ~6e5 Paderborn) |
| P3 | `qat_vs_ptq.png` | Deux chronologies côte à côte : QAT (fake-quant *pendant* l'entraînement, le gradient voit l'erreur, STE) vs PTQ (conversion *après*, le modèle découvre l'erreur au déploiement) — schéma annoté, pas de données | schéma pur (étiqueté « illustration ») |
| P4 | `erreur_quantification_poids.png` | Distribution de l'erreur `x − dequant(quant(x))` sur de vrais poids, pour INT8 échelle fixe 1/128 (legacy) vs INT8 scale calibré vs Q15 — montre *pourquoi* le legacy clampe | checkpoint EWC réel, quantifié par les 3 chemins via l'émulateur S39 |
| P5 | `dynamique_sigma_inv.png` | Le cas d'école « grande dynamique » : valeurs de `sigma_inv_` triées (échelle log) avec les niveaux représentables INT8 vs Q15 superposés — INT8 écrase les grandes valeurs → distances collapsées (Sprint 34) | `sigma_inv_` réel Paderborn ou Pronostia |
| P6 | `fakequant_forward.png` | Zoom sur un neurone : chemin FP32 vs chemin fake-quant (quantize→dequantize inséré) vs chemin firmware (poids INT8 stockés, déquant FPU dans la boucle) — les 3 forward annotés | schéma pur (étiqueté « illustration ») |

## Règles

- P1/P2/P4/P5 utilisent **de vrais tenseurs** du projet (checkpoints, `export_weights_c.py` déjà capable de
  les lire) — la provenance (checkpoint, dataset) figure en note de bas de figure.
- P3/P6 sont des **schémas de mécanisme** : autorisés sans données mais étiquetés comme illustrations.
- Quantification recalculée via `src/utils/int8_c_emulation.py` (bit-exact chemin C) — pas de
  réimplémentation ad hoc dans le catalogue.
- Couleurs par stratégie = palette S4201 (cohérence avec S4204/S4205).
- Labels/titres/légendes **en français** ; chaque figure doit être compréhensible seule (légende
  auto-portante — elles finiront isolées dans des slides).

## Critères d'acceptation

1. Les 6 PNG régénérés par `generate_figures.py --catalog quantization/pedagogy`.
2. Aucune donnée synthétique non étiquetée ; provenance des tenseurs affichée (P1/P2/P4/P5).
3. P4 reproduit qualitativement le constat S39 (erreur legacy ≫ calibré ≫ Q15) — sinon investiguer, ne pas
   maquiller.

## Réalisation (7 juillet 2026)

- `src/figures/catalogs/quant_pedagogy.py` → 6 PNG `docs/figures/quantization/pedagogy/` via `generate_figures.py --catalog quantization/pedagogy`.
- Tenseurs réels : P1/P4 poids EWC `exp_S39_matched/checkpoints/ewc_pronostia_5feat.pt` (fc3 absmax 1,146 → clamp legacy visible) ; P2/P5 `sigma_inv_` `exp_S35_board_5feat_mahalanobis_paderborn/checkpoints/mahalanobis_task0.pkl` (absmax ≈ 6,6e5, 11/25 coefficients écrasés à 0 en INT8). Provenance en note de bas de figure.
- Quantification via `int8_c_emulation` uniquement (`_weight_scales`/`_quant_weight`/`_sat8`/`_trunc_to_int`) ; P3/P6 = schémas matplotlib étiquetés « illustration ».
- Critère 3 ✅ : P4 reproduit legacy ≫ calibré ≫ Q15 (|erreur| moyenne 4,1e-3 / 7,9e-4 / 3,2e-6, calculée à l'exécution) + garde-fou runtime qui alerte si l'ordre n'est pas reproduit.
