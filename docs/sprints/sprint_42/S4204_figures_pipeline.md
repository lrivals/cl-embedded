# S4204 — Figures pipeline : où la quantification s'applique dans la chaîne PC → firmware

| Champ | Valeur |
|-------|--------|
| **Sprint** | 42 |
| **Priorité** | 🟠 Haute |
| **Durée estimée** | ~4h |
| **Statut** | ✅ Implémenté (7 juillet 2026) |
| **Dépendances** | S4201 (infra) · S4202 (inventaire, pour l'exactitude des chemins) |
| **Fichier cible** | `src/figures/catalogs/quant_pipeline.py` → `docs/figures/quantization/pipeline/*.png` |

## Objectif

Des diagrammes de **flux de données** montrant, pour chaque stratégie, *où* la transformation
FP32→quantifié s'applique dans la chaîne réelle du projet :

```
données → entraînement PC → checkpoint → export_weights_c.py → header .h → firmware (RAM) → forward (FPU) → MAJ CL
```

C'est la question qui revient à chaque présentation (« mais l'INT8, il est appliqué où exactement ? ») et
qui explique les différences de résultats (QAT PC ≠ PTQ board alors que le format stocké est le même).

## Figures spécifiées

| # | Fichier | Contenu |
|---|---------|---------|
| F1 | `pipeline_fp32.png` | Chaîne de référence : tout FP32 de bout en bout, RAM et forward FPU annotés — base des 4 suivantes |
| F2 | `pipeline_int8_qat_pc.png` | QAT PC : fake-quant inséré **dans la boucle d'entraînement** ; l'export et le firmware ne sont pas concernés (c'est une évaluation PC) — encadré « pourquoi la métrique tient » |
| F3 | `pipeline_int8_ptq_board.png` | PTQ legacy : entraînement FP32 → `ewc_int8_from_fp32` **one-shot au boot firmware** (échelle fixe 1/128, accumulateur int16) → poids INT8 en RAM, déquant FPU dans la boucle — encadré « pourquoi la métrique s'effondre » + « pourquoi pas plus rapide » |
| F4 | `pipeline_int8_v2_q15.png` | Kernel v2 / Q15 : **calibration du scale côté PC** (per-tensor/per-channel sur données) → `export_weights_c.py --int8-v2` / `--maha-q15` → header généré → firmware déquant FPU — le point d'application du scale est le delta clé vs F3 |
| F5 | `pipeline_comparatif.png` | Vue unique 4 lignes (FP32 / QAT / PTQ legacy / v2+Q15) alignées sur les mêmes étapes, avec un marqueur « ⚡ quantification appliquée ici » par ligne — LA figure de synthèse pour slides |

## Règles

- Diagrammes en **matplotlib** (patches/annotate via helpers S4201) — pas de graphviz ni d'outil externe,
  pour rester régénérables en une commande.
- Les noms d'étapes reprennent les **vrais noms du dépôt** (`export_weights_c.py`, `model_weights.h`,
  `ewc_int8_from_fp32`, `pipeline.c`) — le diagramme sert aussi de carte du code.
- Exactitude technique validée contre le code (et `TODO(dorra)` pour relecture du point d'application des
  scales per-channel dans le kernel v2).
- Pas de chiffres de résultats dans ces figures (rôle de S4205) ; seules les annotations structurelles
  (RAM ÷4, ÷2) issues de la définition des formats sont permises.

## Critères d'acceptation

1. Les 5 PNG régénérés par `generate_figures.py --catalog quantization/pipeline`.
2. F5 lisible seule en slide 16:9 (test visuel à taille réelle).
3. Chaque étape nommée correspond à un fichier/fonction existant du dépôt (revue croisée S4202).

## Réalisation (7 juillet 2026)

- `src/figures/catalogs/quant_pipeline.py` (catalogue `quantization/pipeline`) : F1–F5 en matplotlib pur, chaîne canonique `STAGES` = 8 vrais artefacts du dépôt (`export_weights_c.py`, `model_weights*.h`, `ewc_int8_from_fp32`, `pipeline.c`), marqueur ◆ « quantification appliquée ici » par ligne, encadrés « pourquoi la métrique tient/s'effondre/pas plus rapide ». `TODO(dorra)` porté par F4.
- Helpers de schéma factorisés dans **`src/figures/schematic.py`** (`box`/`arrow`/`footnote`) — partagés pipeline/pédagogie, 0 duplication.
- Aucune donnée d'expérience : seules les annotations de format (RAM ÷4/÷2) apparaissent. Glyphes émojis remplacés par ◆/✓/✗ (rendus par DejaVu Sans, 0 warning glyphe).
- Enregistré via `catalogs/__init__.py` ; `generate_figures.py --catalog quantization/pipeline` produit 5 PNG sous `docs/figures/quantization/pipeline/`.
