# S4606 — Figures + notebook galerie des moments de quantification

| Champ | Valeur |
|-------|--------|
| **Sprint** | 46 |
| **Priorité** | 🟠 Important — rend le message 3-way visuel pour la présentation/manuscrit. |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Durée estimée** | 4h |
| **Dépendances** | S4603/S4604/S4605 ✅ (JSON produits) · `src/figures/` registre ✅ (S4201 : `style.py`, `loaders.py`, `registry.py`, garde AST 0-chiffre-en-dur) |
| **Fichiers cibles** | `src/figures/catalogs/quant_moment.py`, `docs/figures/quantization_moment/`, `notebooks/cl_eval/quant_moment/comparison.ipynb` |
| **Références** | S4205 (figures impact quantization) · S4201 (infra figures) · S4603–S4605 (données) |

---

## Contexte

Le sprint produit des JSON par moment ; cette tâche les rend lisibles via le pipeline de figures existant
(`src/figures/`, registre S4201). Elle enregistre un **nouveau catalogue** `quant_moment` qui charge les
`exp_S46_*` via `load_experiment` (jamais de chiffre en dur, garanti par la garde AST) et régénère les PNG
en une commande. Un notebook galerie commente les figures en français.

## Spec

### 1. Catalogue `src/figures/catalogs/quant_moment.py`

Enregistré via `@register_catalog` (comme `quant_impact.py`, S4205). Produit sous
`docs/figures/quantization_moment/` :

| Fig | Contenu | Source |
|-----|---------|--------|
| **M1** | Barres groupées **avant / après / les-deux** (+ fp32 référence) par (modèle × dataset) pour EWC et TinyOL | `exp_S46_ewc/`, `exp_S46_tinyol/` |
| **M2** | Heatmap **métrique = f(moment, modèle×dataset)** ; cellules HDC/Maha en **gris N/A** | tous `exp_S46_*` |
| **M3** | Effet **calibration** sur `after`/`both` (naïf → per-tensor calibré → per-canal) — lien ablation S39 | `exp_S46_ewc/` (balayage after_scheme) |
| **M4** | Contexte HDC (INT8≡FP32 structurel) + Maha (INT8 vs Q15) en encadré séparé | `exp_S46_context/` |

- Badge plateforme **PC / émulé** ; cellule board `« à mesurer »` (S4608 différé).
- `metric_or_na` : `None`/sentinel jamais rendu 0 ; N/A en gris ; annotation `na_reason`.
- Palette cohérente `STRATEGY_COLORS` (S4201) ; distinction visuelle `before` (borne haute, hachures) vs
  `both` (déploiement, plein).

### 2. Notebook galerie `notebooks/cl_eval/quant_moment/comparison.ipynb`

- Galerie FR commentée (une cellule par figure), résumés chargés depuis les JSON (0 valeur en dur).
- Section « Lecture » : QAT = borne haute, PTQ naïf s'effondre, PTQ calibré récupère, both = déploiement.
- Section « Limites » : HDC/Maha hors-axe, board différée, per-canal absent pour TinyOL.
- `jupyter nbconvert --execute` passe sans erreur.

## Format de sortie

```
docs/figures/quantization_moment/
├── M1_moments_bars.png
├── M2_moment_heatmap.png
├── M3_calibration_effect.png
└── M4_hdc_maha_context.png
notebooks/cl_eval/quant_moment/comparison.ipynb
```

## Contraintes

- **0 chiffre en dur** : validé par la garde AST de `test_figures_library.py` (S4207) — toute valeur via
  `load_experiment`.
- **Régénérable en une commande** : `python scripts/generate_figures.py --catalog quantization/moment`.
- N/A (HDC/Maha, board) jamais extrapolé — gris + légende.

## Vérification

```bash
python scripts/generate_figures.py --catalog quantization/moment
ls docs/figures/quantization_moment/M{1,2,3,4}_*.png
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/cl_eval/quant_moment/comparison.ipynb
# Garde 0-chiffre-en-dur
pytest tests/test_figures_library.py -k "no_hardcoded" -q
```

---

## Résolution (implémentée)

✅ **Implémenté**. Catalogue `src/figures/catalogs/quant_moment.py` enregistré via
`@register_catalog("quantization/moment")` (importé dans `catalogs/__init__.py`), régénérable
par `python scripts/generate_figures.py --catalog quantization/moment`. Écrit sous
`docs/figures/quantization_moment/` (le catalogue passe `OUT_SUBDIR="quantization_moment"` à
`savefig_png` tout en s'enregistrant sous le nom CLI `quantization/moment`).

**4 PNG produits** :

| Fig | Fichier | Contenu |
|-----|---------|---------|
| M1 | `M1_moments_bars.png` | Barres groupées fp32/before/after/both × (EWC, TinyOL) × (Monitoring, Pronostia) ; `before` hachuré (borne haute), `both` plein (déploiement) |
| M2 | `M2_moment_heatmap.png` | Heatmap métrique = f(moment, modèle×dataset) ; colonnes HDC/Maha en **gris N/A** |
| M3 | `M3_calibration_effect.png` | Effet calibration `after`/`both` : legacy C → per-tensor → per-canal (lien ablation S39) |
| M4 | `M4_hdc_maha_context.png` | Contexte HDC (INT8 ≡ FP32 structurel + RAM ×2.33) + Maha (INT8 casse / Q15 récupère) |

**Toute valeur chargée via `load_experiment`** (loaders `exp_S46_{ewc,tinyol,context}/`) —
**0 chiffre en dur** garanti par la garde AST : `tests/test_figures_library.py::test_no_hardcoded_results`
est désormais **paramétré** pour scanner `quant_impact.py` **et** `quant_moment.py`
(`HARDCODE_GUARDED_SRCS`), whitelist de layout étendue aux figsizes. `"quantization/moment"`
ajouté à `QUANT_CATALOGS`.

**Notebook galerie** `notebooks/cl_eval/quant_moment/comparison.ipynb` (11 cellules, FR) :
une section par figure avec résumés **chargés depuis les JSON** (0 valeur en dur), section
« Lecture » (QAT borne haute · PTQ naïf s'effondre · PTQ calibré récupère · both = déploiement)
et « Limites » (HDC/Maha hors-axe · TinyOL collapse recon-error documenté · board différée).
`jupyter nbconvert --execute` **passe sans erreur**.

### Vérification

```
$ python scripts/generate_figures.py --catalog quantization/moment      # 4 PNG
$ ls docs/figures/quantization_moment/M{1,2,3,4}_*.png                   # présents
$ jupyter nbconvert --to notebook --execute --inplace \
    notebooks/cl_eval/quant_moment/comparison.ipynb                      # OK
$ pytest tests/test_figures_library.py -k "no_hardcoded" -q             # 2 passed
```
