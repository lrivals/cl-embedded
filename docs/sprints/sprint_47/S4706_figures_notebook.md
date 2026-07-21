# S4706 — Figures + notebook (catalogue `quant_depth`)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 47 |
| **Priorité** | 🟠 Importante — restitue le sweep en figures régénérables, honnêtes (N/A gris, 0 chiffre en dur). |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Durée estimée** | 4h |
| **Dépendances** | S4703/S4704/S4705 (JSON) · S4201 ✅ (registre `src/figures/`) |
| **Fichiers cibles** | `src/figures/catalogs/quant_depth.py`, `docs/figures/quantization_depth/`, `notebooks/cl_eval/quant_depth/comparison.ipynb` |
| **Références** | `src/figures/style.py` (`STRATEGY_COLORS`), `registry.py` (`@register_catalog`), `loaders.py` (`load_experiment`, `metric_or_na`) |

---

## Contexte

Le sweep S4703/S4704 produit des JSON ; cette tâche les rend en figures via l'infrastructure `src/figures/`
(registre S4201), avec la même discipline que les catalogues `quantization/*` : **toute valeur chargée via
`load_experiment`, 0 chiffre en dur** (garanti par scan AST du test S4707), N/A en gris, honnêteté RAM
« théorique (bit-packée) ».

## Spec

### 1. Catalogue `src/figures/catalogs/quant_depth.py`

Enregistré dans le registre S4201 (`@register_catalog("quant_depth")`), `build(out_root)` produit sous
`docs/figures/quantization_depth/` :

| Figure | Contenu | Source |
|--------|---------|--------|
| `auroc_vs_bits.png` | Courbe AUROC (ou `delta_auroc`) vs `weight_bits`, une ligne par granularité, un panneau par dataset ; « cliff » annoté | `exp_S47_depth/` |
| `heatmap_bits_granularity.png` | Heatmap `delta_auroc` (bits × granularité) par dataset ; N/A gris | `exp_S47_depth/` |
| `ram_vs_bits.png` | Ratio RAM **théorique (bit-packée)** vs `weight_bits`, axe log ; libellé d'honnêteté explicite | `exp_S47_depth/` (`ram_ratio_vs_fp32`) |
| `symmetry_gain.png` | Barres `symmetric` vs `affine` aux bits critiques (gain zero-point) | `exp_S47_symmetry/` |
| `scope_context.png` | Synthèse EWC balayé ∥ HDC/Maha/TinyOL N/A (cartouches justifiés) | `exp_S47_context/context.json` |

Labels FR. Badge d'honnêteté sur `ram_vs_bits` : « RAM théorique — gain réel sous réserve de kernel bit-packé
(Sprint 48) ». Couleurs cohérentes avec `STRATEGY_COLORS` (8 bits = `int8_v2`, 16 = `q15` comme repères).

### 2. Notebook galerie `notebooks/cl_eval/quant_depth/comparison.ipynb`

Galerie FR commentée : consomme `src/figures/`, recharge les valeurs par cellule (jamais en dur), tableau de
synthèse + reco calculés depuis les JSON. Exécuté via nbconvert (doit passer sans erreur).

## Contraintes

- **0 chiffre en dur** (garde AST S4707) : toute valeur vient de `load_experiment`/`metric_or_na`.
- **N/A honnête** : cellules manquantes/hors-axe en gris, jamais 0.
- **Idempotence** : `build(out_root)` reproductible (mêmes PNG à données identiques).
- Nbconvert du notebook sans erreur.

## Vérification

```bash
python scripts/generate_figures.py --catalog quant_depth
ls docs/figures/quantization_depth/*.png | wc -l          # 5 figures
jupyter nbconvert --to notebook --execute notebooks/cl_eval/quant_depth/comparison.ipynb --stdout > /dev/null
```

---

## Résolution (implémentée)

✅ **S4706 implémenté.** Catalogue `quant_depth` + notebook galerie, **0 chiffre en dur**
(garde AST), N/A gris, RAM étiquetée « théorique (bit-packée) ».

**Catalogue** `src/figures/catalogs/quant_depth.py` (`@register_catalog("quant_depth")`,
importé dans `catalogs/__init__.py`) → **5 PNG** sous `docs/figures/quantization_depth/`, toute
valeur via `load_experiment`/`metric_or_na` :
- `auroc_vs_bits.png` — Δ AUROC vs profondeur, 1 ligne/granularité, 1 panneau/dataset, seuil
  cliff −0,02 annoté (rend visible le décrochage Pronostia int2 per-tensor et la récupération
  per-canal) ;
- `heatmap_bits_granularity.png` — heatmap Δ AUROC (granularité × bits) par dataset, N/A gris ;
- `ram_vs_bits.png` — `ram_ratio_vs_fp32` vs bits effectifs (axe log₂), **badge d'honnêteté**
  « RAM théorique — gain réel sous réserve de kernel bit-packé (Sprint 48) » ;
- `symmetry_gain.png` — barres symmetric vs affine aux bits critiques (`exp_S47_symmetry/`) ;
- `scope_context.png` — cartouche EWC balayé + 3 cartouches N/A justifiés (`context.json`).

Couleurs cohérentes `STRATEGY_COLORS` (per-tensor = rouge `int8_ptq_legacy`, per-canal = orange
`int8_v2`, affine = violet `q15`). Ordre des profondeurs par tags + bits effectifs **entiers**
(structurels, pas des résultats) → axe log sans littéral de résultat.

**Notebook** `notebooks/cl_eval/quant_depth/comparison.ipynb` (généré par
`scripts/_build_s47_notebook.py`) : galerie FR + **tableaux rechargés par cellule** (Δ AUROC
par grille, reco « plus petit bits viable » sous seuil −0,02 avec gain RAM, gain affine, contexte
N/A). Exécuté via nbconvert **sans erreur** (0 cellule en échec).

**Garde AST** : `quant_depth.py` ajouté à `HARDCODE_GUARDED_SRCS` + `"quant_depth"` à
`QUANT_CATALOGS` dans `tests/test_figures_library.py` (8 flottants de layout du cartouche
`scope_context` ajoutés à `LAYOUT_WHITELIST`) → `test_no_hardcoded_results` couvre le nouveau
catalogue.

**Vérification** : `python scripts/generate_figures.py --catalog quant_depth` (5 PNG) ;
`jupyter nbconvert --to notebook --execute … --stdout` OK ;
`pytest tests/ -k "s47_quant_depth or figures_library"` → **28 PASS**.
