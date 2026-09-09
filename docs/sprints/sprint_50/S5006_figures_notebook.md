# S5006 — Figures + notebook énergie

| Champ | Valeur |
|-------|--------|
| **Sprint** | 50 |
| **Priorité** | 🟠 Important — visualisation pour rapport/soutenance. |
| **Statut** | ✅ Implémenté — catalogue `src/figures/catalogs/energy_real.py` (5 PNG `docs/figures/energy_real/` : E1–E3 énergie gris « à mesurer », **E4 breakdown latence INT8 mesuré board**, E5 coût/bénéfice RAM×4 / latence×1.5 / énergie à mesurer) ; 0 chiffre en dur (garde AST). Notebook §8 « mesures réelles S50 » (nbconvert OK). |
| **Durée estimée** | 4h |
| **Dépendances** | S5002/S5003/S5004 (JSON) · registre figures `src/figures/` ✅ (S4201) · `notebooks/cl_eval/energy_cost/` ✅ (S33) |
| **Fichiers cibles** | `src/figures/catalogs/energy_real.py` · `docs/figures/energy_real/` · MAJ `notebooks/cl_eval/energy_cost/comparison.ipynb` |
| **Références** | CR §4–5 |

## Contexte

Produire les figures énergie réelles (registre S4201) et mettre à jour le notebook énergie du Sprint 33, avec
badges honnêtes **mesuré / à-mesurer**.

## Spec

### Figures du catalogue `energy_real`

| Fig | Contenu | Source |
|-----|---------|--------|
| E1 | Énergie µJ/inférence par modèle × encodage | `exp_S50_energy/summary.json` |
| E2 | Décomposition par composant (MCU/périph/capteur) | `by_component` |
| E3 | Autonomie (h) par configuration | `autonomy.json` |
| E4 | Breakdown latence INT8 (déquant/MAC/requant) vs FP32 | `exp_S50_int8_latency/` |
| E5 | Coût/bénéfice : RAM ÷4 vs Δaccuracy vs Δlatence/énergie (radar/barres) | S49 + S50 |

- Valeurs via `load_experiment`/`metric_or_na` (0 chiffre en dur, garde AST).
- Badge **« à mesurer »** en gris tant qu'une cellule n'est pas capturée (honnêteté S33).

### Notebook

MAJ `notebooks/cl_eval/energy_cost/comparison.ipynb` : section « mesures réelles » consommant les JSON S50,
nbconvert OK.

## Format de sortie

`src/figures/catalogs/energy_real.py` → PNG `docs/figures/energy_real/` + notebook mis à jour.

## Contraintes

- 0 chiffre en dur (garde AST `test_figures_library.py`).
- « à mesurer » jamais remplacé par 0.

## Vérification

```bash
python scripts/generate_figures.py --catalog energy_real
ls docs/figures/energy_real/*.png
jupyter nbconvert --execute --to notebook notebooks/cl_eval/energy_cost/comparison.ipynb --stdout >/dev/null
python -m pytest tests/test_figures_library.py -q
```
