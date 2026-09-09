# S5106 — Figures + notebook du score système

| Champ | Valeur |
|-------|--------|
| **Sprint** | 51 |
| **Priorité** | 🟢 Bas — visualisation du classement pour rapport/soutenance. |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Durée estimée** | 3h |
| **Dépendances** | S5103 (`exp_S51_system_score/`) · registre figures `src/figures/` ✅ (S4201) |
| **Fichiers cibles** | `src/figures/catalogs/system_score.py` · `docs/figures/system_score/` · `notebooks/cl_eval/system_score/comparison.ipynb` |
| **Références** | CR §3 |

## Contexte

Produire les figures du classement composite (registre S4201) + notebook galerie, avec sensibilité aux poids.

## Spec

| Fig | Contenu | Source |
|-----|---------|--------|
| S1 | Score composite par configuration (barres, à plateforme fixée) | `ranking_{platform}.json` |
| S2 | Radar par dimension (RAM/latence/énergie/params/perf) pour top configs | dims normalisées |
| S3 | Sensibilité : rang sous ≥ 2 jeux de poids (RAM-priorité vs énergie-priorité) | `summary.json` |
| S4 | Dimensions manquantes en gris (`pending`, honnêteté) | — |

- Valeurs via `load_experiment` (0 chiffre en dur, garde AST).
- Board et PC en figures séparées (jamais fusionnés).
- Notebook nbconvert OK, tableaux rechargés par cellule.

## Format de sortie

`src/figures/catalogs/system_score.py` → PNG `docs/figures/system_score/` + notebook.

## Contraintes

- 0 chiffre en dur (garde AST `test_figures_library.py`).
- `pending` en gris, jamais 0.

## Vérification

```bash
python scripts/generate_figures.py --catalog system_score
ls docs/figures/system_score/*.png
python -m pytest tests/test_figures_library.py -q
```
