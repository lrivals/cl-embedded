# S5102 — Implémentation `system_score.py`

| Champ | Valeur |
|-------|--------|
| **Sprint** | 51 |
| **Priorité** | 🟢 Bas — calcule le score et le rang consommés par S5103/S5106. |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Durée estimée** | 4h |
| **Dépendances** | S5101 (définition) · `exp_S49_ram/` · `exp_S50_energy/` · `compute_cost.py` ✅ |
| **Fichiers cibles** | `src/evaluation/system_score.py` · `tests/test_system_score.py` |
| **Références** | S5101 · CR §3 |

## Contexte

Implémenter le score défini en S5101 comme module réutilisable, consommant les JSON S49/S50 et `compute_cost.py`.

## Spec

### 1. API

```python
def compute_system_score(
    cells: list[dict],            # une config = {ram, latency_ms, energy_uj, params, macs, metric}
    weights: dict[str, float],    # pondérations par dimension (défaut : égales)
    normalization: str = "minmax" # minmax | zscore
) -> list[dict]:                  # + score, rank par config
```

- Normalise par dimension (orientation ↑=mieux), pondère, classe.
- Champs manquants (`"à mesurer"`/`null`) → score `pending` pour la config (pas de 0 fabriqué).
- `sensitivity(cells, weight_sets)` → rangs sous plusieurs jeux de poids.

### 2. Réutilisation

- RAM ← `exp_S49_ram/summary.json` ; énergie ← `exp_S50_energy/summary.json` ; params/MACs ← `compute_cost.py` ;
  latence ← DWT board (S29/S36) ; performance ← benchmarks (S28/S36/S46).
- Ne réimplémente aucune de ces mesures ; les **agrège**.

## Format de sortie

`src/evaluation/system_score.py` + `tests/test_system_score.py`.

## Contraintes

- Score borné [0,1] par construction (normalisation) ; rang entier.
- `pending` si une dimension manque (honnêteté).
- Classement à plateforme fixée.

## Vérification

```bash
python -c "from src.evaluation.system_score import compute_system_score; print('ok')"
python -m pytest tests/test_system_score.py -q
```
