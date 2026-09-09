# S5103 — Classement des configurations selon le score global

| Champ | Valeur |
|-------|--------|
| **Sprint** | 51 |
| **Priorité** | 🟢 Bas — livrable exploitable du score (CR §3 : « classer les configurations »). |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Durée estimée** | 2h |
| **Dépendances** | S5102 (`system_score.py`) · sorties S49/S50 |
| **Fichiers cibles** | `experiments/exp_S51_system_score/` |
| **Références** | CR §3 |

## Contexte

Appliquer `system_score.py` à la grille (modèle × carte × encodage) et produire un classement, à plateforme fixée.

## Spec

- Grille : {EWC, HDC, TinyOL, Mahalanobis} × {fp32, int8} × plateforme (board ; PC séparé).
- Sortie : `[platform]` → liste ordonnée `{model, encoding, score, rank, dims_normalisées}`.
- Deux jeux de poids au moins (RAM-priorité, énergie-priorité) → rangs comparés (sensibilité).
- Cellules à dimension manquante → `pending` (jamais exclues silencieusement ni mises à 0).

## Format de sortie

`experiments/exp_S51_system_score/ranking_{platform}.json` + `summary.json` (rangs sous chaque jeu de poids).

## Contraintes

- Lecture seule sur S49/S50 ; aucun recalcul de mesure.
- Board et PC classés séparément (CR §1).
- Aucun score écrit tant que les dimensions ne sont pas disponibles.

## Vérification

```bash
python -c "import json;d=json.load(open('experiments/exp_S51_system_score/ranking_board.json'));\
assert all('rank' in c for c in d)"
grep -l pending experiments/exp_S51_system_score/*.json   # dimensions manquantes honnêtes
```
