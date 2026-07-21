# S4406 — Notebook comparatif, tests, recommandation MCU & clôture

| Champ | Valeur |
|-------|--------|
| **Sprint** | 44 |
| **Priorité** | 🟡 Moyenne — assemblage + **produit la reco de portage** consommée par S45. |
| **Statut** | ✅ Implémenté — notebook + `test_drift_metrics.py` (12) + `test_drift_detectors.py` (50) + reco MCU. |
| **Durée estimée** | 4h |
| **Dépendances** | S4405 ✅ (grille + figures) · S4404 ✅ (métriques) · `pytest`, `nbconvert` |
| **Fichiers cibles** | `notebooks/cl_eval/drift_detection/comparison.ipynb`, `tests/test_drift_detectors.py`, `tests/test_drift_metrics.py`, `docs/roadmap_phase2.md`, `CLAUDE.md` |
| **Références** | Pattern de clôture Sprint 38 S3809 / Sprint 42 S4207 |

---

## Contexte

Clôture du Sprint 44 : galerie comparative commentée, tests de non-régression et d'honnêteté, et surtout
la **recommandation explicite des détecteurs à porter en S45** — décidée sur les critères mesurés (délai,
FAR, état mémoire, latence, viabilité MCU), pas à l'intuition.

## Spec

### 1. Notebook — `notebooks/cl_eval/drift_detection/comparison.ipynb`

Galerie FR : par détecteur, ses métriques (S4404) + figures (S4405) + une explication prête à copier
(principe, coût, supervisé/non-supervisé, forces/limites). Sections transverses :
- **Compromis délai ↔ FAR** commenté ; **coût mémoire/latence** commenté.
- **Axe supervisé ∥ non-supervisé** : ce que chacun apporte (précision vs autonomie).
- **Tableau de synthèse** détecteur × {détection, coût, viabilité MCU}. Exécutable nbconvert.

### 2. Tests

- `tests/test_drift_detectors.py` : comportement de chaque détecteur sur séquences connues ; parité aux
  définitions de référence (littérature/`river`) dans une tolérance ; `get_state_bytes()` constant pour
  O(1) et borné pour O(W) ; interface `BaseDriftDetector` respectée ; déterminisme (seed 42).
- `tests/test_drift_metrics.py` : cas oracle/paresseux/paranoïaque (S4404) ; gestion `null` honnête.
- **0 chiffre en dur** : garde sur notebook/scripts (miroir S4207).

### 3. Recommandation MCU (livrable clé pour S45)

Section dédiée (notebook + résumé dans `docs/context/drift_detectors.md`) : **classer** les détecteurs
par portabilité board sur la base des chiffres S4405 (état borné, latence, sans-label pour l'autonomie).
Proposition attendue (à confirmer par les mesures) : **Page-Hinkley + DDM/EDDM** (O(1), ✅) et **PSI**
(O(bins), ✅ non-supervisé) comme candidats primaires ; **baseline `SlidingWindowDriftDetector`** (déjà
portée) comme référence ; **ADWIN/KSWIN** en secondaire (à valider budget) ; **KS/MMD** PC-only si trop
coûteux. Cette liste **entre** dans `S4501` (sélection de portage).

### 4. Clôture

- `docs/roadmap_phase2.md` : bloc Sprint 44 + ligne de statut.
- `CLAUDE.md` : Sprint 44 dans la ligne de statut sprint.
- `graphify_sprint_update` (skill).
- Si dernière tâche OK → proposer message de commit.

## Contraintes

- Notebook dans `notebooks/` ; aucune donnée brute committée.
- La reco MCU doit être **traçable** (chaque classement pointe vers un chiffre de `results.json`).

## Vérification

```bash
pytest tests/test_drift_detectors.py tests/test_drift_metrics.py -v
jupyter nbconvert --to notebook --execute notebooks/cl_eval/drift_detection/comparison.ipynb
```
- Tous les tests PASS ; notebook exécute sans erreur.
- La section reco MCU nomme les détecteurs retenus pour S45, chacun justifié par un chiffre mesuré.
