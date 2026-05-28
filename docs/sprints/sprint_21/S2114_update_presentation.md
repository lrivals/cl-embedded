# S2114 — Mise à jour présentation board sprints 16–21

| Champ | Valeur |
|-------|--------|
| **Sprint** | 21 |
| **Priorité** | 🟡 Moyenne |
| **Statut** | ✅ Terminé |
| **Durée estimée** | 2 h |
| **Dépendances** | S2113 (résultats E21 finalisés et protocole documenté) |
| **Fichiers cibles** | `docs/presentation_board_sprint16_20.md`, `notebooks/presentation_board_sprint16_20.ipynb` |

---

## Contexte

Les deux fichiers de présentation ont été créés en fin de Sprint 20 pour synthétiser le travail board des sprints 16–20. Depuis, Sprint 21 a produit 6 nouvelles expériences board documentées dans S2113 :

| Expérience | Modèle | Dataset | acc moy ± σ | AF moy ± σ | lat ms ± σ | RAM B |
|------------|--------|---------|:-----------:|:----------:|:----------:|:-----:|
| E21-01 | Mahalanobis | Monitoring | 0.107 ± 0.012 | 0.011 ± 0.008 | 0.004 ± 0.000 | 200 |
| E21-02 | TinyOL | Monitoring | 0.114 ± 0.010 | 0.000 ± 0.000 | 0.004 ± 0.000 | 5 800 |
| E21-03 | Mahalanobis | Pronostia | 0.094 ± 0.007 | 0.000 ± 0.000 | 0.004 ± 0.000 | 200 |
| E21-04 | EWC λ=400 | Pronostia | 0.886 ± 0.023 | 0.146 ± 0.025 | 0.251 ± 0.000 | 9 728 |
| E21-04b | EWC λ=0 | Pronostia | 0.852 ± 0.011 | 0.204 ± 0.017 | 0.250 ± 0.001 | 9 728 |
| E19-02b | EWC λ=400 | Monitoring | 0.896 ± 0.003 | 0.010 ± 0.012 | 0.249 ± 0.001 | 9 728 |

Ces résultats complètent la couverture cross-dataset (CWRU + Monitoring + Pronostia) et valident Gap 2 sur l'ensemble des combinaisons modèle × dataset.

---

## Objectif

Mettre à jour les deux fichiers pour qu'ils reflètent l'état réel du projet après Sprint 21 :

### `docs/presentation_board_sprint16_20.md`

1. **Titre** : renommer en `Sprints 16–21` (titre H1 + en-tête du document)
2. **Table des matières** : ajouter l'entrée 11
3. **Section 10** — "État d'avancement Sprint 20 et prochaines étapes" :
   - Mettre à jour le statut des tâches Sprint 21 désormais terminées
   - Mentionner la couverture Pronostia complétée
4. **Nouvelle section 11** — "Résultats Sprint 21 — couverture cross-dataset complète" :
   - Rappel objectif Sprint 21 (Monitoring complet + Pronostia sur board)
   - Tableau comparatif E21-01 à E21-04b + E19-02b (issue de S2113)
   - Observations clés :
     - Gap 2 validé sur tous les datasets (lat < 100 ms pour les 3 modèles)
     - EWC seul modèle avec accuracy significative sur board (>85 %)
     - Mahalanobis/TinyOL : acc ~10 % = cold start sans poids pré-chargés (voir `FIXME(gap1)` dans S2103)
     - Propriété EWC vérifiée sur Pronostia board : AF(λ=400)=0.146 < AF(λ=0)=0.204
   - Tableau comparatif cross-dataset final (3 datasets × 3 modèles)

### `notebooks/presentation_board_sprint16_20.ipynb`

Mêmes modifications que le `.md`, appliquées aux cellules Markdown correspondantes du notebook.

- Renommer le titre du notebook dans les metadata et la cellule H1
- Ajouter une cellule de résultats Sprint 21 avec le tableau comparatif
- Mettre à jour la cellule "état d'avancement"

---

## Tableau comparatif cross-dataset cible (section 11)

| Modèle | CWRU acc | Monitoring acc | Pronostia acc | lat ms | RAM B | Gap 2 |
|--------|:--------:|:--------------:|:-------------:|:------:|:-----:|:-----:|
| Mahalanobis | — | 0.107 | 0.094 | 0.004 | 200 | ✅ |
| TinyOL | — | 0.114 | — | 0.004 | 5 800 | ✅ |
| EWC λ=400 | — | 0.896 | 0.886 | ~0.250 | 9 728 | ✅ |

> CWRU : expériences sprints 19–20, données à reporter depuis `experiments/exp_S19_*/results.json` et `exp_S20_*/results.json`.

---

## Vérification

```bash
# Vérifier les mises à jour dans le markdown
grep -n "Sprint 21\|E21-0\|Pronostia\|cross-dataset" docs/presentation_board_sprint16_20.md

# Vérifier le titre mis à jour
head -1 docs/presentation_board_sprint16_20.md

# Vérifier que le notebook contient les nouvelles cellules
python -c "
import json, pathlib
nb = json.loads(pathlib.Path('notebooks/presentation_board_sprint16_20.ipynb').read_text())
sources = ' '.join(''.join(c['source']) for c in nb['cells'])
assert 'E21-04' in sources, 'E21-04 manquant dans le notebook'
assert 'Pronostia' in sources, 'Pronostia manquant dans le notebook'
print('Notebook OK')
"
```
