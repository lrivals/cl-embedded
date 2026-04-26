# S10-09 — MAJ roadmap : `FIXME(gap1)` → ✅ résolu

| Champ | Valeur |
|-------|--------|
| **ID** | S10-09 |
| **Sprint** | Sprint 10 — Phase 1 Extension |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 1h |
| **Dépendances** | S10-05 (exp_050–055), S10-06, S10-07 (notebooks exécutés) |
| **Fichiers cibles** | `docs/roadmap_phase1.md` |

---

## Objectif

Clore formellement le sprint en marquant toutes les tâches S10-01 à S10-08 comme terminées dans `roadmap_phase1.md` et en résolvant tous les `FIXME(gap1)` de la Phase 1.

---

## Sous-tâches

### 1. Mettre à jour le tableau Sprint 10 dans `roadmap_phase1.md`

Changer toutes les cases `⬜` en `✅` pour les colonnes Impl., Doc., Exec. :

```markdown
| S10-01 | `pronostia_dataset.py` ... | ✅ | ✅ | N/A | ... |
| S10-02 | Configs YAML ...           | ✅ | ✅ | N/A | ... |
| S10-03 | EDA Pronostia ...          | ✅ | ✅ | ✅  | ... |
| S10-04 | Run exp_044–049 ...        | ✅ | ✅ | ✅  | ... |
| S10-05 | Run exp_050–055 ...        | ✅ | ✅ | ✅  | ... |
| S10-06 | Notebooks individuels ...  | ✅ | ✅ | ✅  | ... |
| S10-07 | Notebook comparaison ...   | ✅ | ✅ | ✅  | ... |
| S10-08 | Tests unitaires ...        | ✅ | ✅ | ✅  | ... |
| S10-09 | MAJ roadmap ...            | ✅ | ✅ | N/A | ... |
```

### 2. Résoudre les `FIXME(gap1)` dans `roadmap_phase1.md`

Localiser toutes les occurrences de `FIXME(gap1)` dans le fichier et les remplacer par une référence vers exp_050–055 :

```bash
# Rechercher toutes les occurrences
grep -n "FIXME(gap1)" docs/roadmap_phase1.md
```

Remplacement type :
```markdown
<!-- Avant -->
`FIXME(gap1)` : valider sur données industrielles réelles

<!-- Après -->
✅ Gap 1 résolu : exp_050–055 (FEMTO PRONOSTIA by_condition) — voir Sprint 10
```

### 3. Mettre à jour le livrable Sprint 10 dans `roadmap_phase1.md`

Le bloc livrable doit refléter le sprint complet :

```markdown
**Livrable sprint 10** : 12 expériences Pronostia (exp_044–055), `FIXME(gap1)` résolu ✅,
8 notebooks `cl_eval/` (6 individuels + 1 comparaison + 1 baseline),
loader `pronostia_dataset.py` validé par 9 tests unitaires.
Gap 1 comblé : premier résultat CL publié sur données industrielles réelles de roulements (FEMTO PRONOSTIA IEEE PHM 2012).
```

### 4. Vérifier les `FIXME(gap1)` dans les notebooks de Phase 1

Les notebooks suivants contiennent des cellules `FIXME(gap1)` qui doivent maintenant pointer vers exp_050–055 :

| Notebook | Section | Action |
|----------|---------|--------|
| `notebooks/cl_eval/monitoring_*/comparison.ipynb` | Section Discussion | Ajouter lien vers exp_050–055 |
| `notebooks/cl_eval/pump_*/comparison.ipynb` | Section Discussion | Ajouter lien vers exp_050–055 |
| `notebooks/03_cl_evaluation.ipynb` | Conclusion | Ajouter paragraphe Gap 1 résolu |

```bash
# Lister tous les notebooks avec FIXME(gap1)
grep -rl "FIXME(gap1)" notebooks/
```

### 5. Mettre à jour la table indicateurs de progression (`roadmap.md`)

```bash
# Vérifier si la table de progression existe
grep -n "Gap 1" docs/roadmap.md
```

La ligne Gap 1 doit passer de ❌ à ✅ :
```markdown
| Gap 1 | Validation sur données industrielles réelles | ✅ exp_050–055 PRONOSTIA |
```

---

## Critères d'acceptation

- [x] Tableau Sprint 10 dans `roadmap_phase1.md` : toutes cases Impl.+Doc.+Exec. = ✅, sprint marqué "✅ TERMINÉ — 24 avril 2026"
- [x] Zéro occurrence de `FIXME(gap1)` non résolue dans `roadmap_phase1.md`
- [x] Bloc livrable Sprint 10 reflète les 8 tâches complètes (12 expériences, 8 notebooks, loader + 9 tests)
- [x] Table Gap dans `roadmap.md` : Gap 1 = ✅ avec référence exp_050–055
- [x] `git grep "FIXME(gap1)" -- docs/` : zéro résultat non commenté dans la roadmap

---

## Commande de vérification finale

```bash
# Vérifier qu'aucun FIXME(gap1) ne subsiste dans la doc (hors commentaires historiques)
git grep "FIXME(gap1)" -- docs/ notebooks/
# Résultat attendu : zéro ligne active
```

---

**Complété le** : 2026-04-24
