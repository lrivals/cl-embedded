# Skill : Génération de Sprint

> **Usage** : Demander à Claude de générer ou décomposer un sprint de travail pour ce projet.  
> **Déclencheur** : "génère un sprint pour [objectif]" / "décompose la tâche [X] en sous-tâches"

---

## Contexte à fournir obligatoirement

Avant de générer un sprint, Claude doit connaître :
1. **Sprint actuel** (numéro + semaine)
2. **Objectif principal** du sprint
3. **État actuel** des implémentations (quels fichiers existent déjà)
4. **Contraintes temporelles** (heures disponibles, deadline)
5. **Blocages éventuels** (questions Dorra, données manquantes)

---

## Format de sortie attendu

Chaque sprint généré doit suivre ce format :

```markdown
### Sprint N — [Période]
**Objectif** : [une phrase]
**Critère de succès** : [ce qui permet de dire que le sprint est terminé]

| ID | Tâche | Priorité (🔴/🟡/🟢) | Fichier cible | Durée est. | Dépendances |
|----|-------|:---:|---------------|:---:|-------------|
| SN-01 | ... | 🔴 | `src/...` | Xh | — |
| SN-02 | ... | 🟡 | `src/...` | Xh | SN-01 |

**Livrable** : [ce qui est produit concrètement]
**Questions ouvertes** : [blocages identifiés pour Arnaud/Dorra/Fred]
```

---

## Règles de décomposition

### Granularité des tâches
- Une tâche = **un fichier principal produit** ou **une expérience lancée**
- Durée : 1h à 4h maximum par tâche. Si > 4h → décomposer davantage
- Toujours inclure : au moins 1 tâche de test et 1 tâche d'expérience

### Priorités
- 🔴 **Critique** : bloque le livrable du sprint
- 🟡 **Important** : améliore la qualité mais ne bloque pas
- 🟢 **Nice-to-have** : si le temps le permet

### Contraintes obligatoires sur chaque sprint
- Toujours une tâche "Expérience + enregistrement résultats" dans `experiments/`
- Toujours une tâche "RAM profiling" si un nouveau modèle est introduit
- Jamais de tâche qui modifie directement les hyperparamètres dans le code source (→ toujours via configs/)
- Les tâches de test (`tests/`) ne sont jamais 🔴 mais jamais absentes

### Ordre recommandé
1. Données / loader (prerequisite pour tout)
2. Modèle (architecture)
3. Boucle d'entraînement
4. Évaluation + métriques
5. Expérience avec configs
6. Tests
7. Documentation

---

## Prompt type à utiliser

```
Sprint generation request :

Sprint : [N]
Semaine : [dates]
Objectif principal : [X]
Heures disponibles : [Y]h
État actuel :
  - Complété : [liste fichiers existants]
  - En cours : [fichiers partiels]
  - Bloqué : [raisons]
Priorité absolue de ce sprint : [1-2 tâches max]

Génère le sprint en suivant le format de skills/sprint_generation.md.
Inclure les dépendances inter-tâches et les questions ouvertes pour les encadrants.
```

---

## Post-sprint : bilan type

Après chaque sprint, générer un bilan avec :

```markdown
## Bilan Sprint N

| Tâche | Statut | Temps réel | Notes |
|-------|:------:|:----------:|-------|
| SN-01 | ✅ | Xh | ... |
| SN-02 | ⚠️ partiel | Xh | Raison du blocage |
| SN-03 | ❌ reporté | — | Reporté sprint N+1 |

**Résultats clés** :
- Métrique 1 : valeur
- RAM mesurée : X Ko

**Reporté au sprint N+1** : [liste des tâches non terminées]
**Questions pour encadrants** : [TODO(arnaud/dorra/fred)]
```
