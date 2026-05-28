# Skill : Graphify Sprint Update

> **Usage** : Évaluer si un update du graphe de connaissance est pertinent après la fin d'une tâche de sprint, puis le lancer uniquement si nécessaire.
> **Déclencheur** : Invoqué automatiquement à la fin de chaque implémentation de tâche (voir CLAUDE.md "Fin d'une implementation")

---

## Pourquoi ce skill existe

`/graphify . --update` est coûteux en tokens et en temps. Il ne vaut la peine d'être lancé que si la tâche a introduit de nouveaux nœuds ou de nouvelles arêtes dans le graphe de connaissance. Ce skill tranche cette décision de façon systématique.

---

## Étape 1 — Évaluation (READ-ONLY, sans coût graphify)

Inspecter les fichiers touchés lors de la tâche (créés ou modifiés). Appliquer les critères ci-dessous.

### Critères qui JUSTIFIENT un update (un seul suffit)

| Critère | Exemples concrets |
|---------|------------------|
| **Nouveau fichier** ajouté | `.c`, `.h`, `.py`, `.yaml`, `.md` non existant avant |
| **Nouveau module / composant** | Nouvelle classe Python, nouveau driver C, nouveau script |
| **Nouvelle dépendance inter-fichiers** | Un fichier importe ou `#include` un autre pour la première fois |
| **Nouvelle spec ou config** | Nouveau fichier dans `docs/models/`, `docs/sprints/`, `configs/` |
| **Refactor structurel** | Renommage de module, déplacement de fichier, changement d'API publique |

### Critères qui NE justifient PAS un update (tâche trop locale)

| Critère | Exemples concrets |
|---------|------------------|
| **Bug fix dans un fichier existant** | Correction de valeur, fix logique, typo dans `.c` / `.py` |
| **Mise à jour de métriques / résultats** | Ajout de chiffres dans un `.md` de sprint déjà graphifié |
| **Nouveau test unitaire** | Test pour une fonction déjà présente dans le graphe |
| **Changement dans un seul fichier** | Sans nouvelles connexions vers d'autres fichiers |
| **Commentaires / formatting** | Reformatage, ajout de commentaires, `black` / `ruff` pass |

### Comment inspecter

```bash
# Fichiers modifiés depuis le dernier commit (ou depuis le début de la tâche)
git status --short

# Fichiers réellement nouveaux (untracked ou added)
git status --short | grep "^??" 
git status --short | grep "^A "
```

---

## Étape 2 — Verdict

Produire un verdict explicite en une ligne :

**Si NON NÉCESSAIRE :**
```
[graphify_sprint_update] NON NÉCESSAIRE — <raison en 1 ligne>
Aucun update du graphe lancé.
```
→ S'arrêter ici.

**Si NÉCESSAIRE :**
```
[graphify_sprint_update] NÉCESSAIRE — <fichier(s) déclencheur(s)>
Lancement de /graphify . --update...
```
→ Continuer à l'Étape 3.

---

## Étape 3 — Lancement de l'update (seulement si NÉCESSAIRE)

Invoquer le skill graphify avec le flag `--update` :

```
/graphify . --update
```

Scope standard du projet : `firmware/` + `docs/` + `scripts/` + `configs/`

Après exécution, confirmer :
```
[graphify_sprint_update] Graphe mis à jour. Nouveaux nœuds / arêtes reflétés dans graphify-out/.
```

---

## Règles

- Ne jamais lancer l'update sans avoir d'abord produit le verdict à l'Étape 2
- Le verdict doit être visible dans la réponse (pas silencieux)
- En cas de doute sur le critère, trancher vers **NON NÉCESSAIRE** — mieux vaut un graphe légèrement en retard qu'un update inutile
- Un update complet (`/graphify .` sans `--update`) ne doit jamais être lancé par ce skill — uniquement `--update`
