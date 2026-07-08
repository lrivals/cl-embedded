# S4101 — Infrastructure du sprint rédaction

| Champ | Valeur |
|-------|--------|
| **Sprint** | 41 |
| **Statut** | ✅ Implémenté (3 juillet 2026) |

## Réalisé

1. **`.gitignore` racine** : ajout de `docs/rapport_de_stage/` (manuscrit Overleaf + md de travail
   non versionnés, comme `docs/prepThese/`). Vérifié :
   `git check-ignore docs/rapport_de_stage/Manuscrit_Final_Rivals/rapport.tex` → ignoré,
   `git status` ne montre plus le dossier.
2. **Arborescence `docs/rapport_de_stage/FIchier_md/`** : `00_README.md` (workflow + budgets) +
   9 squelettes `01_introduction.md` … `09_abstracts_annexes.md` (vides, marqueur « à rédiger » —
   aucun texte tant que la tâche S4105–S4108 correspondante n'a pas été demandée).
3. **`docs/sprints/sprint_41/`** : doc de sprint `S4100_sprint_41.md` + cette doc + fiches S4102 + audits S4103/S4104.

## Rappel des invariants

- Le projet Overleaf `docs/rapport_de_stage/Manuscrit_Final_Rivals/` n'est jamais modifié sans
  instruction explicite.
- Les textes du manuscrit vivent dans `FIchier_md/` (non versionnés) ; la documentation de sprint
  (fiches de cadrage, audits) est versionnée dans `docs/sprints/sprint_41/`.
