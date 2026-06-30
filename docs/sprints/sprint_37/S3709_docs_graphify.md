# S3709 — Clôture : roadmap, statut, graphify

| Champ | Valeur |
|-------|--------|
| **Sprint** | 37 |
| **Priorité** | 🟢 Clôture |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 0.5h |
| **Fichiers cibles** | `docs/roadmap_phase2.md`, `CLAUDE.md` (ligne statut), `docs/sprints/sprint_37/` |
| **Dépendances** | S3701–S3708 ✅ |

## Spec

1. `docs/roadmap_phase2.md` : ligne récap Sprint 37 dans la liste + section détaillée.
2. `CLAUDE.md` : ajout du bloc « Sprint 37 ✅ implémenté … » en fin de ligne « Statut sprint ».
3. Invoquer le skill `graphify_sprint_update` (évalue la pertinence d'un update du graphe — un
   nouveau pipeline + scripts justifie a priori un `--update`).

## Note de cohérence

`CLAUDE.md` et `graphify-out/` restent inchangés dans le **dépôt de travail** (ils y sont utiles).
Ils sont retirés **uniquement** dans la version GitLab, par la transformation S3703 — ce sprint ne
nettoie pas le dépôt de travail, il met en place l'étape qui produit la version propre.
