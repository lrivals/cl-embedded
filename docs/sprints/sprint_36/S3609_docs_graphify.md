# S3609 — Documentation & graphify

| Champ | Valeur |
|-------|--------|
| **Sprint** | 36 |
| **Priorité** | 🟢 Low — clôture du sprint (statut, roadmap, graphe). |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 1h |
| **Dépendances** | S3601–S3608 ✅ · `skills/graphify_sprint_update.md` ✅ |
| **Fichiers cibles** | `docs/sprints/sprint_36/S3600_sprint_36.md` (bilan), `docs/roadmap_phase2.md`, `docs/triple_gap.md`, `CLAUDE.md` |
| **Références** | Sprint 35 S3514 (modèle de clôture) |

---

## Contexte

Clôturer le sprint conformément à la procédure « Fin d'une implémentation » de `CLAUDE.md` :
mettre à jour les docs du sprint et le roadmap, puis évaluer/lancer la mise à jour du graphe.

## Spec

- **Bilan `S3600`** : remplir le tableau « Bilan » (statut/temps réel/notes) à mesure des runs.
- **`docs/roadmap_phase2.md`** : passer l'entrée Sprint 36 de ⬜ → 🟡/✅, compléter les résultats clés (latences inférence vs inf+MAJ, parité frozen 1.000, delta acc_final PC↔board, RAM/`.bss`).
- **`docs/triple_gap.md`** : § Gap 2 — ajouter les latences EWC mesurées (inférence ; inférence+MAJ) sur Pronostia/Monitoring (confirmer ≪ 100 ms).
- **`CLAUDE.md`** : ligne statut Sprint 36.
- **`graphify_sprint_update`** : invoquer le skill (il évalue la pertinence d'un update du graphe avant de le lancer).

**Règles** :
- Aucun chiffre inventé : ne remplir que ce qui a été réellement mesuré.

## Vérification

```bash
grep -n "Sprint 36" docs/roadmap_phase2.md
grep -n "Sprint 36" CLAUDE.md
```

## Implémentation (⬜)

- [ ] Compléter le bilan S3600.
- [ ] MAJ roadmap_phase2.md (statut + résultats clés) + ligne résumé en tête.
- [ ] MAJ triple_gap.md (§ Gap 2 latences EWC).
- [ ] MAJ CLAUDE.md (statut Sprint 36).
- [ ] Invoquer `graphify_sprint_update`.
