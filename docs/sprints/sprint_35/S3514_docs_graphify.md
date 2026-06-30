# S3514 — Documentation + mise à jour du graphe de connaissance

| Champ | Valeur |
|-------|--------|
| **Sprint** | 35 |
| **Priorité** | 🟢 Nice-to-have |
| **Statut** | ⬜ À démarrer |
| **Durée estimée** | 1h |
| **Dépendances** | toutes les tâches S3501–S3513 |
| **Fichiers cibles** | `CLAUDE.md`, `docs/roadmap_phase2.md`, `README.md` (si liste des sprints) |

---

## Spec

1. **`CLAUDE.md`** : ajouter la ligne de statut Sprint 35 dans le bloc « Statut sprint »
   (résumé : étude 3 conditions de features × F1/acc, board ré-architecturé, fix HDC×monitoring,
   12 heatmaps, chiffres mesurés).
2. **`docs/roadmap_phase2.md`** : passer Sprint 35 de ⬜ à ✅ avec le bilan.
3. **Bilan** : compléter le tableau « Bilan (à compléter) » de `S3500_sprint_35.md`
   (statut/temps réel/notes par tâche).
4. **Graphe** : invoquer le skill `graphify_sprint_update` (`skills/graphify_sprint_update.md`)
   — il évalue si un update du graphe est pertinent puis le lance (nouveaux scripts/configs/conditions).

**Règle** : aucun chiffre inventé dans le statut CLAUDE.md ; refléter les mesures réelles.

## Vérification

```bash
grep -q "Sprint 35" CLAUDE.md && grep -q "Sprint 35" docs/roadmap_phase2.md && echo OK
```
