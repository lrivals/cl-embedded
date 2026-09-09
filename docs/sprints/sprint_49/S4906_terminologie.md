# S4906 — Terminologie : « RAM modulable » → « mémoire non constante »

| Champ | Valeur |
|-------|--------|
| **Sprint** | 49 |
| **Priorité** | 🟡 Moyen — correction terminologique demandée (CR §2 et §7). |
| **Statut** | ✅ Implémenté — 4 occurrences corrigées dans `presentation_ram_measurement.md` (`ram_breakdown.py` n'en contenait aucune ; grep hors CR/sprint_49 VIDE). |
| **Durée estimée** | 1h |
| **Dépendances** | — (indépendante ; à coordonner avec S4905 pour les légendes) |
| **Fichiers cibles** | `docs/presentation_ram_measurement.md` · `scripts/ram_breakdown.py` |
| **Références** | CR §2 « Remplacer la mention "RAM modulable" par "mémoire non constante" dans tous les documents » |

## Contexte

Le terme « RAM modulable » doit être remplacé par « **mémoire non constante** » dans tous les documents.
Inventaire réel (grep repo, hors CR qui énonce l'instruction) :

| Fichier | Occurrences | Note |
|---------|:---:|------|
| `docs/presentation_ram_measurement.md` | 3 (lignes 221, 227, 234) | « modulable (état CL) », en-tête de table, « statique et modulable » |
| `scripts/ram_breakdown.py` | 1 (ligne ~254 / commentaire de table) | « Split statique/modulable par modèle » |
| `docs/presentation/CR_reunion_16juillet2026.md` | 2 (65, 172) | **NE PAS MODIFIER** — c'est l'instruction elle-même |

## Spec

### 1. Remplacements

- « RAM modulable » / « modulable » (au sens mémoire) → « mémoire non constante » / « non constante ».
- Adapter la grammaire : « état CL non constant », « statique vs non constante », en-têtes de tables.
- **Exclure** le CR du 16 juillet (les 2 occurrences y décrivent la tâche).

### 2. Cohérence

Coordonner avec S4905 (légendes de figures) : le terme dans les plots suit la même règle.

## Format de sortie

Éditions en place de `docs/presentation_ram_measurement.md` et `scripts/ram_breakdown.py`.

## Contraintes

- Ne pas altérer le sens technique (mémoire non constante = `.bss` in-place mis à jour à bord).
- Ne pas toucher le CR source.
- Vérifier qu'aucune autre occurrence n'apparaît après coup (nouveaux docs S49).

## Vérification

```bash
# plus aucune occurrence hors CR :
grep -rniE "modulable" --include=*.md --include=*.py docs/ src/ scripts/ | grep -v "CR_reunion_16juillet2026"
# doit être VIDE
grep -ci "mémoire non constante" docs/presentation_ram_measurement.md    # > 0
```
