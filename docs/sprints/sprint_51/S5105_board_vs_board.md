# S5105 — Réorientation board↔board (perspective)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 51 |
| **Priorité** | 🟢 Bas — perspective documentée (dépend d'une 2e carte, non garantie). |
| **Statut** | 📝 Doc — spec/perspective ; implémentation conditionnée à une 2e carte |
| **Durée estimée** | 2h |
| **Dépendances** | Harnais PC↔board S36 ✅ (`run_sprint36_board.py`, `board_pc_parity.py`) · une **2e carte** (non garantie) |
| **Fichiers cibles** | `docs/sprints/sprint_51/S5105_board_vs_board.md` |
| **Références** | CR §1 « Réorientation de la démarche » |

## Contexte

Le CR conclut que la mise en place (mêmes données, même évaluation) est à **réutiliser pour comparer deux cartes
entre elles**, comparaison plus pertinente et directement interprétable que PC↔board. Cette tâche documente
**comment** réorienter le harnais existant, sans exécuter (dépend d'une 2e carte).

## Spec

### 1. Réutilisation du harnais S36

Le harnais S36 fixe déjà : mêmes groupes de données, même protocole MAJ+éval, parité par échantillon, DWT. Pour
board↔board, il suffit de **paramétrer la cible** (`--port`/build par carte) et de comparer deux runs board au
lieu de board vs PC.

| Élément S36 | board↔board |
|-------------|-------------|
| `run_sprint36_board.py` | exécuté sur carte A puis carte B (mêmes échantillons/seed/ordre) |
| `board_pc_parity.py` | devient `board_board_parity` (A vs B au lieu de board vs PC) |
| Métriques | RAM totale (S49), latence DWT, énergie (S50), accuracy — comparables **car même architecture d'exécution** |

### 2. Pourquoi c'est interprétable (vs PC↔board)

Deux MCU exécutent séquentiellement, sans overhead de scheduling multi-cœur → les écarts (latence, énergie,
RAM) reflètent **le matériel**, pas l'OS. Le score système (S5101) devient alors un vrai critère de choix de
carte.

### 3. Conditions

- Nécessite une 2e carte (ex. autre STM32, ou carte sans FPU / INT8 natif — lien avec la perspective quantif CR).
- Tant qu'indisponible : **perspective**, aucun run, aucun chiffre.

## Format de sortie

`docs/sprints/sprint_51/S5105_board_vs_board.md` (procédure de réorientation + conditions).

## Contraintes

- Ne pas exécuter sans 2e carte ; aucun résultat fabriqué.
- Réutiliser S36 (0 réécriture du harnais).

## Vérification

```bash
grep -i "board.*board\|2e carte\|S36\|séquentiel" docs/sprints/sprint_51/S5105_board_vs_board.md
```
