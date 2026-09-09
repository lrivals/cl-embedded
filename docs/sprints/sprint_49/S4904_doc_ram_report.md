# S4904 — Doc structurée des consommations RAM par expérience (export rapport)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 49 |
| **Priorité** | 🔴 Critique — livrable direct du CR (« créer un fichier qui documente les consommations RAM en détail »). |
| **Statut** | ✅ Implémenté — `scripts/aggregate_ram.py` → `summary.json` (indexé, ratio calculé) + `docs/context/ram_report.md` généré. |
| **Durée estimée** | 3h |
| **Dépendances** | S4903 (`exp_S49_ram/`) |
| **Fichiers cibles** | `docs/context/ram_report.md` (nouveau) · `experiments/exp_S49_ram/summary.json` |
| **Références** | CR §2 conclusion · §7 action 🔴 |

## Contexte

Le CR demande un **fichier de documentation détaillée des consommations RAM par expérience**, structuré pour
export vers le rapport. Cette tâche agrège `exp_S49_ram/` (lecture seule) en un document lisible + un
`summary.json` indexé.

## Spec

### 1. `summary.json` (agrégation, lecture seule)

Indexé `[model][dataset][condition][encoding][platform]` → `{data, bss, stack_peak_inference, stack_peak_update,
total, ratio_int8_vs_fp32}`. Généré par un agrégateur (`aggregate_ram.py` ou section du driver), **jamais à la
main**. Deltas int8 vs fp32 calculés, pas saisis.

### 2. `docs/context/ram_report.md` (document rapport)

```
# Consommations RAM par expérience — CL-Embedded
## 1. Méthode (renvoi ram_measurement.md : .data + .bss + pic de pile)
## 2. Tableau par modèle × dataset × encodage × plateforme
##    colonnes : .data | .bss | pic inférence | pic MAJ | total | ratio int8/fp32
## 3. Historique d'évolution du pic de pile (par modèle, phase par phase)
## 4. Constats (généraux, sans chiffre en dur — texte renvoyant aux cellules)
## 5. Limites & N/A honnêtes
```

Toutes les valeurs numériques du tableau sont **rechargées depuis `summary.json`** (le `.md` peut être généré
ou porter des renvois ; aucune valeur figée à la main).

### 3. Historique du pic

Section dédiée (CR §2 : « côté historique d'évolution du pic de pile durant exécution intéressant ») —
une entrée par modèle listant `stack_history` (phase → pic), pour montrer que la MAJ CL creuse plus la pile.

## Format de sortie

- `experiments/exp_S49_ram/summary.json` (indexé, deltas calculés).
- `docs/context/ram_report.md` (tableau + historique + constats).

## Contraintes

- Lecture seule sur `exp_S49_ram/` ; aucun recalcul de mesure.
- Aucun chiffre en dur dans le `.md` (rechargé/renvoi).
- Board et PC en colonnes distinctes.

## Vérification

```bash
test -f docs/context/ram_report.md experiments/exp_S49_ram/summary.json
python -c "import json;s=json.load(open('experiments/exp_S49_ram/summary.json'));print(list(s))"  # indexé par modèle
grep -i "pic de pile\|historique\|total" docs/context/ram_report.md
```
