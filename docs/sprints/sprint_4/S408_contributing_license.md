# S4-08 — `CONTRIBUTING.md` + `LICENSE`

| Champ | Valeur |
|-------|--------|
| **ID** | S4-08 |
| **Sprint** | Sprint 4 — Semaine 4 (6–13 mai 2026) |
| **Priorité** | 🟢 Nice-to-have |
| **Durée estimée** | 1h |
| **Dépendances** | — |
| **Fichiers cibles** | `CONTRIBUTING.md`, `LICENSE` (racine du dépôt) |
| **Statut** | ⬜ Non démarré |

---

## Objectif

Ajouter les fichiers de gouvernance standard du dépôt avant la présentation aux encadrants et l'éventuelle publication sur GitHub.

**Critère de succès** : `ls CONTRIBUTING.md LICENSE` dans la racine du dépôt retourne les deux fichiers.

---

## Sous-tâches

### 1. `LICENSE` — MIT

```
MIT License

Copyright (c) 2026 Léonard Rivals — ISAE-SUPAERO (DISC)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 2. `CONTRIBUTING.md`

Structure recommandée :

```markdown
# Guide de contribution — CL-Embedded

## Prérequis

- Python ≥ 3.10
- pip install -e ".[dev]"
- STM32CubeMX 6.17.0 (Phase 2 uniquement)

## Workflow de développement

1. Créer une branche pour chaque sprint : `git checkout -b sprint-N`
2. Respecter les conventions CLAUDE.md (style, nommage, annotations # MEM:)
3. Lancer les tests avant tout commit : `pytest tests/ -v`
4. Formatter : `black src/ && ruff check src/ --fix`
5. Mettre à jour le fichier de sprint dans `docs/sprints/sprint_N/`

## Structure des expériences

Toute expérience doit :
- Avoir un ID unique `exp_XXX_description`
- Contenir un `config_snapshot.yaml` (reproductibilité)
- Exporter `metrics.json` et `memory_report.json`
- Utiliser `seed=42` par défaut

## Conventions de commit

- `feat(sprintN): description` — nouvelle fonctionnalité
- `fix(module): description` — correction de bug
- `docs(sprintN): description` — documentation uniquement
- `exp(expXXX): description` — nouvelle expérience

## Contacts

- Arnaud Dion (superviseur ISAE-SUPAERO) — questions architecture CL
- Dorra Ben Khalifa — questions quantification et hardware MCU
- Frédéric Zbierski (Edge Spectrum) — contexte industriel
```

---

## Commandes de création

```bash
# Depuis la racine du dépôt
# 1. Créer LICENSE (texte MIT complet ci-dessus)
# 2. Créer CONTRIBUTING.md (structure ci-dessus)

# Vérifier
ls LICENSE CONTRIBUTING.md
git add LICENSE CONTRIBUTING.md
# git commit effectué à la fin du sprint avec les autres fichiers S4
```

---

## Critères d'acceptation

- [ ] `LICENSE` présent à la racine, contenu MIT avec année 2026 et auteur Léonard Rivals
- [ ] `CONTRIBUTING.md` présent à la racine avec les sections : Prérequis, Workflow, Expériences, Commits, Contacts
- [ ] `git status` montre les deux fichiers comme tracked

---

## Questions ouvertes

- `TODO(arnaud)` : Licence MIT ou license académique ISAE-SUPAERO ? Le dépôt sera-t-il public sur GitHub ou interne ?
