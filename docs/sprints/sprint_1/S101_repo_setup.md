# S1-01 — Créer le dépôt GitHub + structure de dossiers

| Champ | Valeur |
|-------|--------|
| **ID** | S1-01 |
| **Sprint** | Sprint 1 — Semaine 1 (15–22 avril 2026) |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 1h |
| **Dépendances** | — |
| **Fichiers cibles** | `README.md`, `.gitignore`, `pyproject.toml` |

---

## Objectif

Disposer d'un dépôt GitHub `cl-embedded` opérationnel : clonable proprement, structure de dossiers conforme à `CLAUDE.md`, environnement installable sans erreur via `pip install -e ".[dev]"`.

**Critère de succès** : un collaborateur peut cloner le repo et lancer `pytest tests/` en moins de 5 minutes.

---

## Sous-tâches

### 1. Créer le dépôt GitHub

- Aller sur GitHub → New repository → `cl-embedded`
- Visibilité : **à confirmer avec Arnaud** (public recommandé pour valorisation, privé si données sensibles) — `TODO(arnaud)`
- Ne pas initialiser avec README ni .gitignore depuis GitHub (fichiers déjà locaux)
- Ajouter les remote : `git remote add origin git@github.com:<org>/cl-embedded.git`

### 2. Vérifier `.gitignore`

S'assurer que les éléments suivants sont exclus :

```
data/raw/
data/processed/
experiments/
__pycache__/
*.pyc
.env
*.egg-info/
dist/
.ipynb_checkpoints/
```

Référence : contrainte CLAUDE.md — *"Ne pas committer des données brutes"*.

### 3. Vérifier `pyproject.toml`

Contrôler la présence et la cohérence de :
- `[project]` : `name`, `version`, `requires-python = ">=3.10"`, `dependencies`
- `[project.optional-dependencies]` : groupe `dev` (pytest, black, ruff, jupyter)
- `[tool.black]` : `line-length = 100`, `target-version = ["py310"]`
- `[tool.ruff]` : `line-length = 100`, règles `E, F, I, N, W`
- `[tool.pytest.ini_options]` : `testpaths = ["tests"]`

### 4. Vérifier `README.md`

Le README doit comporter les sections suivantes :
- Badge statut (optionnel à ce stade)
- **Overview** — résumé du projet en 3 lignes
- **The Three Models** — tableau M1/M2/M3 avec RAM cible
- **Scientific Positioning — Triple Gap** — lien vers `docs/context/triple_gap.md`
- **Quick Start** — commandes `pip install`, `train`, `evaluate`
- **Project Structure** — arborescence principale

### 5. Installer l'environnement et valider

```bash
pip install -e ".[dev]"
python -c "import torch; import sklearn; import pandas; print('OK')"
pytest tests/ -v
```

Tous les imports doivent passer sans erreur. Les tests peuvent être vides à ce stade.

### 6. Premier commit + push

```bash
git init  # si pas encore fait
git add README.md .gitignore pyproject.toml requirements.txt CLAUDE.md
git add configs/ docs/ src/ tests/ scripts/ skills/ notebooks/ experiments/
git commit -m "feat: initial project structure — cl-embedded Sprint 1"
git push -u origin main
```

> Ne pas inclure `data/` ni `experiments/` avec des résultats — vérifier que `.gitignore` est actif.

---

## Critères d'acceptation

- [x] Repo GitHub accessible à l'URL attendue — [github.com/lrivals/cl-embedded](https://github.com/lrivals/cl-embedded)
- [x] `git clone <url> && pip install -e ".[dev]"` fonctionne sans erreur (env `ln2_Project_Env`)
- [x] `pytest tests/ -v` — 21/21 tests passent
- [x] `data/raw/` absent du repo distant
- [x] Structure conforme à `CLAUDE.md` (tous les dossiers présents)
- [x] `black --check src/` passe (ou code vide)
- [x] `ruff check src/` passe (ou code vide)

**Complété le 2 avril 2026** — commit `96fcc0a`, push HTTPS vers `origin/main`.

---

## Questions ouvertes

- `TODO(arnaud)` : visibilité du repo GitHub — public ou privé ? Sous quelle organisation (ISAE-SUPAERO, ENAC, personnelle) ?
- `TODO(arnaud)` : accès à donner aux encadrants (arnaud, dorra, fred) — invitations GitHub à préparer ?
