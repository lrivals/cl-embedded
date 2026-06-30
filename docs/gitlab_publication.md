# Publication GitLab — Runbook

Ce document décrit comment produire et publier la **version GitLab** (ISAE-SUPAERO) du projet
à partir du dépôt de travail (GitHub privé), proprement et de façon reproductible.

## Principe

Le dépôt de travail **n'est jamais poussé tel quel** vers GitLab. Une étape de **transformation /
préparation** génère un *dépôt exporté séparé*, débarrassé de toute trace d'outillage interne ou
d'IA générative, avec une documentation d'onboarding neutre. C'est cette version exportée — et
elle seule — qui est poussée vers GitLab, après validation manuelle.

```
GitHub (privé, travail)                       GitLab (ISAE-SUPAERO, public/équipe)
   │                                                  ▲
   │  scripts/prepare_gitlab_release.py               │  git push (manuel)
   ▼                                                  │
git ls-files → exclusions → réécritures docs → docs neutres
            → gate dur (check_ai_traces.py) → dépôt séparé (commit propre)
```

## Quand publier

Uniquement quand une **feature complète et testée** a été validée (tests Python + Unity firmware
verts). La commande `make gitlab-release` lance les tests avant l'export et refuse de produire un
snapshot si la suite échoue.

## Composants

| Fichier | Rôle |
|---------|------|
| `configs/gitlab_release.yaml` | **Source de vérité** : chemins exclus, patterns interdits, réécritures, docs neutres, métadonnées de release |
| `scripts/check_ai_traces.py` | Scanner : échoue si une trace interdite subsiste (gate dur + check CI) |
| `scripts/prepare_gitlab_release.py` | Transformation complète → dépôt exporté séparé |
| `docs/gitlab/README_gitlab.md`, `docs/gitlab/CONTRIBUTING.md` | Gabarits des docs neutres déposées dans l'export |
| `.github/workflows/ai-trace-guard.yml` | Garde-fou CI : signale les nouvelles traces côté source |
| `Makefile` | `make gitlab-release` / `gitlab-release-dry` / `gitlab-check` |

## Workflow

### 1. Vérifier le plan (sans rien écrire)

```bash
make gitlab-release-dry
```

Affiche les fichiers exclus, les docs neutres générées, et le répertoire de sortie.

### 2. Produire l'export sanitisé

```bash
make gitlab-release            # lance pytest, exporte, applique le gate dur
```

L'export est écrit dans `../cl-embedded-gitlab` (configurable via `output_dir` ou `--output-dir`).
Le **gate dur** rescanne l'export : 0 trace tolérée. En cas d'échec, la commande indique les
fichiers fautifs — corrigez `configs/gitlab_release.yaml` puis relancez.

### 3. Première mise en place GitLab (une seule fois)

```bash
git -C ../cl-embedded-gitlab remote add gitlab <URL_DU_PROJET_GITLAB>.git
```

### 4. Publier

```bash
git -C ../cl-embedded-gitlab push gitlab main
# ou, en une fois après l'export :
make gitlab-release ARGS=--push
```

## Gérer les ajouts futurs

Le pipeline couvre automatiquement le **code futur** parce qu'il est piloté par
`configs/gitlab_release.yaml`, pas par du code en dur. Quand un nouvel ajout introduit une trace :

- **Nouveau fichier/répertoire interne** (skill, outil, sortie générée) → ajouter son chemin à
  `exclude_paths`.
- **Nouveau mot/marqueur à proscrire** → ajouter un regex à `forbidden_patterns`.
- **Doc utile mentionnant l'outillage interne** → ajouter une `rewrite_rules` (suppression de
  lignes ou de section) pour la conserver en version nettoyée.
- **Faux positif légitime** (ex. terme anglais contenant une sous-chaîne interdite) → l'ajouter à
  `allowlist` avec le glob du fichier concerné.

Le garde-fou CI (`ai-trace-guard.yml`) tourne sur chaque PR/push. Comme le dépôt de travail
contient légitimement des références internes partout, un scan brut serait toujours rouge :
l'invariant vérifié est donc **« l'export sort-il toujours propre ? »**. La CI construit l'export
dans un dossier jetable et applique le gate (`prepare_gitlab_release.py --check-only`). Si un ajout
introduit une trace **non couverte** par les règles, la CI échoue — on ajoute alors la règle
manquante dans `configs/gitlab_release.yaml`. En local :

```bash
make gitlab-check          # = prepare_gitlab_release.py --check-only (aucun commit)
```

## Garanties

- Le `.git` du dépôt de travail n'est **jamais** modifié par le pipeline.
- L'export est un **dépôt git indépendant** : aucun historique de commits du dépôt de travail (et
  donc aucun footer de co-auteur) n'atteint GitLab — chaque release est un snapshot propre.
- L'export est **idempotent** : relancer sans changement source ne produit aucun nouveau commit.
