# Sprint 37 — Pipeline de publication GitLab (export sanitisé)

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 37 |
| **Semaine** | 18 – 24 août 2026 |
| **Statut** | ✅ Implémenté (S3701–S3709) |
| **Priorité globale** | 🔴 Critique — Mettre en place la **transformation reproductible « dépôt de travail → version GitLab »** pour publier le projet sur le **GitLab ISAE-SUPAERO**, propre et professionnel (aucune trace d'outillage interne / IA générative), utilisable par les prochains contributeurs. La transformation doit couvrir **le code existant ET les ajouts futurs**, et n'est jamais contournée : le dépôt de travail n'est **jamais** poussé tel quel. |
| **Durée estimée totale** | ~16h |
| **Dépendances** | `git` ✅ · `PyYAML` ✅ (déjà dans les deps) · motif CLI argparse des `scripts/export_weights_c.py` ✅ · `pyproject.toml` (black/ruff) ✅ · `.gitignore` ✅ · inventaire des traces (CLAUDE.md, `skills/`, `graphify-out/`, `.claude/`, mentions Claude/graphify dans docs+code+notebooks) ✅ |

---

## Contexte et motivation

Le projet doit être publié sur le **GitLab ISAE-SUPAERO** pour être repris par de futurs
utilisateurs. Trois exigences structurantes (validées avec l'auteur) :

1. **Jamais pousser le dépôt de travail tel quel.** Il existe toujours une étape de
   **transformation / préparation** qui produit une *version GitLab* dérivée de l'état courant.
2. **Propreté professionnelle.** La version GitLab ne doit montrer **aucune trace** d'assistant IA
   ou d'outillage interne (mentions Claude/Anthropic, footers de co-auteur, `graphify`, `.claude/`,
   `skills/`, `CLAUDE.md`, emojis-signature).
3. **Couvrir aussi les ajouts futurs.** Chaque nouvelle feature complète et testée doit pouvoir
   être re-publiée par la même chaîne, sans réécriture manuelle, après validation de l'auteur.

### Décisions validées (utilisateur)

- **Mécanisme** : *dépôt exporté séparé* — un dépôt git indépendant (hors du dépôt de travail), que
  l'auteur pousse manuellement vers GitLab. Bénéfice clé : l'historique du dépôt de travail (et donc
  le footer `Co-Authored-By` du commit IA recensé) **n'atteint jamais GitLab** — chaque release est
  un snapshot propre.
- **Déclencheur** : *commande locale manuelle* (`make gitlab-release`), lancée après validation
  d'une feature. Pas de mirror automatique non désiré ; la validation est explicite par
  construction (la commande lance les tests avant l'export).
- **Fichiers IA** : *exclure entièrement* `CLAUDE.md`, `skills/`, `graphify-out/`, `.claude/`, et
  **générer une doc d'onboarding neutre** (README + CONTRIBUTING) à la place.

### État des lieux (inventaire)

- Traces recensées : `CLAUDE.md`, `skills/`, `graphify-out/` (~79 Mo), `.claude/`, **1 commit** avec
  footer `Co-Authored-By: Claude`, ~53 mentions `graphify` (18 fichiers), 56 docs référençant
  `CLAUDE.md`, mentions « CLAUDE.md »/« Claude » dans des commentaires `.py` et cellules `.ipynb`.
- Infra existante réutilisée : `git ls-files`, `PyYAML`, conventions argparse, GitHub Actions.

---

## Architecture livrée

```
configs/gitlab_release.yaml          ← source de vérité (exclude_paths, forbidden_patterns,
                                        rewrite_rules, allowlist, neutral_docs, release)
scripts/check_ai_traces.py           ← scanner réutilisable (gate dur + scan manuel d'un arbre)
scripts/prepare_gitlab_release.py    ← transformation : git ls-files → exclusions → réécritures →
                                        docs neutres → gate → dépôt séparé (commit propre)
docs/gitlab/README_gitlab.md         ← gabarit README neutre (déposé comme README.md dans l'export)
docs/gitlab/CONTRIBUTING.md          ← gabarit onboarding dev neutre
docs/gitlab_publication.md           ← runbook (workflow, 1ère config GitLab, ajout de règles)
.github/workflows/ai-trace-guard.yml ← garde-fou CI : `--check-only` (l'export reste-t-il propre ?)
Makefile                             ← `gitlab-release` / `gitlab-release-dry` / `gitlab-check`
tests/test_gitlab_release.py         ← 12 tests (exclusions, réécritures, gate, idempotence, dry-run)
```

**Principe « couvre les ajouts futurs »** : la transformation est **pilotée par la config** (denylist
de chemins + regex), donc tout nouveau fichier est traité par les mêmes règles sans toucher au code ;
et le **garde-fou** (`--check-only`, en CI et en local) construit l'export dans un dossier jetable et
applique le gate — toute trace non couverte par une règle fait échouer tôt, avant publication.

---

## Contraintes techniques honorées

- Le `.git` et l'historique du **dépôt de travail ne sont jamais modifiés** ni poussés.
- L'export est un **dépôt git indépendant** ; commit à message neutre, sans footer.
- Export **idempotent** : relancer sans changement source ne crée aucun nouveau commit.
- Aucune nouvelle dépendance lourde (PyYAML déjà présent).
- Le scan ignore les blobs binaires/base64 (frontières de mot `\b` → pas de faux positifs notebooks).

## Critères de succès

1. `make gitlab-release-dry` liste exclusions/docs neutres sans rien écrire. ✅
2. `make gitlab-release` produit un export **0 trace** (gate dur) dans un dépôt séparé. ✅
3. `check_ai_traces.py` sur l'export → exit 0 ; sur une trace semée → exit 1 avec rapport. ✅
4. `CLAUDE.md` / `skills/` / `graphify-out/` / `.claude/` absents de l'export ; docs neutres présentes. ✅
5. `git log` du dépôt séparé : commits propres, **sans footer IA**. ✅
6. Garde-fou `--check-only` (CI + `make gitlab-check`) vert sur le repo courant, rouge si trace non couverte. ✅
7. `pytest -k gitlab_release` vert (12/12). ✅

→ Détail des tâches : `S3701`–`S3709` dans ce répertoire.
