# S4004 — Structure LaTeX de l'article (squelette partagé FR/EN)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 40 |
| **Priorité** | 🔴 Critique — socle des deux versions |
| **Statut** | ✅ Implémenté — arbre `docs/article/ewc_int8_mcu/` (classe `article`, bib autonome, Makefile), `make all` compile FR+EN |
| **Durée estimée** | ~4h |
| **Dépendances** | S4003 (figures) · `references.bib` du projet manuscrit |
| **Fichiers cibles** | `docs/article/ewc_int8_mcu/` (`main_fr.tex`, `main_en.tex`, `references.bib`, `Makefile`) |
| **Références** | clés BibTeX projet (CLAUDE.md) · `docs/triple_gap.md` |

## Contexte

Article **standalone** (indépendant du manuscrit), en **deux versions FR + EN**. Ce ticket pose le squelette
LaTeX commun, la bibliographie et le `Makefile`, avant la rédaction du contenu (S4005/S4006).

## Spec

### Arborescence `docs/article/ewc_int8_mcu/`
```
main_fr.tex          ← document français (classe article/IEEEtran, à décider avec Arnaud)
main_en.tex          ← document anglais (miroir)
references.bib       ← clés projet réutilisées
figures/             ← liens/copies depuis docs/figures/sprint40_article/
Makefile             ← make fr | make en | make all | make clean
```

### Plan de l'article (identique FR/EN)
| Section | Contenu | Appui données |
|---------|---------|---------------|
| Abstract | EWC sur MCU : parité FP32, piège PTQ naïve, récupération calibrée | — |
| 1. Introduction | maintenance prédictive embarquée, **triple gap**, contribution | `triple_gap.md` |
| 2. Related work | CL sur MCU, quantification INT8/QAT/PTQ | BibTeX (voir ci-dessous) |
| 3. Méthode | EWC + portage C ; protocole apparié PC↔board ; schémas quantif (legacy/per-channel/Q15) | S36/S39 |
| 4. Setup expérimental | NUCLEO-F439ZI, Pronostia + Monitoring, **conditions identiques** (seed/ordre/normalisation) | exp_S36, exp_S40 |
| 5. Résultats | (a) parité FP32 PC↔board ; (b) INT8 legacy→v2 ; (c) latence/RAM Gap 2/3 | Figures S4003 |
| 6. Discussion | paradoxe latence FPU ; honnêteté « mesuré board » vs « émulé PC » | S39 |
| 7. Conclusion & travaux futurs | récupération board confirmée ; SIMD CMSIS-NN | S3910/S3917 |

### Bibliographie (clés projet à réutiliser)
`Kirkpatrick2017EWC`, `Ren2021TinyOL`, `Ravaglia2021QLRCL`, `Kwon2023LifeLearner`,
`Capogrosso2023TinyML`, `DeLange2021Survey`, `Hurtado2023CLPdM`, `Benatti2019HDC`,
`Nectoux2012Pronostia` (cité en \S\,4, jeu de données Pronostia).

### Makefile
- `make fr` → `main_fr.pdf` (pdflatex + bibtex + pdflatex ×2).
- `make en` → `main_en.pdf`. `make all` → les deux. `make clean` → artefacts.
- `make watch` / `make watch-en` → boucle de rédaction (shell pur, `latexmk` absent
  de l'environnement).
- Chaîne complète : `pdflatex → bibtex → makeglossaries → pdflatex ×2`. L'étape
  `makeglossaries` est **obligatoire** : `pdflatex` ne collecte que le `.acn`, c'est
  `makeglossaries` qui produit le `.acr` lu par `\printglossary`. Sans elle la liste
  des acronymes sort vide et tous les liens `\gls` pointent vers une ancre absente.
- `-halt-on-error` retiré : la première passe n'a pas encore de `.acr` et
  `\printglossary` avertit légitimement. Le build affiche à la place, comme le
  manuscrit, le compte d'erreurs LaTeX et de liens de glossaire cassés.

### Glossaire des acronymes
`glossary_fr.tex` / `glossary_en.tex` à la **racine** de l'article (pas sous
`sections/`, que `test_sprint40_article.py::_tex_sources` balaie pour le miroir
numérique FR/EN). Patron repris du manuscrit : `\glsnoexpandfields`,
`\setacronymstyle{long-short-desc}`, `\newacronym[description={...}]`.
**24 acronymes**, mêmes clés dans les deux langues, 21 descriptions adaptées du
manuscrit et 6 écrites pour l'article (SIMD, MAC, FLOP, BOP, CMSIS-NN, WFI).
Liste imprimée après la bibliographie, comme dans le manuscrit.

Règle de balisage : `\gls{}` à la première occurrence **dans le corps**, jamais dans
le titre, l'abstract ni les mots-clés (un abstract doit se lire seul). Les
développements manuels du corps ont été retirés là où `\gls` prend le relais, sinon
le sigle serait développé deux fois.

### Préparation de la rédaction locale (chaîne du rapport de stage)
La chaîne TeX Live 2023 installée pour le rapport suffit : ni `siunitx` (macros de
repli `\providecommand` dans les deux `main_*.tex`), ni `biblatex`, ni `glossaries`.
Trois correctifs ont été apportés avant d'ouvrir la boucle d'édition :

- **Prérequis Makefile** — `$(wildcard sections/**/*.tex)` ne listait *rien* (GNU make
  n'expanse pas `**` récursivement) : éditer une section ne déclenchait aucune
  recompilation et l'on relisait un PDF périmé. Remplacé par `SECTIONS_FR` /
  `SECTIONS_EN`, déclarés en prérequis propres à chaque langue.
- **Artefacts hors dépôt** — `.gitignore` local + `git rm --cached` sur les 10 fichiers
  `main_{fr,en}.{aux,bbl,blg,out,pdf}` qui étaient suivis, pour que les diffs de
  rédaction ne montrent que du texte.
- **Citation manquante** — `Nectoux2012Pronostia` figurait dans `references.bib` sans
  être cité (8 `\bibitem` sur 9 entrées) ; ajouté dans `04_setup.tex` FR **et** EN.

## Vérification

```bash
cd docs/article/ewc_int8_mcu && make all   # main_fr.pdf + main_en.pdf sans erreur
```

> `TODO(arnaud)` : classe LaTeX cible (IEEEtran conf. TinyML vs article générique) et longueur visée.
