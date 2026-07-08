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
`Capogrosso2023TinyML`, `DeLange2021Survey`, `Hurtado2023CLPdM`, `Benatti2019HDC`.

### Makefile
- `make fr` → `main_fr.pdf` (pdflatex + bibtex + pdflatex ×2).
- `make en` → `main_en.pdf`. `make all` → les deux. `make clean` → artefacts.

## Vérification

```bash
cd docs/article/ewc_int8_mcu && make all   # main_fr.pdf + main_en.pdf sans erreur
```

> `TODO(arnaud)` : classe LaTeX cible (IEEEtran conf. TinyML vs article générique) et longueur visée.
