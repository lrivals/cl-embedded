# Makefile — article standalone « EWC INT8 sur MCU » (Sprint 40)
# Cibles : make fr | make en | make all | make clean
#
# Chaîne LaTeX : pdflatex -> bibtex -> makeglossaries -> pdflatex x2.
# makeglossaries est OBLIGATOIRE : pdflatex ne collecte que le .acn, c'est
# makeglossaries qui produit le .acr lu par \printglossary. Sans elle, la liste
# des acronymes sort vide et tous les liens \gls pointent vers une ancre absente.
# TODO(arnaud): passer à IEEEtran (conférence TinyML) => apt install texlive-publishers.

LATEX   := pdflatex
BIBTEX  := bibtex
GLOSS   := makeglossaries
# Pas de -halt-on-error : la premiere passe n'a pas encore de .acr et
# \printglossary avertit legitimement. Le compte d'erreurs ci-dessous fait foi.
LFLAGS  := -interaction=nonstopmode

# `**` n'est PAS expansé récursivement par GNU make : le motif `sections/**/*.tex`
# produisait une liste vide, donc éditer une section ne déclenchait aucune
# recompilation. Les deux langues sont listées séparément pour qu'une retouche
# française ne reconstruise pas l'anglais.
SECTIONS_FR := $(wildcard sections/fr/*.tex) glossary_fr.tex
SECTIONS_EN := $(wildcard sections/en/*.tex) glossary_en.tex
FIGURES     := $(wildcard figures/*.png)

.PHONY: all fr en watch watch-en clean

all: fr en

fr: main_fr.pdf
en: main_en.pdf

# Prérequis propres à chaque langue ; la recette vient de la règle générique.
main_fr.pdf: $(SECTIONS_FR)
main_en.pdf: $(SECTIONS_EN)

# Règle générique : $* ∈ {main_fr, main_en}. references.bib partagé.
# Les figures sont générées par le catalogue `article_ewc_int8` puis synchronisées ici :
# les déclarer en prérequis évite qu'un PDF reste compilé sur des figures périmées.
main_%.pdf: main_%.tex references.bib $(FIGURES)
	$(LATEX) $(LFLAGS) main_$*
	-$(BIBTEX) main_$*
	-$(GLOSS) main_$*
	$(LATEX) $(LFLAGS) main_$*
	$(LATEX) $(LFLAGS) main_$*
	@echo "--- main_$* : erreurs LaTeX : $$(grep -a -c '^!' main_$*.log)"
	@echo "--- main_$* : liens de glossaire casses : $$(grep -a -c 'has been referenced but does not exist' main_$*.log)"

# Boucle de rédaction : recompile dès qu'une source change (Ctrl-C pour sortir).
# Volontairement en shell pur — `latexmk` n'est pas installé sur la machine.
watch:    LANGUE := fr
watch-en: LANGUE := en
watch watch-en:
	@echo "Surveillance de main_$(LANGUE) — Ctrl-C pour arrêter."
	@while true; do \
		$(MAKE) --no-print-directory main_$(LANGUE).pdf || true; \
		sleep 2; \
	done

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.pdf sections/*/*.aux \
	      *.glo *.gls *.glg *.acn *.acr *.alg *.ist
