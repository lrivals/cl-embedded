# Makefile — article standalone « EWC INT8 sur MCU » (Sprint 40)
# Cibles : make fr | make en | make all | make clean
#
# Chaîne LaTeX : pdflatex -> bibtex -> pdflatex x2 (résolution des \cite et \ref).
# TODO(arnaud): passer à IEEEtran (conférence TinyML) => apt install texlive-publishers.

LATEX   := pdflatex
BIBTEX  := bibtex
LFLAGS  := -interaction=nonstopmode -halt-on-error

.PHONY: all fr en clean

all: fr en

fr: main_fr.pdf
en: main_en.pdf

# Règle générique : $* ∈ {main_fr, main_en}. references.bib partagé.
main_%.pdf: main_%.tex references.bib $(wildcard sections/**/*.tex)
	$(LATEX) $(LFLAGS) main_$*
	-$(BIBTEX) main_$*
	$(LATEX) $(LFLAGS) main_$*
	$(LATEX) $(LFLAGS) main_$*

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.pdf sections/*/*.aux
