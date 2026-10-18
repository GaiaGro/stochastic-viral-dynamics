PY ?= ../.venv/bin/python
LATEXMK = latexmk -pdf -interaction=nonstopmode -halt-on-error
FIGS = $(wildcard ../figures/fig*.pdf)

.PHONY: all generated figures full anon clean

all: generated figures full anon

generated:
	$(PY) ../scripts/paper_tables.py

figures:
	@[ -n "$(FIGS)" ] || $(MAKE) -C .. figures PY=.venv/bin/python

full:
	$(LATEXMK) -jobname=main main.tex

anon:
	$(LATEXMK) -jobname=main_anon -usepretex='\def\ANONYMOUS{1}' main.tex

clean:
	latexmk -C -jobname=main main.tex
	latexmk -C -jobname=main_anon main.tex
	rm -f main.bbl main_anon.bbl
