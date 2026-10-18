# Manuscript

One LaTeX source, two builds: `main.pdf` (full, for arXiv) and `main_anon.pdf`
(double-blind, for Letters in Biomathematics). Structure follows the journal's
Introduction, Materials and Methods, Results, Discussion. Layout: two columns,
10 pt Times (`newtx`), letter paper, conference-style numbered headings and
"Figure 1." captions; all figures span both columns (`figure*`). The double-blind
build adds line numbers and a "confidential review copy" header.

## Build locally

```bash
make -C paper            # generated tables + figures (if missing) + both PDFs
make -C paper full       # only main.pdf
make -C paper anon       # only main_anon.pdf
make -C paper clean
```

`generated/presets.tex` and `generated/numbers.tex` are written by
`scripts/paper_tables.py` from the code and the results CSVs; every number quoted in
the text is a macro from `numbers.tex`, never typed by hand. Rerun `make -C paper
generated` after any experiment changes.

## Move to Overleaf

Upload the contents of `paper/` (keep the `sections/` and `generated/` folders) and
the figure PDFs from `figures/` into a `figures/` folder next to `main.tex`; the
`\graphicspath` already looks there. Set the compiler to pdfLaTeX and the main
document to `main.tex`. For the double-blind build, uncomment the line
`% \def\ANONYMOUS{1}` at the top of `main.tex`.

## Files

| file | content |
|---|---|
| `main.tex` | preamble, anonymity switch, title block, abstract, keywords, inputs |
| `sections/01_introduction.tex` … `05_conclusion.tex` | the body, one file per section |
| `sections/06_backmatter.tex` | code and data availability, author contributions, conflict of interest, acknowledgements |
| `sections/appendix_a_parameters.tex`, `appendix_b_derivations.tex` | parameter table (generated), derivations |
| `generated/` | tables and number macros produced from code and results |
| `references.bib` | bibliography; every entry must be verified against the publisher record before submission |
| `notebook_corrections.md` | what the course notebook got wrong, with evidence files |

Drafting notes appear in gray via `\note{...}`; delete the macro's uses before submission.
