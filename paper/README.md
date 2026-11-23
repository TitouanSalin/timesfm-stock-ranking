# Paper (ACM ICAIF format)

LaTeX sources of the paper, in the format of the ACM International Conference on AI in Finance (ICAIF): class `acmart` 2.21 with the `sigconf` option, included here so that the paper compiles without a full TeX installation. The call for papers of ICAIF 2026 allowed 8 pages including references, double-blind review and no supplementary material.

- `main.tex`: main document. For a double-blind submission, use `\documentclass[sigconf,review,anonymous]{acmart}`.
- `sec/`: one file per section.
- `tables/`: table bodies written by `make_tables.py` from the saved results.
- `figures/`: pgfplots figures, drawn from `figures/data/*.tsv`, written by `figures/export_data.py`.
- `main.pdf`: compiled paper.

Regenerating the tables and the figure data needs the daily series in `data/predictions/`, so it runs after the experiments:

```bash
python paper/make_tables.py
python paper/figures/export_data.py
```

Compilation, in this folder: `pdflatex main && bibtex main && pdflatex main && pdflatex main`. On Overleaf, upload the content of this folder as a new project.
