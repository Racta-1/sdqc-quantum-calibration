# Paper source

`main.tex` (IEEEtran conference format). All V5.1 numbers come from the 10-seed run
summarised in `../paper_results/v5_1_10seeds/SUMMARY.md`:

    python scripts/run_experiment.py --feature-device cuda --jobs 6 --seeds 0 1 2 3 4 5 6 7 8 9 --out results/v5_1_10seeds

Figures needed in this folder:
- `sdqc_intro.pdf`, `sdqc_methodology.pdf`, `v4_partB_pneumoniamnist.pdf`, `hard_case.pdf` (included; vector PDFs with embedded TrueType fonts)
- `hard_case.pdf` is drawn from the 10-seed run; `scripts/analyze.py` writes it to `<results>/hard_case.pdf`

Upload the folder to Overleaf (or run `pdflatex main.tex` twice).
