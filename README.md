# pdftable

**Turn every page of a PDF into correct tables: local, explainable, measured.**

A PDF stores glyphs at coordinates, not rows and columns. `pdftable` rebuilds the tables a reader sees,
including borderless tables, tables printed sideways and multi-level shaded headers. Every value stays
attached to its **row label** and **column header**, and every table carries a reliability score.
Uncertain tables are flagged instead of being silently wrong.

- Runs **fully locally** as a Python command-line tool, with no cloud service.
- Reads the PDF's **own text layer** first; an OCR model (PaddleOCR PP-StructureV3) runs **only** for
  scanned pages or tables that fail the checks.
- Exports **JSON** (complete structure), **CSV** per table, a **review HTML** page and a **printable PDF report**.
- Comes with an **evaluation toolkit**: hand-verified ground truth, a relational correctness metric and an
  arithmetic self-check of printed totals.

---

## What "correct" means

A table is correct when every value keeps its relationship

```
row label  ->  column-header path  ->  value
Food delivery | FY24  ->  7,792
```

with merged headers, blank cells, title and unit intact, and nothing from the rest of the page mixed in.
Cell text is kept exactly as printed (`△2,979`, `(1,015)`, `78,72,500¹`); only whitespace is normalised.

Correctness is measured five ways, reported side by side:

| Measure | Question it answers |
|---|---|
| Detection P / R | Was every table found, and nothing else (bbox IoU >= 0.5)? |
| **Relationship F1** | Did each (row label, column-header path, value) fact survive? |
| GriTS Top / Con | How similar are the grid structure and cell content? |
| Exact cells, CER | How exact is the cell text? |
| Strict accuracy | Are shape, header levels, every span and every cell exact, with title and unit kept? |

## Results

Measured against 44 hand-verified tables on 30 pages: a 1-page Japanese balance sheet, plus 43 tables on
29 pages of a 371-page annual report. 29 of those tables lie on randomly drawn pages that were transcribed
**blind**, before looking at the output.

| Split (tables) | Detection P / R | Relationship F1 | Exact cells | Strict |
|---|---|---|---|---|
| Development (14) | 1.00 / 1.00 | 1.000 | 1.000 | 14 / 14 |
| Held-out A (14) | 1.00 / 0.86 | 0.997 | 1.000 | 12 / 14 |
| Held-out B (15) | 1.00 / 0.80 | 0.921 | 0.983 | 11 / 15 |
| Japanese balance sheet (1) | 1.00 / 1.00 | 1.000 | 1.000 | 1 / 1 |
| **All (44)** | **1.00 / 0.89** | **0.986** | **0.995** | **38 / 44** |

- No fake tables were reported on any annotated page (precision 1.00).
- On their **first blind run** the held-out sets scored relationship F1 **0.84** and **0.80**. That is the
  honest expectation for unseen pages of this kind of document.
- The arithmetic audit finds that 342 of 370 printed totals in the annual report (**92.4%**) re-add
  correctly from the extracted cells.
- The 371-page report runs in about **6 s** natively (20 CPU workers), or about **5 min** when the 20
  image-heavy pages also go through OCR.

Accuracy is only claimed where it was measured. A new kind of document needs its own verified sample
(see [Measuring accuracy on your own PDFs](#measuring-accuracy-on-your-own-pdfs)).

---

## Installation

Requires Python 3.11 (Linux, macOS or WSL).

```bash
git clone https://github.com/<your-username>/ur_AI.git
cd ur_AI
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt            # core: PyMuPDF + NumPy (enough for digital PDFs)
pip install -r requirements-visual.txt     # optional: OCR fallback for scanned pages (PaddleOCR)
pip install -r requirements-dev.txt        # optional: pytest
```

## Usage

```bash
# extract every table: JSON, HTML, CSV per table and a PDF report
python main.py extract your.pdf -o output -v

# fastest: native text layer only, no OCR
python main.py extract your.pdf -o output --fallback never

# only some pages / only some formats
python main.py extract your.pdf -o output --pages 1-10,24 --formats json,pdf

# a folder of PDFs
python main.py extract pdfs/ -o output
```

Try it on the included sample:

```bash
python main.py extract balance_sheet_after_correction.pdf -o output -v
```

### Outputs (in `output/`)

| File | Contents |
|---|---|
| `<name>.json` | complete result: every table, every cell with row, column, spans, text and box, plus title, unit, reliability and review flag |
| `<name>_tables.pdf` | printable report: every table drawn as a table; flagged tables show the reasons and the source region |
| `<name>.html` | all tables with merged cells, viewable in a browser |
| `<name>_pNNN_tNN.csv` | one CSV per table (UTF-8 with BOM, opens in Excel) |

### Checking the totals (no ground truth needed)

```bash
python main.py audit output/your.json
```

This checks that every "Total" row equals the rows above it. A mismatch is a pointer for review, not
proof of an error.

### Measuring accuracy on your own PDFs

```bash
# 1. draft ground truth for a few randomly chosen pages
python main.py annotate your.pdf --pages 4,17,23 -o ground_truth_new
# 2. open ground_truth_new/your.json, correct every cell by hand against the PDF,
#    then set "verified": true
# 3. score the extraction against it
python main.py evaluate --gt ground_truth_new --pred output
```

---

## How it works

```
PDF page
 -> 1 Inspection        words + size/weight/colour/direction, vector rules and shading;
                        sideways pages read upright; running headers/footers removed
 -> 2 Two finders       (a) ruled lattices (PyMuPDF line strategy, genuine lattices only)
                        (b) borderless engine: alignment, whitespace, hairlines, shading
 -> 3 Reconcile         one reading per table region (coverage >= 60%)
 -> 4 Reliability gate  nine named checks; failing tables carry readable reasons
 -> 5 OCR fallback      PP-StructureV3, only for scans, image pages and failed checks
 -> 6 Canonical JSON    -> CSV, review HTML, printable PDF report
      evaluate | audit  separate quality tools, never part of the extraction path
```

## Project structure

```
main.py                 entry point (runs the CLI)
pdftable/
  cli.py                commands: extract | evaluate | audit | annotate
  pipeline.py           page routing, process pool, OCR stage, coordinate mapping
  inspection.py         page in its reading frame: words + style, furniture, rules, fills, separators
  textgrid.py           borderless table engine
  native.py             ruled-lattice finder, lattice test, reconciliation
  reconstruct.py        cell boxes -> grid with spans; text from words; headers above ruled grids
  reliability.py        the reliability checks and thresholds
  visual.py             PP-StructureV3 OCR adapter
  models.py             Cell / Table / PageResult / DocumentResult and JSON I/O
  export.py             JSON, review HTML, per-table CSV
  report.py             printable PDF report
  evaluate.py           correctness metrics (detection, relationships, GriTS, exact cells, strict)
  audit.py              arithmetic self-check of printed totals
tests/                  39 tests: synthetic PDFs, metrics, report, real-page regressions
ground_truth/           hand-verified reference tables (development + blind held-out splits)
balance_sheet_after_correction.pdf   sample input
```

## Tests

```bash
python -m pytest -q                          # 37 tests (the annual-report regressions skip if its PDF is absent)
PDFTABLE_TEST_VISUAL=1 python -m pytest -q   # + 2 tests of the OCR fallback
```
