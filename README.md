# pdftable - hybrid PDF-to-table parser

Converts each page of a PDF into correctly structured tables, and measures how
correct they are.  Local, command-line Python; no server, no cloud calls.

```
PDF page
  -> inspection     words + style, vector rules, fills; rotated pages read upright;
                    running headers / footers removed
  -> native         PyMuPDF line strategy (ruled lattices)
                    + borderless engine (alignment, whitespace, hairline rules)
  -> reliability    geometric + semantic checks (is it a table? are its columns coherent?)
  -> visual         PaddleOCR PP-StructureV3, only for scanned / image pages and failed checks
  -> canonical JSON -> HTML / CSV / PDF report
  -> evaluate       against hand-verified ground truth;  audit: totals must add up
```

## What "correct" means here

A table is correct when every value keeps its relationship to its **row label**
and its **column header** - `Food delivery | FY24 -> 7,792` - with the right
merged headers, blank cells, title and unit, and nothing from the surrounding
page mixed in.  Finding the right words somewhere on the page is not enough.
Correctness is measured five ways, deliberately not collapsed into one number:

| Measure | What it checks |
|---|---|
| **Detection** P / R / IoU | every table found, no prose / chart / page frame reported as a table |
| **Relationships** P / R / F1 | each body value as a triple *(row label, column-header path, value)*, pooled per page; a value under the wrong header or beside the wrong label is wrong, however it happened |
| **GriTS** Top / Con | grid topology and cell content (Smock et al. 2023), tolerant to small shifts |
| **Exact cells**, CER | cell text exactly as printed (only whitespace normalised) |
| **Strict table accuracy** | rows, columns, header rows, every span and every cell exact, unit and title kept |

Ground truth (`ground_truth/*.json`) is transcribed by hand from the rendered
page and its text layer.  Pages are split into:

* `dev` - pages used while developing the parser (drafted from parser output, then verified cell by cell);
* `heldout_a`, `heldout_b` - pages drawn **at random with a fixed seed** and transcribed **blind**,
  before the parser's output on them was looked at.  Each held-out set was scored once blind;
  later fixes were checked against it, so only the first score is a true held-out number.

`audit` adds a check that needs no ground truth and scales to every page:
"Total" rows must equal the sum of the rows above them (within rounding).

## Results on `file.pdf` (Zomato annual report FY24, 371 pages)

Native path, `--fallback never` (the fallback does not touch the annotated pages):

| | Detection P / R | Relations F1 | GriTS Top / Con | Exact cells | Strict |
|---|---|---|---|---|---|
| previous parser (49-min run, `output_previous/`) | 0.71 / 0.47 | 0.02 | 0.77 / 0.69 | 0.34 | 0 / 43 |
| this parser, all 29 annotated pages | **1.00 / 0.88** | **0.985** | 0.99 / 0.99 | 0.995 | 37 / 43 |
| - `heldout_b`, **first blind run** | 1.00 / 0.73 | 0.80 | 0.97 / 0.97 | 0.89 | 8 / 15 |
| - `heldout_a`, first blind run | 0.83 / 0.71 | 0.84 | 0.95 / 0.93 | 0.79 | 6 / 14 |

The blind runs are the honest estimate for unseen pages of this kind of
document; the remaining misses are listed under *Limitations*.  Arithmetic audit
over the whole document: 342 of 370 checkable totals add up (92.4%; previous
parser 174 / 206); a sample of the rest are audit-rule limits (net-of columns,
totals continuing from the previous page), not extraction errors.  Whole
document: 358 tables in about 6 s on the native path (previously 266 tables in
49 min); 20 image-heavy pages go to the visual fallback (about 20 s each; they
contain no tables), so the default run takes about 5 minutes.  The supplied Japanese balance sheet stays at 100% strict.

## Using it on a new PDF

Nothing in the parser is specific to one document: it decides from geometry and
typography (alignment, whitespace, rules, shading, font size / weight, text
direction).  What is *measured* is this annual report; a new kind of document
should be checked the same way before trusting its numbers.

```bash
cd ~/BT2024094/UrAI && source .venv/bin/activate

# 1. extract (copy the PDF into the folder or give any path; folders are searched recursively)
python main.py extract /path/to/new.pdf -o output_new -v
#    -> output_new/new.json, new.html, new_pNNN_tNN.csv, new_tables.pdf
#    quick look first:  --pages 1-10   |   no OCR (fast):  --fallback never

# 2. look: open new_tables.pdf (one table per page, NEEDS REVIEW banners) or new.html,
#    read the per-page lines printed by -v, and check the totals
python main.py audit output_new/new.json

# 3. measure: pick pages at random, draft their ground truth, correct it by hand against the
#    PDF (rows, header_rows, spans, title, unit), set "verified": true, then score
python main.py annotate /path/to/new.pdf --pages 4,17,23,58 -o ground_truth_new
#    ... edit ground_truth_new/new.json by hand ...
python main.py extract /path/to/new.pdf -o output_new --formats json
python main.py evaluate --gt ground_truth_new --pred output_new
```

What to expect on other PDFs:

| Kind of PDF | Path | Expectation |
|---|---|---|
| digital, ruled tables | line strategy | strong (Japanese sample: 100% strict) |
| digital, borderless / hairline tables (reports, statements, filings) | borderless engine | strong on this report family (F1 0.80 blind → 0.98 tuned); check a sample |
| tables printed rotated, two-column pages | reading frame, gutter split | handled |
| scanned / image-only pages | PaddleOCR PP-StructureV3 | grid usually right; OCR misreads characters, neighbouring tables can merge; ~20-40 s per page on CPU |
| designed layouts (HTML-to-PDF reports, KPI tiles, infographics, code listings) | borderless engine | weaker: on an unseen HTML-made report (the earlier 6-page project report, since replaced), 3 of 6 tables came out right, 3 were missed or broken (a table split by a page break, a grid cut by inline-code shading) and a code listing was reported as a table |
| right-to-left scripts, handwriting, forms | - | not handled |

## Layout

```
pdftable/
  inspection.py  page inspection: words with font size / bold / colour, vector rules and fills,
                 reading frame for rotated pages, running header / footer removal
  textgrid.py    borderless engine: lines -> phrases -> table regions -> columns (whitespace
                 projection) -> rows (hairline rules or line logic) -> multi-level header
                 (shaded cells, stacked lines, centred group headings) -> title / unit
  native.py      PyMuPDF line strategy for real lattices + the borderless engine, reconciled
  reliability.py geometric and semantic checks, each with a threshold and a readable reason
  visual.py      PP-StructureV3 fallback (scanned / image pages, failed checks)
  reconstruct.py lattice cell boxes -> grid with spans; text from word coordinates
  models.py      canonical Cell / Table / PageResult / DocumentResult (JSON)
  export.py      JSON, HTML (rowspan / colspan), CSV, PDF report
  report.py      PDF report layout (real tables, wrapping, repeated headers, review banners)
  pipeline.py    page routing, process pool, visual stage, frame -> page coordinates
  evaluate.py    correctness metrics against ground truth, per split
  audit.py       arithmetic self-consistency of totals
  cli.py         extract | evaluate | audit | annotate
```

## Install and run

```bash
cd ~/BT2024094/UrAI && source .venv/bin/activate
pip install -r requirements.txt            # fast path only (PyMuPDF, numpy)
pip install -r requirements-visual.txt     # + visual fallback (PaddleOCR)
pip install -r requirements-dev.txt        # + pytest

python main.py extract file.pdf -o output -v               # JSON, HTML, CSV, PDF report
python main.py extract file.pdf -o output --fallback never # native only (~6 s for 371 pages)
python main.py extract file.pdf -o output --pages 24,199,263-265 --formats json,pdf
python main.py evaluate --gt ground_truth --pred output    # per split, with failures listed
python main.py audit output/file.json                      # do the totals add up?
python main.py annotate some.pdf --pages 5,9 -o drafts     # ground-truth draft to correct by hand
python -m pytest -q
```

Performance: the native path runs in a process pool (pages handed out in small tasks, each worker
opens the PDF once); PyMuPDF's ruled-table finder - three quarters of the native time when
profiled - is skipped on pages without vertical separators (no lattice is possible there) and
without its per-table header guess (only cell boxes are used); vector graphics are read once per
page.  Every optimisation was checked to give byte-identical tables on all four PDFs (1,205
tables).  Annual report: 19.5 s -> about 6 s native, about 9 s with every export.

PaddleOCR's start-up messages (glog lines, "No ccache found", "Creating model ...") are hidden;
run with `PDFTABLE_MODEL_LOGS=1 python main.py extract ...` to see them when debugging the
visual fallback.

Outputs: `output/<name>.json` (canonical), `<name>.html`, one CSV per table
(`<name>_pNNN_tNN.csv`, UTF-8 with BOM) and `<name>_tables.pdf` - a cover page
with a linked index, then every table drawn as a table: cell boxes, column
widths from content, wrapped text, merged cells, shaded header rows repeated on
continuation pages, portrait or landscape as needed, CJK / ₹ / △ / superscripts
printed, and for tables flagged `needs_review` the reasons plus a picture of the
source region.  The report renders the extraction faithfully; it does not hide
or repair errors.

## Canonical JSON

```json
{"row": 2, "col": 1, "text": "2,002,429", "rowspan": 1, "colspan": 1,
 "bbox": [356.9, 172.3, 397.5, 181.2], "source": "native_pdf_text", "is_header": false}
```

Cell text is kept as printed (`△2,979`, `－`, `(1,015)`, footnote marks as
superscripts: `78,72,500¹`); nothing is translated or normalised.  A table also
carries `header_rows`, `title`, `unit`, `caption`, `method`
(`native:borderless`, `native:lines`, `visual:pp-structurev3`), `rotation`
(90 / 270 for tables printed sideways; coordinates are always on the original
page), `reliability` (scores and reasons) and `needs_review`.

## How the borderless engine decides

* **Rotated pages**: when most text runs vertically, the page is re-imposed upright and
  everything (words, rules, PyMuPDF, OCR rendering) runs on that view.
* **Phrases**: a line splits at gaps clearly wider than its word spacing (justified prose
  stays whole); two adjacent numbers are always two cells; superscript footnote marks stay
  with their value.
* **Regions**: anchored on value rows (numbers, nil dashes, dates to the right of a label);
  broken by prose, headings, unit lines, large gaps; a bold header line starts a new table.
  Charts are rejected (bars of different heights next to the "rows"); text-only tables are
  accepted when a styled header sits over rows separated by drawn rules.
* **Columns**: whitespace projection over the body rows.  **Rows**: hairline rules when they
  delimit every row (wrapped labels and centred values then fall into place), otherwise
  line logic (lowercase continuations, values beside wrapped labels).
* **Headers**: shaded header cells give exact extents; otherwise stacked phrases form
  multi-line cells and centred group headings span the columns under them; levels and
  rowspans follow from what sits above what.

## Limitations

* Cells that contain only icons (attendance grids) have no text; those tables are missed
  or flagged.
* Q&A forms whose answers are paragraphs spanning many columns, and tables whose cells are
  long paragraphs (auditor's key audit matters), are not detected as tables.
* Sub-rows stacked inside one cell (`(+) 5` / `(-) 5` against two fair-value changes)
  come out as separate rows without the row label repeated.
* Charts are recognised and skipped, not converted to tables.  Designed infographic pages
  (KPI tiles, business cards) may give partial tables, usually flagged.
* Text-only tables need drawn rules between their rows.
* The audit is a heuristic signal; the ground truth covers 43 tables on 29 pages of one
  document - other document families need their own annotated sample before their accuracy
  can be claimed.
* The visual fallback costs about 20 s per page on CPU.

## Documents

* `Project_Report.pdf` - the project report (7 pages): problem, definition of correctness,
  architecture, design decisions with evidence, results, limitations.  Source: `report/report1/` (`build.sh`).
* `Architecture_Guide.pdf` - how the code works, module by module, with a worked example.
  Source: `report/report2/` (`build.sh`).
* `Presentation_1_Business.pptx` (problem statement, business view) and
  `Presentation_2_Technical.pptx` (technical deep dive), built by `report/slides/build_decks.js`.
* Charts and diagrams: `report/assets/` (generators in `report/assets/src/`).
