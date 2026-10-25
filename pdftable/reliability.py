"""Phase 2 / 5 - reliability checks.

A native result is only kept when every check passes; otherwise the page is
routed to the visual fallback (or flagged ``needs_review`` when no fallback is
available).  Each check is a value in [0, 1] with an explicit threshold so the
routing decision is explainable in the output JSON.

Two kinds of checks:

* **geometric** - words cut by cell boundaries, fill ratio, empty columns, text
  layer / OCR quality.  These catch a wrong split of a real table.
* **semantic** - is this a table at all, and are its columns coherent?  A grid
  laid over running text or a page border can be geometrically clean (every
  word inside one cell) and still be wrong.  These checks look at what the cells
  *contain*: prose sentences in body cells, several numbers packed into one cell
  (two columns merged), mostly-empty rows, a "table" covering the whole page.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .models import BBox, Reliability, Table
from .reconstruct import FillStats

_TOKEN_VALUE = re.compile(r"^[(（\[]?[-–−+△▲]?[₹$€£¥]?\d[\d,.]*[%％]?[)）\]]?[*#⁰¹²³⁴⁵⁶⁷⁸⁹⁽⁾]*$")
_SENTENCE = re.compile(r"[a-z]{3,}[.;:]\s+[A-Z]|[a-z]{3,}\.$")
_MONTH = re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[.,]?$", re.I)


@dataclass(frozen=True, slots=True)
class Thresholds:
    max_straddle: float = 0.05  # words cut by a column/row boundary
    min_fill: float = 0.20  # non-empty cells / all cells
    max_empty_cols: float = 0.34  # fully empty columns / columns
    min_text_quality: float = 0.98  # clean characters in the text layer
    min_ocr_confidence: float = 0.80  # mean recognition score when text comes from OCR
    max_prose_cells: float = 0.15  # body cells that read like running text
    max_packed_cells: float = 0.20  # body cells holding several separate numbers
    max_empty_rows: float = 0.35  # body rows with no text at all
    max_page_cover: float = 0.85  # table bbox / page area


def is_prose(text: str) -> bool:
    words = text.split()
    return len(words) >= 14 or (len(words) >= 8 and bool(_SENTENCE.search(text)))


def packed_values(text: str) -> int:
    """Number of separate numeric tokens in a cell made (mostly) of numbers
    ("8,55,35,09,770 855" -> 2); a label that mentions numbers ("6,000,000 shares of INR 1
    each") is not a packed cell and gives 0."""
    toks = text.split()
    if any(_MONTH.match(w) for w in toks):  # a date ("April 25, 2023") is one value
        return 0
    n = sum(1 for w in toks if _TOKEN_VALUE.match(w))
    return n if toks and n >= 0.6 * len(toks) else 0


def assess(table: Table, stats: FillStats, text_quality: float, th: Thresholds = Thresholds(),
           ocr: bool = False, page_rect: Optional[BBox] = None) -> Reliability:
    body = [c for c in table.cells if not c.is_header]
    n_body_rows = table.n_rows - table.header_rows
    checks: dict[str, float] = {}
    reasons: list[str] = []

    shape_ok = n_body_rows >= 1 and table.n_cols >= 2 and (n_body_rows >= 2 or table.header_rows >= 1)
    checks["shape"] = 1.0 if shape_ok else 0.0
    if not shape_ok:
        reasons.append(f"degenerate grid {table.n_rows}x{table.n_cols}")

    straddle = stats.straddling / stats.n_words if stats.n_words else 0.0
    checks["straddle_ratio"] = straddle
    if straddle > th.max_straddle:
        reasons.append(f"{stats.straddling}/{stats.n_words} words cut by cell boundaries")

    filled = [c for c in body if c.text]
    fill = len(filled) / len(body) if body else 0.0
    checks["fill_ratio"] = fill
    if fill < th.min_fill:
        reasons.append(f"sparse grid (fill {fill:.2f})")

    used_cols = {k for c in filled for k in range(c.col, c.col + c.colspan)}
    empty_cols = 1 - len(used_cols) / table.n_cols if table.n_cols else 1.0
    checks["empty_col_ratio"] = empty_cols
    if empty_cols > th.max_empty_cols:
        reasons.append(f"{empty_cols:.0%} of columns empty")

    # ---- semantic checks ----------------------------------------------------
    # running text mixed into a table: a sentence in a column that otherwise holds numbers, or a
    # paragraph swallowed as a row of its own (one long cell, the rest empty).  A remarks column
    # of sentences is legitimate table content and is not counted.
    by_col: dict[int, list[str]] = {}
    for c in filled:
        by_col.setdefault(c.col, []).append(c.text)
    numeric_cols = {k for k, v in by_col.items() if k > 0 and len(v) >= 2
                    and sum(packed_values(x) >= 1 or _TOKEN_VALUE.match(x.strip() or "x") is not None
                            for x in v) >= 0.6 * len(v)}
    row_cells: dict[int, list] = {}
    for c in filled:
        row_cells.setdefault(c.row, []).append(c)
    contaminated = sum(1 for c in filled if c.col in numeric_cols and is_prose(c.text))
    contaminated += sum(1 for cs in row_cells.values() if len(cs) == 1 and len(cs[0].text.split()) >= 15
                        and table.n_cols > 1)
    prose = contaminated / len(filled) if filled else 0.0
    checks["prose_cells"] = prose
    if prose > th.max_prose_cells:
        reasons.append(f"{prose:.0%} of cells are running text, not table cells")

    packed = sum(1 for c in filled if packed_values(c.text) >= 2) / len(filled) if filled else 0.0
    checks["packed_cells"] = packed
    if packed > th.max_packed_cells:
        reasons.append(f"{packed:.0%} of cells hold several numbers (columns merged)")

    rows_with_text = {r for c in filled for r in range(c.row, c.row + c.rowspan)}
    empty_rows = 1 - len(rows_with_text & set(range(table.header_rows, table.n_rows))) / n_body_rows \
        if n_body_rows > 0 else 1.0
    checks["empty_row_ratio"] = empty_rows
    if empty_rows > th.max_empty_rows:
        reasons.append(f"{empty_rows:.0%} of rows empty")

    if page_rect is not None and table.bbox is not None:
        pa = (page_rect[2] - page_rect[0]) * (page_rect[3] - page_rect[1])
        b = table.bbox
        cover = (b[2] - b[0]) * (b[3] - b[1]) / pa if pa else 0.0
        checks["page_cover"] = cover
        if cover > th.max_page_cover:
            reasons.append(f"grid covers {cover:.0%} of the page (page frame, not a table)")

    # text_quality is the clean-character ratio of the PDF text layer, or the mean
    # recognition score when the text comes from OCR
    key, limit = ("ocr_confidence", th.min_ocr_confidence) if ocr else ("text_quality", th.min_text_quality)
    checks[key] = text_quality
    if text_quality < limit:
        reasons.append(f"{key.replace('_', ' ')} {text_quality:.3f} < {limit}")

    score = min(
        checks["shape"],
        1.0 - min(1.0, straddle / (2 * th.max_straddle)) * 0.5 if straddle <= th.max_straddle else 0.4,
        min(1.0, fill / th.min_fill),
        1.0 if empty_cols <= th.max_empty_cols else 0.4,
        1.0 - prose if prose <= th.max_prose_cells else 0.3,
        1.0 - 0.5 * packed if packed <= th.max_packed_cells else 0.3,
        1.0 if empty_rows <= th.max_empty_rows else 0.4,
        1.0 if checks.get("page_cover", 0.0) <= th.max_page_cover else 0.0,
        min(1.0, text_quality / limit),
    )
    return Reliability(score=score, reliable=not reasons, checks=checks, reasons=reasons)
