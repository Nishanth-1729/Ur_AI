"""Phase 2 - native extraction (fast path).

Two structure finders run on the page's reading frame (see ``inspection``):

* PyMuPDF's **line strategy** - ruled lattices and shaded grids;
* the **borderless engine** (``textgrid``) - tables defined by alignment,
  whitespace and hairline rules, which is how most report tables are printed.

Where both claim the same region, the line-strategy grid is kept only if it
is a genuine lattice (visible vertical rules between its columns); a grid
that the line finder derived from a page frame, a shaded header band or
horizontal rules alone is replaced by the borderless reading.  Line-strategy
candidates that are structurally not tables (page-sized frames, mostly empty
grids) are discarded rather than sent to the visual fallback.

Only the *structure* (cell boxes) is taken from PyMuPDF - text is rebuilt from
word coordinates in Phase 3.
"""
from __future__ import annotations

import inspect
from contextlib import contextmanager
from dataclasses import dataclass

import pymupdf
import pymupdf.table

from .inspection import PageInfo
from .models import BBox, Table
from .reconstruct import assemble, build_grid
from .reliability import Thresholds, assess
from .textgrid import chart_like, extract_borderless


@dataclass(slots=True)
class NativeResult:
    reliable: list[Table]
    unreliable: list[Table]


def overlap_ratio(a: BBox, b: BBox) -> float:
    """Intersection over the smaller box."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return iw * ih / smaller if smaller > 0 else 0.0


def covers(a: BBox, b: BBox) -> float:
    """Share of ``b``'s area inside ``a``.  A small grid inside a large table (a shaded
    header strip inside the whole table) covers little of it, although it overlaps it fully."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    area = (b[2] - b[0]) * (b[3] - b[1])
    return iw * ih / area if iw > 0 and ih > 0 and area > 0 else 0.0


# newer PyMuPDF gates candidates through its ONNX layout model by default; that costs ~5x the
# finder itself and is not used here (the reliability checks decide)
_FINDER_KW = {"strategy": "lines"}
if "use_layout" in inspect.signature(pymupdf.table.find_tables).parameters:
    _FINDER_KW["use_layout"] = False
# failures that mean "this grid is not a table at all" rather than "a table, read badly"
_NOT_A_TABLE = ("page frame", "sparse grid", "rows empty", "degenerate grid")


@contextmanager
def _cells_only():
    """PyMuPDF's ``Table`` guesses a header row for every table it finds (text extraction and
    pixmap renders, about a quarter of the finder's time).  Only the cell boxes are used here,
    and the guess plays no part in finding them, so it is skipped while the finder runs."""
    table_cls = pymupdf.table.Table
    original = table_cls._get_header
    table_cls._get_header = lambda self, y_tolerance=3: None
    try:
        yield
    finally:
        table_cls._get_header = original


def _candidates(info: PageInfo) -> list[tuple[BBox, list[BBox]]]:
    """Table bbox and cell boxes from PyMuPDF's line strategy.  The finder is the costliest
    step of a page, so it is skipped when the page draws no vertical separator at all: no grid
    can then pass the lattice test (``_is_lattice``)."""
    if not len(info.v_edges):
        return []
    try:
        with _cells_only():
            found = info.page.find_tables(**_FINDER_KW)
    except (RuntimeError, ValueError, IndexError, ZeroDivisionError):
        found = None  # malformed vector graphics can make the finder raise: treat as "no table"
    if found is None:  # the finder also reports its own failures by returning None
        return []
    out = []
    for t in found.tables:
        boxes = [tuple(c) for c in t.cells if c is not None]
        if boxes:
            out.append((tuple(t.bbox), boxes))
    return out


def _build(info: PageInfo, cands, th: Thresholds) -> list[Table]:
    tables: list[Table] = []
    for bbox, boxes in sorted(cands, key=lambda c: (c[0][1], c[0][0])):
        if any(overlap_ratio(bbox, t.bbox) > 0.5 for t in tables):
            continue
        grid = build_grid(boxes)
        if grid is None:
            continue
        # header/caption search must stop at the nearest table above this one
        floor = max([t.bbox[3] for t in tables
                     if t.bbox[3] <= bbox[1] + 1 and min(t.bbox[2], bbox[2]) > max(t.bbox[0], bbox[0])],
                    default=info.rect[1])
        table, stats = assemble(info, grid, "native:lines", floor, drop_thin=0.6 * info.median_word_height)
        table.reliability = assess(table, stats, info.text_quality, th, page_rect=info.rect)
        tables.append(table)
    return tables


def _not_a_table(t: Table) -> bool:
    return any(k in r for r in t.reliability.reasons for k in _NOT_A_TABLE)


def _is_lattice(info: PageInfo, t: Table) -> bool:
    body = [c for c in t.cells if not c.is_header and c.bbox is not None]
    xs = sorted({round(c.bbox[0], 1) for c in body} | {round(c.bbox[2], 1) for c in body})
    return info.has_lattice(t.bbox, xs)


def extract_native(info: PageInfo, th: Thresholds = Thresholds()) -> NativeResult:
    if not info.words:
        return NativeResult([], [])
    # a line-strategy grid is only trusted when it is a real lattice; grids the finder derives
    # from header pills, card backgrounds or horizontal rules alone are left to the borderless
    # engine, which reads those tables from their text
    ruled = [t for t in _build(info, _candidates(info), th)
             if not _not_a_table(t) and not chart_like(info, t.bbox) and _is_lattice(info, t)
             and t.reliability.checks.get("empty_col_ratio", 0) <= th.max_empty_cols]
    borderless = []
    for t, stats in extract_borderless(info):
        t.reliability = assess(t, stats, info.text_quality, th, page_rect=info.rect)
        borderless.append(t)
    tables: list[Table] = []
    for r in ruled:
        over = [b for b in borderless if overlap_ratio(r.bbox, b.bbox) > 0.3]
        # the lattice reading replaces the borderless one only when it is reliable and spans
        # (most of) the same table; a lattice fragment inside a larger table - e.g. a shaded
        # header strip - is dropped in favour of the whole table
        if over and (not r.reliability.reliable or any(covers(r.bbox, b.bbox) < 0.6 for b in over)):
            continue
        borderless = [b for b in borderless if b not in over]
        tables.append(r)
    tables += borderless
    tables.sort(key=lambda t: (t.bbox[1], t.bbox[0]))
    return NativeResult([t for t in tables if t.reliability.reliable],
                        [t for t in tables if not t.reliability.reliable])
