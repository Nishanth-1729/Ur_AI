"""Phase 3 - cell reconstruction and reconciliation.

Turns a predicted structure (cell boxes from PyMuPDF or from the visual model)
into a canonical ``Table``:

* grid lines are recovered from the cell boxes, so merged cells get proper
  rowspan / colspan and uncovered grid positions become explicit empty cells;
* cell text is rebuilt from word coordinates (never from the finder's own
  text, which can garble glyphs on different baselines, e.g. "2 002 429\\n, ,");
* column headers, titles and units printed *above* the detected body are
  located and attached instead of being dropped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .inspection import UNIT_RE, PageInfo, WordSet, group_lines, join_words, phrases
from .models import BBox, Cell, Table

NUMERIC_RE = re.compile(r"^[\s△▲▵\-−–－(（$¥￥€£]*[\d][\d,.\s]*[)）%％]*$")


def cluster(values, tol: float, is_end=None, min_support: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Sorted 1-D clustering: values closer than ``tol`` collapse to their mean.
    Returns (centers, support counts).

    When ``is_end`` (bool per value: box end vs box start) is given, the input is treated
    as noisy model output and an interior boundary must be *two-sided* - some box ends
    there and another starts there - and backed by at least ``min_support`` of the median
    cluster size.  A box edge that merely hugs right-aligned text produces a one-sided
    cluster and is snapped to the real boundary instead of creating a phantom column.
    """
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return values, values
    order = np.argsort(values, kind="stable")
    v = values[order]
    cuts = np.nonzero(np.diff(v) > tol)[0] + 1
    groups = np.split(v, cuts)
    centers = np.array([g.mean() for g in groups])
    n = np.array([len(g) for g in groups])
    if is_end is None or len(centers) <= 2:
        return centers, n
    ends = np.split(np.asarray(is_end, bool)[order], cuts)
    keep = np.array([e.any() and not e.all() for e in ends]) & (n >= min_support * np.median(n))
    keep[[0, -1]] = True
    return centers[keep], n[keep]


def _drop_textless_columns(xs: np.ndarray, support: np.ndarray, words: WordSet, y0: float, y1: float,
                           slack: float = 2.0):
    """Noisy grids only: a column that holds no word *entirely* is a phantom split (a word
    cut by a boundary is evidence against that boundary).  Remove the column's weaker
    interior boundary (fewest supporting box edges) until every column holds a word."""
    m = (words.cy >= y0) & (words.cy <= y1) & (words.cx >= xs[0]) & (words.cx <= xs[-1])
    wx = words.wbox[m][:, [0, 2]]
    if not len(wx):
        return xs
    while len(xs) > 3:
        holds = (wx[:, None, 0] >= xs[None, :-1] - slack) & (wx[:, None, 1] <= xs[None, 1:] + slack)
        empty = np.nonzero(~holds.any(0))[0]
        cand = [i for k in empty for i in (k, k + 1) if 0 < i < len(xs) - 1]
        if not cand:
            break
        drop = min(cand, key=lambda i: support[i])
        xs, support = np.delete(xs, drop), np.delete(support, drop)
    return xs


@dataclass(slots=True)
class Grid:
    xs: np.ndarray  # column boundaries (n_cols + 1)
    ys: np.ndarray  # row boundaries (n_rows + 1)
    cells: list[Cell]
    occ: np.ndarray  # (n_rows, n_cols) -> index into cells

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.ys) - 1, len(self.xs) - 1


def build_grid(boxes: list[BBox], tol: float = 3.0, ytol: float | None = None,
               noisy: bool = False, words: WordSet | None = None) -> Grid | None:
    """Recover a row/column lattice from cell rectangles; spans follow from snapping.
    Exact vector boxes need a small ``tol``.  ``noisy`` model boxes use a larger one,
    two-sided consensus filtering of boundaries (see ``cluster``) and, given ``words``,
    removal of columns that contain no text at all."""
    if not boxes:
        return None
    b = np.asarray(boxes, dtype=np.float64)
    side = np.r_[np.zeros(len(b), bool), np.ones(len(b), bool)] if noisy else None
    xs, xn = cluster(np.concatenate([b[:, 0], b[:, 2]]), tol, side)
    ys, _ = cluster(np.concatenate([b[:, 1], b[:, 3]]), tol if ytol is None else ytol, side)
    if len(xs) < 2 or len(ys) < 2:
        return None
    if noisy and words is not None:
        xs = _drop_textless_columns(xs, xn, words, ys[0], ys[-1])
    nr, nc = len(ys) - 1, len(xs) - 1
    c0 = np.abs(b[:, 0, None] - xs).argmin(1)
    c1 = np.abs(b[:, 2, None] - xs).argmin(1)
    r0 = np.abs(b[:, 1, None] - ys).argmin(1)
    r1 = np.abs(b[:, 3, None] - ys).argmin(1)
    occ = np.full((nr, nc), -1, dtype=np.int64)
    cells: list[Cell] = []
    # larger cells first so a spurious small box cannot punch a hole in a merged cell
    for i in np.argsort(-(c1 - c0) * (r1 - r0), kind="stable"):
        if c1[i] <= c0[i] or r1[i] <= r0[i] or (occ[r0[i]:r1[i], c0[i]:c1[i]] >= 0).any():
            continue
        occ[r0[i]:r1[i], c0[i]:c1[i]] = len(cells)
        cells.append(Cell(int(r0[i]), int(c0[i]), "", int(r1[i] - r0[i]), int(c1[i] - c0[i]),
                          (xs[c0[i]], ys[r0[i]], xs[c1[i]], ys[r1[i]]), "empty"))
    for r, c in zip(*np.nonzero(occ < 0)):  # implicit blank cells keep the grid rectangular
        occ[r, c] = len(cells)
        cells.append(Cell(int(r), int(c), "", 1, 1, (xs[c], ys[r], xs[c + 1], ys[r + 1]), "empty"))
    return Grid(xs, ys, cells, occ)


def drop_thin_blank(grid: Grid, min_size: float) -> Grid:
    """Ruled grids: a row or column that holds no text and is thinner than ``min_size`` (less
    than a line of type) is a drawing artefact - the edge of a zebra stripe, cell padding, a
    double rule - not a row or column.  It is removed and its width / height given to the
    neighbour before it; spans are re-anchored."""
    nr, nc = grid.shape
    blank_r = [not any(grid.cells[i].text for i in np.unique(grid.occ[r])) for r in range(nr)]
    blank_c = [not any(grid.cells[i].text for i in np.unique(grid.occ[:, c])) for c in range(nc)]
    keep_r = [r for r in range(nr) if not (blank_r[r] and grid.ys[r + 1] - grid.ys[r] < min_size)]
    keep_c = [c for c in range(nc) if not (blank_c[c] and grid.xs[c + 1] - grid.xs[c] < min_size)]
    if (len(keep_r), len(keep_c)) == (nr, nc) or not keep_r or not keep_c:
        return grid
    ys = np.asarray([grid.ys[0]] + [grid.ys[r] for r in keep_r[1:]] + [grid.ys[-1]])
    xs = np.asarray([grid.xs[0]] + [grid.xs[c] for c in keep_c[1:]] + [grid.xs[-1]])
    rmap = {r: i for i, r in enumerate(keep_r)}
    cmap = {c: i for i, c in enumerate(keep_c)}
    cells: list[Cell] = []
    occ = np.full((len(keep_r), len(keep_c)), -1, dtype=np.int64)
    for i in np.unique(grid.occ):
        cell = grid.cells[i]
        rs = [rmap[r] for r in range(cell.row, cell.row + cell.rowspan) if r in rmap]
        cs = [cmap[c] for c in range(cell.col, cell.col + cell.colspan) if c in cmap]
        if not rs or not cs:
            continue  # lay entirely in removed (blank) rows / columns
        r0, c0 = min(rs), min(cs)
        cell.row, cell.col, cell.rowspan, cell.colspan = r0, c0, max(rs) - r0 + 1, max(cs) - c0 + 1
        cell.bbox = (float(xs[c0]), float(ys[r0]), float(xs[c0 + cell.colspan]), float(ys[r0 + cell.rowspan]))
        if (occ[r0:r0 + cell.rowspan, c0:c0 + cell.colspan] >= 0).any():
            continue
        occ[r0:r0 + cell.rowspan, c0:c0 + cell.colspan] = len(cells)
        cells.append(cell)
    for r, c in zip(*np.nonzero(occ < 0)):
        occ[r, c] = len(cells)
        cells.append(Cell(int(r), int(c), "", 1, 1, (float(xs[c]), float(ys[r]), float(xs[c + 1]), float(ys[r + 1])), "empty"))
    return Grid(xs, ys, cells, occ)


@dataclass(slots=True)
class FillStats:
    n_words: int = 0
    straddling: int = 0
    mean_score: float | None = None  # OCR confidence of the assigned words


def fill_text(grid: Grid, info: WordSet, source: str = "native_pdf_text") -> FillStats:
    """Assign every word whose centre lies in the grid to its cell (vectorised)."""
    nr, nc = grid.shape
    idx = info.words_in((grid.xs[0], grid.ys[0], grid.xs[-1], grid.ys[-1]))
    stats = FillStats(len(idx))
    if not len(idx):
        return stats
    col = np.clip(np.searchsorted(grid.xs, info.cx[idx], "right") - 1, 0, nc - 1)
    row = np.clip(np.searchsorted(grid.ys, info.cy[idx], "right") - 1, 0, nr - 1)
    owner = grid.occ[row, col]
    cb = np.asarray([c.bbox for c in grid.cells])[owner]
    w = info.wbox[idx]
    # a word sticking out of its cell by more than a third of its width means the
    # column split cuts through text -> structural error signal for reliability
    over = np.maximum(cb[:, 0] - w[:, 0], 0) + np.maximum(w[:, 2] - cb[:, 2], 0)
    stats.straddling = int(np.count_nonzero(over > 0.34 * (w[:, 2] - w[:, 0])))
    order = np.argsort(owner, kind="stable")
    owner_s, idx_s = owner[order], idx[order]
    for cell_i, grp in zip(*_groups(owner_s, idx_s)):
        cell = grid.cells[cell_i]
        cell.text = join_words(info.words, grp.tolist())
        cell.source = source
        if info.scores is not None:
            cell.confidence = float(info.scores[grp].mean())
    if info.scores is not None:
        stats.mean_score = float(info.scores[idx].mean())
    return stats


def _groups(keys: np.ndarray, vals: np.ndarray):
    starts = np.r_[0, np.nonzero(np.diff(keys))[0] + 1]
    return keys[starts], np.split(vals, starts[1:])


def is_numeric(text: str) -> bool:
    return bool(NUMERIC_RE.match(text))


# ---------------------------------------------------------------------------
# Headers / captions located above the table body
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class HeaderBlock:
    rows: list[list[Cell]]  # header rows with row indices starting at 0
    top: float


def _span_cols(x0: float, x1: float, xs: np.ndarray) -> tuple[int, int] | None:
    ov = np.minimum(x1, xs[1:]) - np.maximum(x0, xs[:-1])
    hit = np.nonzero(ov > 0.15 * max(x1 - x0, 1e-6))[0]
    return (int(hit[0]), int(hit[-1]) + 1) if len(hit) else None


def detect_header(info: WordSet, rules: np.ndarray, bbox: BBox, xs: np.ndarray, floor: float,
                  source: str = "native_pdf_text") -> HeaderBlock | None:
    """Find column headings printed above the table body (``bbox``).

    The search region is bounded by the nearest full-width rule above the body if
    there is one (typical for financial statements), otherwise by line spacing.
    ``floor`` is the bottom of any table above, which the search must not cross.
    ``info`` is the native text layer or page-level OCR lines; ``rules`` is (n, 3).
    """
    x0, top, x1, _ = bbox
    width = x1 - x0
    lh = info.median_word_height
    fence = None
    if len(rules):
        full = rules[(rules[:, 2] < top - 2) & (rules[:, 2] > floor)
                     & (np.minimum(rules[:, 1], x1) - np.maximum(rules[:, 0], x0) > 0.85 * width)]
        if len(full):
            fence = float(full[:, 2].max())
    limit = fence if fence is not None else max(floor, top - 6 * lh)
    idx = info.words_in((x0 - 2, limit, x1 + 2, top))
    if not len(idx):
        return None
    lines = group_lines(info.words, idx.tolist())
    # walk upwards from the body; stop at gaps, units, or paragraph-like lines
    picked = []
    edge = top
    for ln in reversed(lines):
        if fence is None and edge - ln.y1 > 1.5 * lh:
            break
        if UNIT_RE.search(ln.text) or (len(ln.idx) > 1 and ln.x1 - ln.x0 > 0.6 * width
                                        and len(phrases(ln, info.words, lh)) == 1):
            break
        picked.append(ln)
        edge = ln.y0
        if len(picked) == 4:
            break
    if not picked:
        return None
    picked.reverse()

    # per line: phrase -> column span
    layouts = []
    for ln in picked:
        lay = []
        for px0, px1, ids in phrases(ln, info.words, lh):
            span = _span_cols(px0, px1, xs)
            if span:
                lay.append((span, ids))
        layouts.append(lay)
    if fence is None and not any(s[0] > 0 for lay in layouts for s, _ in lay):
        return None  # text only over the stub column: more likely a caption than a header

    # consecutive lines with the same single-column layout and no rule between them
    # are one header row with multi-line cells; a colspan phrase starts a new level.
    def signature(lay):
        return tuple(sorted(s for s, _ in lay))

    def rule_between(a, b):
        return len(rules) and bool(((rules[:, 2] > a.y1) & (rules[:, 2] < b.y0)).any())

    groups: list[list[int]] = [[0]]
    for i in range(1, len(picked)):
        prev, cur = layouts[groups[-1][-1]], layouts[i]
        spans_multi = any(s[1] - s[0] > 1 for s, _ in prev + cur)
        if spans_multi or rule_between(picked[i - 1], picked[i]) or not (
                set(signature(cur)) <= set(signature(prev)) or set(signature(prev)) <= set(signature(cur))):
            groups.append([i])
        else:
            groups[-1].append(i)

    header_top = fence if fence is not None else picked[0].y0 - 1
    nc = len(xs) - 1
    rows: list[list[Cell]] = []
    bounds = [header_top] + [picked[g[-1]].y1 + (picked[g[-1] + 1].y0 - picked[g[-1]].y1) / 2
                             for g in groups[:-1]] + [top]
    for r, g in enumerate(groups):
        ids_by_span: dict[tuple[int, int], list[int]] = {}
        for li in g:
            for span, ids in layouts[li]:
                ids_by_span.setdefault(span, []).extend(ids)
        row: list[Cell] = []
        covered = np.zeros(nc, bool)
        for (c0, c1), ids in sorted(ids_by_span.items()):
            if covered[c0:c1].any():  # overlapping phrases: widen into the existing cell
                continue
            covered[c0:c1] = True
            row.append(Cell(r, c0, join_words(info.words, ids), 1, c1 - c0,
                            (xs[c0], bounds[r], xs[c1], bounds[r + 1]), source, True,
                            None if info.scores is None else float(info.scores[ids].mean())))
        for c in np.nonzero(~covered)[0]:
            row.append(Cell(r, int(c), "", 1, 1, (xs[c], bounds[r], xs[c + 1], bounds[r + 1]), "empty", True))
        row.sort(key=lambda c: c.col)
        rows.append(row)

    # stub-column heading printed once across a multi-level header -> rowspan
    if len(rows) > 1:
        for c in range(nc):
            col_cells = [next((x for x in row if x.col == c and x.colspan == 1), None) for row in rows]
            if all(col_cells):
                filled = [x for x in col_cells if x.text]
                if len(filled) <= 1:
                    keep = filled[0] if filled else col_cells[0]
                    keep.row, keep.rowspan = 0, len(rows)
                    keep.bbox = (keep.bbox[0], bounds[0], keep.bbox[2], bounds[-1])
                    for row in rows:
                        row[:] = [x for x in row if not (x.col == c and x.colspan == 1)]
                    rows[0].append(keep)
                    rows[0].sort(key=lambda x: x.col)
    return HeaderBlock(rows, header_top)


def detect_caption(info: WordSet, bbox: BBox, top: float, floor: float, max_lines: int = 3):
    """Title / unit lines directly above the table (e.g. "（単位：千円）")."""
    x0, _, x1, _ = bbox
    lh = info.median_word_height
    idx = info.words_in((x0 - 2, max(floor, top - (max_lines + 1) * 2.5 * lh), x1 + 2, top))
    if not len(idx):
        return [], None, None
    lines = group_lines(info.words, idx.tolist())
    out = []
    edge = top
    for ln in reversed(lines):
        if ln.y1 > top + 1 or edge - ln.y1 > 1.5 * lh:
            break
        out.append(ln.text)
        edge = ln.y0
        if len(out) == max_lines:
            break
    out.reverse()
    unit = next((t for t in reversed(out) if UNIT_RE.search(t)), None)
    title = next((t for t in reversed(out) if t != unit), None)
    return out, title, unit


def internal_header_rows(cells: list[Cell], n_rows: int, n_cols: int) -> int:
    """Use the first grid row as header when at least half of it is non-numeric text
    sitting over columns that hold numbers further down (an empty stub cell is allowed)."""
    if n_rows < 3:
        return 0
    first = [c for c in cells if c.row == 0 and c.text]
    if 2 * sum(c.colspan for c in first) < n_cols or any(is_numeric(c.text) for c in first):
        return 0
    head_cols = {k for c in first for k in range(c.col, c.col + c.colspan)}
    numeric_below = any(c.row > 0 and c.col in head_cols and c.text and is_numeric(c.text) for c in cells)
    return 1 if numeric_below else 0


_NO_RULES = np.zeros((0, 3))


def assemble(info: PageInfo, grid: Grid, method: str, floor: float,
             words: WordSet | None = None, source: str = "native_pdf_text",
             context: WordSet | None | bool = True, drop_thin: float = 0.0) -> tuple[Table, FillStats]:
    """Body grid + text + external header + caption -> canonical Table.

    ``words`` fills the cells: the page's text layer by default, OCR words for scans.
    ``context`` is searched for headers / title / unit above the body: ``True`` means the
    text layer (if trustworthy), a WordSet (e.g. page-level OCR lines) is used as is,
    ``None`` / ``False`` disables the search.  ``drop_thin`` removes blank rows / columns
    thinner than that (drawing artefacts of ruled grids, see ``drop_thin_blank``).
    """
    stats = fill_text(grid, info if words is None else words, source)
    if drop_thin:
        grid = drop_thin_blank(grid, drop_thin)
    body_bbox = tuple(float(v) for v in (grid.xs[0], grid.ys[0], grid.xs[-1], grid.ys[-1]))
    if context is True:
        context = info if info.text_layer_ok else None
    rules = info.h_rules if context is info else _NO_RULES
    header = detect_header(context, rules, body_bbox, grid.xs, floor, source) if context else None
    nh = len(header.rows) if header else 0
    cells = [c for row in header.rows for c in row] if header else []
    for c in grid.cells:
        c.row += nh
    cells.extend(grid.cells)
    n_rows, n_cols = grid.shape[0] + nh, grid.shape[1]
    if not nh:
        nh = internal_header_rows(cells, n_rows, n_cols)
        for c in cells:
            c.is_header = c.row < nh
    top = header.top if header else body_bbox[1]
    caption, title, unit = detect_caption(context, body_bbox, top, floor) if context else ([], None, None)
    cells.sort(key=lambda c: (c.row, c.col))
    table = Table(page=info.number, index=0, bbox=(body_bbox[0], top, body_bbox[2], body_bbox[3]),
                  n_rows=n_rows, n_cols=n_cols, cells=cells, header_rows=nh, method=method,
                  caption=caption, title=title, unit=unit)
    return table, stats
