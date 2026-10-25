"""Phase 2a/3 - borderless (unruled or hairline-ruled) table engine.

PyMuPDF's line strategy needs a lattice of vector rules, and its text strategy
builds a grid from *every* aligned word on the page, which turns multi-column
report pages, titles and running headers into fake tables.  Most tables in
annual reports, filings and statements have neither: columns are defined by
alignment and whitespace, rows by line spacing or thin hairlines.  This module
reconstructs such tables from the word layer, in five explainable steps:

1. **Lines and phrases.**  Words are grouped into visual lines; a line is split
   into phrases at gaps clearly wider than a word space (adapting to justified
   prose), and two adjacent numeric tokens are always separate cells
   ("257 97.72%").
2. **Regions.**  A table is a run of lines anchored by *value rows* (a value
   phrase to the right of other content).  The run is broken by prose, headings
   (larger or bold-larger type), unit lines, large gaps and sentences that run
   into the value columns; a bold multi-phrase line after value rows starts a
   new table.
3. **Columns** come from a whitespace projection of the body rows (a column is
   an x-range covered by phrases; a gap crossed by only a few spanning phrases
   is still a gap).
4. **Rows** come from visible horizontal rules when the table draws one
   between most of its lines (multi-line cells then fall out naturally);
   otherwise from the lines, joining wrapped labels and vertically centred
   values to their row.
5. **Headers** above the first value row become a multi-level header.  Shaded
   header cells, when drawn, give each cell's exact extent; otherwise phrases
   that stack one-to-one form a multi-line cell and a phrase centred over two or
   more cells below it spans their columns.  Levels follow from which cells sit
   above which (by column overlap); a cell with nothing below it extends down to
   the body (rowspan).  Title and unit are the heading / unit lines just above.

Every decision uses geometry or typography visible on the page; nothing is
specific to one document.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .inspection import UNIT_RE, PageInfo, WordSet, group_lines, join_words
from .models import BBox, Cell, Table
from .reconstruct import FillStats, internal_header_rows

# a printed value: number (with grouping, sign, brackets, %, footnote marks), a nil dash, NA,
# or a date ("31-Mar-24", "10/08/2022", "2023-24", "Apr-23")
_MON = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*"
VALUE_RE = re.compile(
    r"^[(（\[]?[-–−+△▲]?[₹$€£¥]?\s?\d[\d,.]*[%％]?[)）\]]?[*#†‡^⁰¹²³⁴⁵⁶⁷⁸⁹⁽⁾⁺⁻]*$"
    r"|^[(（]?[-–—−－]{1,2}[)）]?[*#]*$|^(NA|N\.A\.|N/A|n/a|Nil|nil|NIL)[*]*$"
    rf"|^\d{{1,2}}[-/ ]{_MON}[-/ ]\d{{2,4}}$|^\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{2,4}}$|^{_MON}[-/ ]\d{{2,4}}$")
_LOWER_START = re.compile(r"^[a-z(]")
_PERIOD = re.compile(r"^(FY\s?)?(19|20)\d{2}([-–/](19|20)?\d{2})?[*]*$|^FY\s?\d{2}([-–/]\d{2})?$")
_ENUM = re.compile(r"^(\(?[a-zA-Z0-9]{1,3}[.)]|\(?[ivxIVX]{1,4}[.)])$")


def is_value(text: str) -> bool:
    return bool(VALUE_RE.match(text.strip()))


# ---------------------------------------------------------------------------
# 1. lines and phrases
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Phrase:
    ids: list[int]
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    value: bool
    bold: bool
    size: float

    @property
    def n_words(self) -> int:
        return len(self.ids)

    @property
    def width(self) -> float:
        return self.x1 - self.x0


@dataclass(slots=True)
class TLine:
    phrases: list[Phrase]
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    kind: str = ""
    side: str = ""  # on a two-column page: "L" / "R" of the gutter, "F" across it

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def text(self) -> str:
        return " ".join(p.text for p in self.phrases)

    @property
    def bold(self) -> bool:
        return all(p.bold for p in self.phrases)

    def values_right(self) -> list[Phrase]:
        """Value phrases that sit to the right of other content (or 2+ values).  An
        enumerator followed by text ("6. The Secretarial ...", "1. Permanent") is a label."""
        ph = self.phrases
        vals = [p for i, p in enumerate(ph) if p.value and not (
            _ENUM.match(p.text) and i + 1 < len(ph) and not ph[i + 1].value)]
        if len(vals) >= 2:
            return vals
        return [p for p in vals if p is not ph[0]]


def _make_phrase(info: WordSet, ids: list[int]) -> Phrase:
    w = info.wbox[ids]
    text = " ".join(info.words[i][4] for i in ids)
    return Phrase(ids, float(w[:, 0].min()), float(w[:, 1].min()), float(w[:, 2].max()), float(w[:, 3].max()),
                  text, is_value(text), bool(info.wbold[ids].mean() > 0.5), float(np.median(info.wsize[ids])))


def split_phrases(info: WordSet, ids: list[int]) -> list[Phrase]:
    """Split one visual line (word ids, any order) into phrases."""
    ids = sorted(ids, key=lambda i: info.wbox[i, 0])
    size = float(np.median(info.wsize[ids]))
    toks = [info.words[i][4] for i in ids]
    gaps = [info.wbox[ids[k + 1], 0] - info.wbox[ids[k], 2] for k in range(len(ids) - 1)]
    thr = 0.45 * size  # a word space is ~0.2 em, justified text stretches it to ~0.45 em
    if len(gaps) >= 3 and max(gaps) <= 1.0 * size and max(gaps) <= 2.2 * min(gaps):
        thr = size  # uniformly stretched word spaces: one justified line of prose
    elif len(gaps) >= 3 and min(gaps) < 0.5 * size:  # adapt to the line's own word spacing
        thr = min(max(thr, 2.0 * min(gaps)), 1.6 * size)
    groups = [[ids[0]]] if ids else []
    for k, g in enumerate(gaps):
        a, b = toks[k], toks[k + 1]
        numeric_pair = is_value(a) and is_value(b) and not a.endswith(",") and g > 0.1 * size
        if g > thr or numeric_pair:
            groups.append([ids[k + 1]])
        else:
            groups[-1].append(ids[k + 1])
    return [_make_phrase(info, g) for g in groups]


def _tline(info: WordSet, ids: list[int], side: str = "") -> TLine:
    ph = split_phrases(info, ids)
    w = info.wbox[ids]
    return TLine(ph, float(w[:, 0].min()), float(w[:, 1].min()), float(w[:, 2].max()), float(w[:, 3].max()),
                 float(np.median([p.size for p in ph])), side=side)


def find_gutter(info: WordSet, idx) -> Optional[float]:
    """x of the gutter of a two-column text layout, or None.  Evidence: several lines that hold
    justified prose on both sides of the same gap - the left part starts at the left margin
    and runs up to the gap, the right part starts just after it, each with 4+ words.  A table
    whose columns hold descriptions does not produce this (its left cells end raggedly)."""
    lines = group_lines(info.words, list(idx))
    if len(lines) < 8:
        return None
    left = min(l.x0 for l in lines)
    width = max(l.x1 for l in lines) - left
    cands = []
    for ln in lines:
        ws = ln.idx
        for k in range(len(ws) - 1):
            a, b = info.wbox[ws[k]], info.wbox[ws[k + 1]]
            size = float(info.wsize[ws[k]])
            if b[0] - a[2] < 0.9 * size or k + 1 < 4 or len(ws) - k - 1 < 4:
                continue
            words_right = [info.words[i][4] for i in ws[k + 1:k + 5]]
            prose_right = sum(any(ch.isalpha() for ch in w) for w in words_right) >= 3
            if prose_right and info.wbox[ws[0], 0] <= left + 2 * size:
                cands.append(((a[2] + b[0]) / 2, a[2], b[0]))
    if len(cands) < 4:
        return None
    centers = np.sort(np.asarray([c[0] for c in cands]))
    best = max(centers, key=lambda x: int(np.sum(np.abs(centers - x) <= 6)))
    near = [c for c in cands if abs(c[0] - best) <= 6]
    if len(near) < 4 or not 0.38 <= (best - left) / width <= 0.62:
        return None  # a page gutter sits near the middle of the text block
    # the left column is justified: most of its lines end at (nearly) the same x
    ends = np.asarray([c[1] for c in near])
    if np.sum(ends >= ends.max() - 3) < max(4, 0.6 * len(near)):
        return None
    return float(np.median([c[0] for c in near]))


def _column_prose(info: WordSet, ids: list[int], x_from: float, x_to: float) -> bool:
    """Words that read as a line of column text: one run (no column-sized gap), 4+ words
    mostly alphabetic, filling the column from ``x_from`` to ``x_to``."""
    if len(ids) < 4:
        return False
    size = float(np.median(info.wsize[ids]))
    w = info.wbox[ids]
    if (w[1:, 0] - w[:-1, 2]).max(initial=0) > 0.9 * size:
        return False
    if sum(any(ch.isalpha() for ch in info.words[i][4]) for i in ids[:4]) < 3:
        return False
    return w[0, 0] <= x_from + 2 * size and w[-1, 2] >= x_to - 2 * size


def build_lines(info: WordSet, idx, gutter: Optional[float] = None) -> list[TLine]:
    """Visual lines as phrases.  With a ``gutter``, a line is split into its two text columns
    (side "L" / "R") when one of its sides is column prose; anything else - a full-width table
    row with a column gap at the gutter, a heading - stays whole ("F")."""
    lines = group_lines(info.words, list(idx))
    if gutter is None:
        return [_tline(info, ln.idx) for ln in lines]
    x_left = min(l.x0 for l in lines)
    x_right = max(l.x1 for l in lines)
    kinds = []  # "split" (two columns of text), "F" (full width), "L" / "R" (one side only)
    for ln in lines:
        left = [i for i in ln.idx if info.wbox[i, 2] <= gutter]
        right = [i for i in ln.idx if info.wbox[i, 0] >= gutter]
        clean = len(left) + len(right) == len(ln.idx)
        if clean and (not left or not right):
            kinds.append("L" if left else "R")
        elif clean and (_column_prose(info, left, x_left, gutter) or _column_prose(info, right, gutter, x_right)):
            kinds.append("split")
        else:
            kinds.append("F")
    out = []
    for k, ln in enumerate(lines):
        kind = kinds[k]
        if kind in ("L", "R"):
            # a one-sided line between full-width rows (a section label, a wrapped cell) belongs
            # to the full-width block; next to column text it belongs to its column
            nb = [kinds[j] for j in (k - 1, k + 1) if 0 <= j < len(lines) and kinds[j] in ("F", "split")
                  and abs(lines[j].y0 - ln.y0) < 3 * info.median_word_height]
            if nb and all(n == "F" for n in nb):
                kind = "F"
        if kind == "split":
            left = [i for i in ln.idx if info.wbox[i, 2] <= gutter]
            out += [_tline(info, left, "L"), _tline(info, [i for i in ln.idx if i not in left], "R")]
        else:
            out.append(_tline(info, ln.idx, kind))
    return out


# ---------------------------------------------------------------------------
# 2. regions
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Region:
    lines: list[TLine]  # body lines
    header: list[TLine] = field(default_factory=list)


def classify(line: TLine, body_size: float, content_w: float, left: float = 0.0) -> str:
    """``left`` is the page's left text margin: a heading starts there (or is centred on the
    text block); a larger bold fragment sitting over the value columns is a header cell."""
    ph = line.phrases
    if line.values_right():
        return "value"
    # "b. Details of ...", "IV. Employees": an enumerator does not make a line two cells
    if len(ph) == 2 and _ENUM.match(ph[0].text) and not ph[1].value:
        ph = [Phrase(ph[0].ids + ph[1].ids, ph[0].x0, min(ph[0].y0, ph[1].y0), ph[1].x1, max(ph[0].y1, ph[1].y1),
                     ph[0].text + " " + ph[1].text, False, ph[1].bold, ph[1].size)]
    placed = line.x0 <= left + 0.15 * content_w or abs((line.x0 + line.x1) / 2 - (left + content_w / 2)) < 0.08 * content_w
    big = lambda p: placed and (p.size >= 1.25 * body_size or (p.bold and p.size >= 1.12 * body_size))  # noqa: E731
    if len(ph) == 1:
        p = ph[0]
        if UNIT_RE.search(p.text) and p.n_words <= 8:
            return "unit"
        if big(p):
            return "heading"
        if p.n_words >= 7 and p.width >= 0.42 * content_w:
            return "prose"
        return "single"
    if len(ph) == 2 and any(UNIT_RE.search(p.text) and p.n_words <= 6 for p in ph) and not any(p.value for p in ph):
        return "unit"  # "Part A: Subsidiaries ...... INR crore": a caption line with its unit
    long = [p for p in ph if p.n_words >= 5 and p.width >= 0.25 * content_w]
    if len(long) >= 2 or any(p.n_words >= 9 and p.width >= 0.42 * content_w for p in ph):
        return "prose"  # multi-column running text
    if placed and all(p.size >= 1.4 * body_size for p in ph):
        return "heading"
    return "multi"


def _value_x(lines: list[TLine]) -> float:
    """Left edge of the value columns: values that follow a row's leading phrase (a leading
    serial number "1." is part of the row's label side)."""
    xs = [p.x0 for l in lines if l.kind == "value" for p in l.values_right() if p is not l.phrases[0]]
    return min(xs) if xs else min(p.x0 for l in lines if l.kind == "value" for p in l.values_right())


def _segments(lines: list[TLine], body_size: float) -> list[list[TLine]]:
    """Split the page's lines into runs that may each hold one table."""
    segs: list[list[TLine]] = [[]]
    for ln in lines:
        cur = segs[-1]
        if ln.kind == "prose" and cur and ln.y0 - cur[-1].y1 < 1.5 * ln.size:
            # a long label that stays within the label column (left of the value columns, or
            # left of the second column of the header above) is a row label, not running text
            if any(l.kind == "value" for l in cur):
                limit = _value_x(cur)
            else:
                heads = [l for l in cur if l.kind == "multi" and len(l.phrases) >= 2]
                limit = min(p.x0 for p in heads[-1].phrases[1:]) if heads else -1.0
            if ln.x1 < limit - ln.size:
                ln.kind = "single"
        if ln.kind == "prose" and len(ln.phrases) >= 2 and cur:
            # a row of a text table with long description cells: its phrases start exactly where
            # the columns of the bold header row above start
            heads = [l for l in cur if l.kind == "multi" and l.bold and len(l.phrases) >= 2]
            if heads:
                hx = [p.x0 for p in heads[-1].phrases]
                if len(ln.phrases) <= len(hx) and all(min(abs(p.x0 - x) for x in hx) <= 0.5 * ln.size
                                                      for p in ln.phrases):
                    ln.kind = "multi"
        if ln.kind in ("prose", "heading"):
            segs.append([])
            continue
        if cur:
            prev = cur[-1]
            if ln.y0 - prev.y1 > 3.2 * max(body_size, ln.size):
                segs.append([ln])
                continue
            vals = [l for l in cur if l.kind == "value" and not _header_like(l)]
            if vals:
                vx = _value_x(vals)
                right = max(l.x1 for l in vals)
                first = ln.phrases[0]
                sentence = (ln.kind == "single" and not first.bold and first.n_words >= 6
                            and ln.x1 > vx + 0.35 * (right - vx))
                if sentence:  # a sentence running into the value columns: prose
                    segs.append([])
                    continue
                if ln.kind == "unit" or (ln.kind == "multi" and ln.bold and ln.phrases[-1].x0 >= vx - ln.size):
                    segs.append([ln])  # a new table's unit line / header row
                    continue
        segs[-1].append(ln)
    return [s for s in segs if s]


def _numbering(line: TLine) -> bool:
    """Values 1, 2, 3, ... (3 or more) across the row: column numbers, not data."""
    vals = [p.text.rstrip(".") for p in line.values_right()]
    return len(vals) >= 3 and all(v.isdigit() for v in vals) and \
        [int(v) for v in vals] == list(range(int(vals[0]), int(vals[0]) + len(vals)))


def _header_like(ln: TLine) -> bool:
    """A line with numbers that is still a header row: a label-less row of periods ("2024 2023
    2024 2023", "FY23 FY24"), a bold header line whose only numbers are periods ("2023 |
    statement of profit and loss | ... | 2024"), or a row numbering the columns."""
    return (all(_PERIOD.match(p.text) for p in ln.phrases)
            or (ln.bold and all(_PERIOD.match(p.text) for p in ln.values_right()))
            or _numbering(ln))


def _rules_y(rules: np.ndarray, x0: float, x1: float, share: float = 0.5) -> np.ndarray:
    """y of the rules spanning at least ``share`` of [x0, x1]."""
    if not len(rules):
        return np.zeros(0)
    ov = np.minimum(rules[:, 1], x1) - np.maximum(rules[:, 0], x0)
    return rules[ov >= share * (x1 - x0), 2]


def _text_table(seg: list[TLine], rules: np.ndarray) -> Optional[Region]:
    """A table without numbers (names / positions / categories): a styled header line
    followed by multi-phrase rows that share column positions and are separated by drawn
    rules.  Rules are required - aligned text alone (signature blocks, address lists,
    two-column layouts) is not enough evidence of a table."""
    # header: a bold multi-phrase line followed by non-bold rows (the last such, nearest the data)
    heads = [i for i, l in enumerate(seg[:-1]) if l.kind == "multi" and l.bold and not seg[i + 1].bold]
    if not heads:
        return None
    h = heads[-1]
    head = seg[h]
    end = h + 1
    while end < len(seg) and seg[end].kind in ("multi", "single") and not seg[end].bold:
        end += 1
    data = seg[h + 1:end]
    while data and data[-1].kind != "multi":  # trailing single lines belong to a row only if wrapped
        glued = len(data) > 1 and data[-1].y0 - data[-2].y1 <= 0.3 * data[-1].size
        if glued:
            break
        data.pop()
    starts = [l for l in data if l.kind == "multi"]
    if not starts or len(head.phrases) < 2:
        return None
    # every data row (a multi-phrase line) shares the header's column count, within one
    if any(abs(len(l.phrases) - len(head.phrases)) > 1 for l in starts):
        return None
    x0, x1 = min(l.x0 for l in [head] + data), max(l.x1 for l in [head] + data)
    ry = _rules_y(rules, x0, x1)
    sep = sum(bool(((ry > a.cy) & (ry < b.cy)).any()) for a, b in zip(starts, starts[1:]))
    under_head = bool(((ry > head.cy) & (ry < starts[0].cy)).any())
    if len(starts) >= 2:
        ok = sep >= 0.6 * (len(starts) - 1)
    else:  # a single row: needs a rule under the header and the header's full column count
        ok = under_head and len(head.phrases) >= 3 and len(starts[0].phrases) == len(head.phrases)
    return Region(data, [head]) if ok else None


def _split_region(seg: list[TLine], rules: np.ndarray) -> Optional[Region]:
    vidx = [i for i, l in enumerate(seg) if l.kind == "value"]
    while len(vidx) > 1 and _header_like(seg[vidx[0]]):
        seg[vidx[0]].kind = "multi"
        vidx.pop(0)
    if not vidx:
        return _text_table(seg, rules)
    first, last = vidx[0], vidx[-1]
    vx = _value_x(seg)
    # trailing continuation lines of the last row's label: lowercase, or glued to it
    end = last
    while end + 1 < len(seg):
        nxt = seg[end + 1]
        gap = nxt.y0 - seg[end].y1
        wrapped_label = nxt.kind == "single" and nxt.x1 < vx and gap < 0.6 * nxt.size and (
            _LOWER_START.match(nxt.phrases[0].text) or (gap < 0.3 * nxt.size and nxt.phrases[0].n_words <= 6))
        # wrapped text cells of the last row (remarks columns) glued under it
        wrapped_cells = nxt.kind in ("single", "multi") and gap < 0.3 * nxt.size and not nxt.bold \
            and all(p.n_words <= 8 for p in nxt.phrases) and not nxt.values_right()
        if wrapped_label or wrapped_cells:
            end += 1
        else:
            break
    body = seg[first:end + 1]
    # pre-body lines, bottom-up: lines starting in the label column directly above the first
    # value row are section rows; above them, lines over the value columns form the header
    # (a right-aligned heading may be wider than its column, so what counts is where it starts)
    stub_x0 = min(seg[i].phrases[0].x0 for i in vidx)

    def in_stub(ln: TLine) -> bool:
        return ln.x0 < min(vx - ln.size, stub_x0 + 0.3 * (vx - stub_x0))

    pre = seg[:first]
    k = len(pre)
    while k > 0 and pre[k - 1].kind == "single" and in_stub(pre[k - 1]):
        k -= 1
    head_end = k
    header: list[TLine] = []
    while k > 0 and len(header) < 10 and pre[k - 1].kind in ("multi", "single"):
        header.insert(0, pre[k - 1])  # stub-only lines are kept only when sandwiched (below)
        k -= 1
    while header and header[0].kind == "single" and in_stub(header[0]):
        header.pop(0)  # a stub-only line on top is a caption, not a header row
    section = pre[head_end:] if header else []
    # a lowercase line glued under the header ("...subscribed and fully" / "paid") continues it
    while header and section and _LOWER_START.match(section[0].phrases[0].text) \
            and section[0].y0 - header[-1].y1 < 0.3 * section[0].size:
        header.append(section.pop(0))
    if len(vidx) < 2 and not (header and len(seg[first].values_right()) >= 2):
        return None
    return Region(section + body, header)


def find_regions(lines: list[TLine], body_size: float, content_w: float, left: float = 0.0,
                 rules: np.ndarray = np.zeros((0, 3))) -> list[Region]:
    for ln in lines:
        ln.kind = classify(ln, body_size, content_w, left)
    out = []
    for seg in _segments(lines, body_size):
        r = _split_region(seg, rules)
        if r is not None:
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# 3. columns
# ---------------------------------------------------------------------------

def find_columns(lines: list[TLine], res: float = 0.5) -> list[tuple[float, float]]:
    """Column x-extents from a whitespace projection over the multi-phrase body lines.
    A dip crossed by at most ~5% of the rows (a few long labels) still separates two
    columns."""
    multi = [l for l in lines if len(l.phrases) >= 2] or lines
    lo = min(p.x0 for l in multi for p in l.phrases)
    hi = max(p.x1 for l in multi for p in l.phrases)
    grid = np.arange(lo, hi + res, res)
    cover = np.zeros(len(grid), np.int64)
    for l in multi:
        m = np.zeros(len(grid), bool)
        for p in l.phrases:
            m |= (grid >= p.x0 - 0.01) & (grid <= p.x1 + 0.01)
        cover += m
    occupied = cover > int(0.05 * len(multi))
    edges = np.flatnonzero(np.diff(np.r_[0, occupied.astype(np.int8), 0]))
    cols = [(float(grid[a]), float(grid[b - 1])) for a, b in zip(edges[::2], edges[1::2])]
    if not cols:
        return [(lo, hi)]
    # labels printed on lines of their own (values vertically centred beside a wrapped label)
    # never share a line with a value: they form the label column left of the projected ones
    stub = [l.phrases[0] for l in lines if len(l.phrases) == 1 and l.phrases[0].x1 < cols[0][0] - 2]
    if stub:
        cols.insert(0, (min(p.x0 for p in stub), max(p.x1 for p in stub)))
    return cols


def _zones(cols: list[tuple[float, float]], lo: float, hi: float) -> np.ndarray:
    """Column boundaries: outer edges plus midpoints of the gaps."""
    return np.asarray([lo] + [(cols[i][1] + cols[i + 1][0]) / 2 for i in range(len(cols) - 1)] + [hi])


def split_at_columns(info: WordSet, p: Phrase, xs: np.ndarray, cols: list[tuple[float, float]]) -> list[Phrase]:
    """Split a body phrase whose words sit in two or more columns ("29 September 21 March
    2024" across two date columns).  A cut needs a word gap wider than the phrase's own word
    spacing, and every part must overlap its own column's extent, so a label overflowing into
    an empty neighbour ("Other than Permanent workers") is left whole."""
    if p.n_words < 2:
        return [p]
    zone = np.clip(np.searchsorted(xs, info.cx[p.ids], "right") - 1, 0, len(cols) - 1)
    if zone.min() == zone.max():
        return [p]
    gaps = info.wbox[p.ids[1:], 0] - info.wbox[p.ids[:-1], 2]
    ref = float(np.median(gaps)) if len(gaps) > 2 else 0.2 * p.size
    parts: list[list[int]] = []
    for k, (i, z) in enumerate(zip(p.ids, zone)):
        # cut only where the column changes at a gap wider than the phrase's word spacing
        if parts and (zone[k - 1] == z or gaps[k - 1] < 1.5 * ref):
            parts[-1].append(i)
        else:
            parts.append([i])
    if len(parts) == 1:
        return [p]
    out = [_make_phrase(info, ids) for ids in parts]
    zs = [int(zone[p.ids.index(ids[0])]) for ids in parts]
    if len(set(zs)) != len(zs) or any(_overlap(q.x0, q.x1, *cols[z]) <= 0 for q, z in zip(out, zs)):
        return [p]
    return out


def _assign_col(p: Phrase, xs: np.ndarray, first: bool = False) -> int:
    """Column of a body phrase: a line's leading text goes where it starts (labels overflow
    to the right), other phrases where most of them lie."""
    ov = np.minimum(p.x1, xs[1:]) - np.maximum(p.x0, xs[:-1])
    if not p.value:
        start = int(np.clip(np.searchsorted(xs, p.x0 + 1, "right") - 1, 0, len(xs) - 2))
        if first or ov[start] >= 0.3 * p.width:
            return start
    return int(ov.argmax())


# ---------------------------------------------------------------------------
# 4. rows
# ---------------------------------------------------------------------------

def _rule_between(rules_y: np.ndarray, a: TLine, b: TLine) -> bool:
    lo, hi = a.cy + 0.25 * a.size, b.cy - 0.25 * b.size
    return bool(((rules_y > lo) & (rules_y < hi)).any())


def build_rows(body: list[TLine], xs: np.ndarray, rules: np.ndarray) -> list[list[TLine]]:
    """Group body lines into rows: by the drawn rules when they delimit every row (rule mode),
    otherwise line by line."""
    if len(body) < 2:
        return [body]
    width = xs[-1] - xs[0]
    ry = np.zeros(0)
    if len(rules):
        ov = np.minimum(rules[:, 1], xs[-1]) - np.maximum(rules[:, 0], xs[0])
        ry = rules[(ov >= 0.3 * width) & (rules[:, 2] > body[0].y0 - 2) & (rules[:, 2] < body[-1].y1 + 2), 2]
    between = [_rule_between(ry, body[i], body[i + 1]) for i in range(len(body) - 1)]

    def value_cols(line: TLine) -> set[int]:
        return {_assign_col(p, xs, i == 0) for i, p in enumerate(line.phrases) if p.value} - {0}

    # rules delimit the rows when grouping the lines by rules never puts two values in the same
    # column into one group (wrapped labels and centred values then fall into their row);
    # rules drawn only under the header or above totals leave many rows in one group
    groups: list[list[TLine]] = [[body[0]]]
    for i in range(1, len(body)):
        if between[i - 1]:
            groups.append([body[i]])
        else:
            groups[-1].append(body[i])

    def clashes(g: list[TLine]) -> bool:
        seen: set[int] = set()
        for l in g:
            cols = value_cols(l) if any(p.value for p in l.phrases) else (
                {_assign_col(p, xs, j == 0) for j, p in enumerate(l.phrases)} - {0} if len(l.phrases) >= 2 else set())
            if seen & cols and l.values_right():
                return True
            seen |= cols
        return False

    n_clash = sum(clashes(g) for g in groups)
    rule_mode = len(ry) >= 1 and len(groups) >= 2 and n_clash <= 0.05 * len(groups)

    rows: list[list[TLine]] = [[body[0]]]
    for i in range(1, len(body)):
        prev_row, ln = rows[-1], body[i]
        prev = prev_row[-1]
        row_vals = set().union(*(value_cols(l) for l in prev_row))
        ln_vals = value_cols(ln)
        clash = bool(row_vals & ln_vals)
        if rule_mode:
            new = between[i - 1] or clash
        else:
            tight = ln.y0 - prev.y1 < 0.55 * ln.size
            first = ln.phrases[0]
            stub_text = _assign_col(first, xs, True) == 0 and not first.value
            same_style = first.bold == prev.phrases[0].bold
            if clash or not tight:
                new = True
            elif stub_text and _LOWER_START.match(first.text) and same_style:
                new = False  # wrapped label continues
            elif not stub_text and not row_vals and ln_vals:
                new = False  # values centred / bottom-aligned on a wrapped label
            else:
                new = True
        if new:
            rows.append([ln])
        else:
            prev_row.append(ln)
    return rows


# ---------------------------------------------------------------------------
# 5. header
# ---------------------------------------------------------------------------

@dataclass(slots=True, eq=False)
class _HCell:
    ids: list[int]
    x0: float
    y0: float
    x1: float
    y1: float
    fill: Optional[tuple] = None  # shading drawn behind this cell only
    level: int = 0
    cols: tuple[int, int] = (0, 1)  # [c0, c1)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2


def _overlap(a0, a1, b0, b1) -> float:
    return min(a1, b1) - max(a0, b0)


def _fill_of(info: WordSet, fills: np.ndarray, ids: list[int]) -> int:
    """Index of the smallest fill containing the centre of every word in ``ids`` (-1: none)."""
    if not len(fills):
        return -1
    cx, cy = info.cx[ids], info.cy[ids]
    inside = ((fills[:, 0, None] <= cx) & (fills[:, 2, None] >= cx) & (fills[:, 1, None] <= cy)
              & (fills[:, 3, None] >= cy)).all(1)
    hit = np.flatnonzero(inside)
    if not len(hit):
        return -1
    area = (fills[hit, 2] - fills[hit, 0]) * (fills[hit, 3] - fills[hit, 1])
    return int(hit[area.argmin()])


def _split_wide_gap(info: WordSet, p: Phrase, xs: np.ndarray) -> list[Phrase]:
    """Split a header phrase at a word gap that is clearly wider than its other word gaps
    and holds a column boundary ("Zomato Media    Zomato Media" over two columns).  A group
    heading centred over several columns has even word spacing and stays whole."""
    if p.n_words < 2:
        return [p]
    ids = sorted(p.ids, key=lambda i: info.wbox[i, 0])
    gaps = [info.wbox[ids[k + 1], 0] - info.wbox[ids[k], 2] for k in range(len(ids) - 1)]
    ref = float(np.median(gaps)) if len(gaps) > 2 else 0.22 * p.size
    inner = xs[1:-1]
    parts, cur = [], [ids[0]]
    for k, g in enumerate(gaps):
        a, b = info.wbox[ids[k], 2], info.wbox[ids[k + 1], 0]
        if g >= max(1.8 * ref, 0.25 * p.size) and ((inner > a) & (inner < b)).any():
            parts.append(cur)
            cur = []
        cur.append(ids[k + 1])
    parts.append(cur)
    return [p] if len(parts) == 1 else [_make_phrase(info, q) for q in parts]


def build_header(info: WordSet, header: list[TLine], cols: list[tuple[float, float]], xs: np.ndarray,
                 fills: np.ndarray, rules: np.ndarray = np.zeros((0, 3))):
    """Multi-level header -> ([(cell, rowspan)], n_header_rows)."""
    if not header:
        return [], 0
    # header-region fills; a fill holding two phrases side by side is a band background
    y0, y1 = header[0].y0 - 2, header[-1].y1 + 2
    if len(fills):
        fills = fills[(fills[:, 3] > y0) & (fills[:, 1] < y1)]

    def rule_between(q: Phrase, p: Phrase) -> bool:
        if not len(rules):
            return False
        r = rules[(rules[:, 2] > q.y1 - 0.2 * q.size) & (rules[:, 2] < p.y0 + 0.2 * p.size)]
        return bool(len(r)) and bool((np.minimum(r[:, 1], max(p.x1, q.x1)) - np.maximum(r[:, 0], min(p.x0, q.x0)) > 0).any())

    # (a) phrases, split where their words sit in different fills ("Number %" over two cells)
    items: list[tuple[int, Phrase, int]] = []
    for li, ln in enumerate(header):
        for p in (q for p0 in ln.phrases for q in _split_wide_gap(info, p0, xs)):
            fid = [_fill_of(info, fills, [i]) for i in p.ids]
            start = 0
            for k in range(1, len(p.ids) + 1):
                if k == len(p.ids) or fid[k] != fid[k - 1]:
                    items.append((li, _make_phrase(info, p.ids[start:k]), fid[start]))
                    start = k
    band = {f for f in {f for _, _, f in items if f >= 0}
            if any(sum(1 for li2, _, f2 in items if f2 == f and li2 == li) > 1 for li in range(len(header)))}
    # (b) cells: phrases sharing a private fill are one cell; otherwise phrases on consecutive
    # lines that overlap one-to-one stack into a multi-line cell
    cells: list[_HCell] = []
    by_fill: dict[int, _HCell] = {}
    prev_line: list[tuple[_HCell, Phrase]] = []
    for li in range(len(header)):
        cur_line = []
        line_items = [(p, f) for l2, p, f in items if l2 == li]
        for p, f in line_items:
            if f >= 0 and f not in band:
                c = by_fill.get(f)
                if c is None:
                    c = by_fill[f] = _HCell(list(p.ids), p.x0, p.y0, p.x1, p.y1, tuple(fills[f]))
                    cells.append(c)
                else:
                    c.ids += p.ids
                    c.x0, c.x1, c.y0, c.y1 = min(c.x0, p.x0), max(c.x1, p.x1), min(c.y0, p.y0), max(c.y1, p.y1)
                cur_line.append((c, p))
                continue
            above = [(c, q) for c, q in prev_line if c.fill is None
                     and _overlap(q.x0, q.x1, p.x0, p.x1) > 0.3 * min(q.width, p.width)]
            if len(above) == 1:
                c, q = above[0]
                below_q = [r for r, _ in line_items if _overlap(q.x0, q.x1, r.x0, r.x1) > 0.3 * min(q.width, r.width)]
                if len(below_q) == 1 and p.y0 - q.y1 < 0.7 * p.size and q.value == p.value \
                        and not rule_between(q, p):
                    c.ids += p.ids
                    c.x0, c.x1, c.y1 = min(c.x0, p.x0), max(c.x1, p.x1), max(c.y1, p.y1)
                    cur_line.append((c, p))
                    continue
            c = _HCell(list(p.ids), p.x0, p.y0, p.x1, p.y1)
            cells.append(c)
            cur_line.append((c, p))
        prev_line = cur_line
    # (c) columns: a shaded cell covers the columns inside its shading; a text cell the
    # columns its text overlaps (nearest column centre if none)
    ext = np.asarray(cols)
    cc = ext.mean(1)
    for c in cells:
        if c.fill is not None:
            inside = np.minimum(ext[:, 1], c.fill[2]) - np.maximum(ext[:, 0], c.fill[0])
            hit = np.flatnonzero(inside >= 0.5 * (ext[:, 1] - ext[:, 0]))
        else:
            ov = np.minimum(ext[:, 1] + 1, c.x1) - np.maximum(ext[:, 0] - 1, c.x0)
            hit = np.flatnonzero(ov > 0.5 * np.minimum(c.x1 - c.x0, ext[:, 1] - ext[:, 0] + 2))
        if len(hit):
            c.cols = (int(hit[0]), int(hit[-1]) + 1)
        else:
            k = int(np.abs(cc - c.cx).argmin())
            c.cols = (k, k + 1)
    # (d) unshaded group headings: a cell centred over 2+ cells of the band below it (within
    # its zone, half-way to its neighbours) spans their columns
    for c in sorted(cells, key=lambda c: c.y0):
        if c.fill is not None:
            continue
        same = [o for o in cells if o is not c and _overlap(o.y0, o.y1, c.y0, c.y1) > 0]
        left = max([o.x1 for o in same if o.x1 <= c.x0], default=None)
        right = min([o.x0 for o in same if o.x0 >= c.x1], default=None)
        z0 = (left + c.x0) / 2 if left is not None else xs[0] - 1
        z1 = (right + c.x1) / 2 if right is not None else xs[-1] + 1
        below = [o for o in cells if o.y0 >= c.y1 - 1 and z0 <= o.cx <= z1]
        if not below:
            continue
        top = min(o.y0 for o in below)
        kids = [o for o in below if o.y0 < top + 0.5 * (o.y1 - o.y0) + 1]
        if len(kids) < 2:
            continue
        u0, u1 = min(k.x0 for k in kids), max(k.x1 for k in kids)
        if abs((u0 + u1) / 2 - c.cx) > 0.3 * (u1 - u0) and not (c.x0 <= u0 + 2 and c.x1 >= u1 - 2):
            continue
        c.cols = (min(c.cols[0], *(k.cols[0] for k in kids)), max(c.cols[1], *(k.cols[1] for k in kids)))
    # (e) levels: one below the deepest cell above it that shares a column
    order = sorted(cells, key=lambda c: c.y0)
    for i, c in enumerate(order):
        above = [o for o in order[:i] if o.y1 <= c.y0 + 1 and _overlap(o.cols[0], o.cols[1], c.cols[0], c.cols[1]) > 0]
        c.level = 1 + max(o.level for o in above) if above else 0
    n_levels = max(c.level for c in cells) + 1
    # (f) two cells on the same level and column are one cell (merged text)
    grid: dict[tuple[int, int], _HCell] = {}
    out: list[_HCell] = []
    for c in sorted(cells, key=lambda c: (c.level, c.cols, c.y0)):
        clash = next((grid[(c.level, k)] for k in range(*c.cols) if (c.level, k) in grid), None)
        if clash is not None:
            clash.ids += c.ids
            continue
        for k in range(*c.cols):
            grid[(c.level, k)] = c
        out.append(c)
    # (g) rowspan: down to the next level used in the cell's columns, else to the body
    result = []
    for c in out:
        below = [o.level for o in out if o.level > c.level and _overlap(o.cols[0], o.cols[1], c.cols[0], c.cols[1]) > 0]
        result.append((c, (min(below) if below else n_levels) - c.level))
    return result, n_levels


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def _caption(page_lines: list[TLine], top: float, x0: float, x1: float, floor: float):
    """Up to three lines directly above the table: unit line and title."""
    above = [l for l in page_lines if l.y1 <= top + 1 and l.y0 >= floor - 1
             and _overlap(l.x0, l.x1, x0, x1) > 0]
    picked, edge, heading = [], top, None
    for ln in reversed(above):
        if heading is not None:  # past a heading only its own wrapped lines belong to the caption
            if not (ln.bold == heading.bold and abs(ln.size - heading.size) <= 1.5
                    and edge - ln.y1 < 0.6 * heading.size) or len(picked) >= 10:
                break
        elif edge - ln.y1 > 3.0 * ln.size or len(picked) == 8 or ln.kind == "value":
            break  # (an explanatory paragraph between a heading and its table is walked over)
        picked.append(ln)
        edge = ln.y0
        if heading is None and (ln.kind == "heading" or (ln.bold and ln.kind != "multi")):
            heading = ln
    picked.reverse()
    texts = [l.text for l in picked]
    unit = next((p.text for l in reversed(picked) for p in l.phrases
                 if UNIT_RE.search(p.text) and len(p.text) < 60), None)
    rest = lambda l: " ".join(p.text for p in l.phrases if p.text != unit)  # noqa: E731
    cand = [l for l in picked if rest(l)]
    styled = [i for i, l in enumerate(cand) if l.kind == "heading" or l.bold]
    i = styled[-1] if styled else len(cand) - 1
    if i < 0:
        return texts, None, unit
    # a heading wrapped over several lines is one title
    j = i  # walk up while the lines continue one heading; a numbered line ("25. ...") starts it
    while j > 0 and not _ENUM.match(cand[j].phrases[0].text.split()[0]) \
            and cand[j - 1].bold == cand[i].bold and abs(cand[j - 1].size - cand[i].size) <= 1.5 \
            and cand[j].y0 - cand[j - 1].y1 < 0.6 * cand[i].size:
        j -= 1
    return texts, " ".join(rest(l) for l in cand[j:i + 1]), unit


def assemble_region(info: PageInfo, region: Region, rules: np.ndarray, fills: np.ndarray,
                    page_lines: list[TLine], floor: float) -> Optional[tuple[Table, FillStats]]:
    body = region.lines
    cols = find_columns(body)
    if len(cols) < 2:
        return None
    all_lines = region.header + body
    lo = min(min(l.x0 for l in all_lines), cols[0][0]) - 1
    hi = max(max(l.x1 for l in all_lines), cols[-1][1]) + 1
    xs = _zones(cols, lo, hi)
    nc = len(cols)
    for ln in body:
        if len(ln.phrases) > 1:  # a lone phrase (section label) is never split
            ln.phrases = [q for p in ln.phrases for q in split_at_columns(info, p, xs, cols)]
    rows = build_rows(body, xs, rules)
    hcells, nh = build_header(info, region.header, cols, xs, fills, rules)

    words = info.words
    cells: list[Cell] = []
    top = (region.header or body)[0].y0 - 1
    if nh:
        bottom = (body[0].y0 + max(l.y1 for l in region.header)) / 2
        lv_y = [top] + [min([c.y0 for c, _ in hcells if c.level == lv], default=top) - 0.5 for lv in range(1, nh)] + [bottom]
        taken = np.zeros((nh, nc), bool)
        for c, rs in hcells:
            c0, c1 = c.cols
            cells.append(Cell(c.level, c0, join_words(words, c.ids), rs, c1 - c0,
                              (float(xs[c0]), float(lv_y[c.level]), float(xs[c1]), float(lv_y[c.level + rs])),
                              "native_pdf_text", True))
            taken[c.level:c.level + rs, c0:c1] = True
        for r, k in zip(*np.nonzero(~taken)):
            cells.append(Cell(int(r), int(k), "", 1, 1, (float(xs[k]), float(lv_y[r]), float(xs[k + 1]), float(lv_y[r + 1])),
                              "empty", True))
    ybounds = [rows[0][0].y0 - 1] + [(rows[i][-1].y1 + rows[i + 1][0].y0) / 2 for i in range(len(rows) - 1)] + \
              [rows[-1][-1].y1 + 1]
    n_words = straddling = 0
    for r, row in enumerate(rows):
        by_col: dict[int, list[int]] = {}
        for ln in row:
            for j, p in enumerate(ln.phrases):
                k = _assign_col(p, xs, j == 0)
                by_col.setdefault(k, []).extend(p.ids)
                n_words += len(p.ids)
                if not (j == 0 and k == 0):  # a leading label may overflow into empty cells
                    w = info.wbox[p.ids]
                    over = np.maximum(xs[k] - w[:, 0], 0) + np.maximum(w[:, 2] - xs[k + 1], 0)
                    straddling += int(np.count_nonzero(over > 0.34 * (w[:, 2] - w[:, 0])))
        for k in range(nc):
            ids = by_col.get(k, [])
            text = join_words(words, ids) if ids else ""
            cells.append(Cell(nh + r, k, text, 1, 1, (float(xs[k]), float(ybounds[r]), float(xs[k + 1]), float(ybounds[r + 1])),
                              "native_pdf_text" if text else "empty"))
    n_rows = nh + len(rows)
    if not nh:
        nh = internal_header_rows(cells, n_rows, nc)
        for c in cells:
            c.is_header = c.row < nh
    caption, title, unit = _caption(page_lines, top, xs[0], xs[-1], floor)
    if unit is None:  # unit printed inside the header (e.g. "INR crore, unless otherwise mentioned")
        unit = next((c.text for c in cells if c.is_header and c.text and UNIT_RE.search(c.text)), None)
    cells.sort(key=lambda c: (c.row, c.col))
    bbox = (float(xs[0]), float(top), float(xs[-1]), float(ybounds[-1]))
    table = Table(page=info.number, index=0, bbox=bbox, n_rows=n_rows, n_cols=nc, cells=cells, header_rows=nh,
                  method="native:borderless", caption=caption, title=title, unit=unit)
    return table, FillStats(n_words, straddling)


def chart_like(info: PageInfo, bbox: BBox) -> bool:
    """Data labels of a bar chart line up like table rows.  A region is chart labels when
    two or more bars - tall, narrow filled shapes of *different* heights (a shaded table
    column would be uniform) - sit on or next to it."""
    f = info.fills
    if not len(f):
        return False
    s = info.body_size
    w, h = f[:, 2] - f[:, 0], f[:, 3] - f[:, 1]
    near = ((np.minimum(f[:, 2], bbox[2]) - np.maximum(f[:, 0], bbox[0]) > 0)
            & (f[:, 1] < bbox[3] + 4 * s) & (f[:, 3] > bbox[1] - 4 * s))
    bars = f[near & (h >= 1.2 * w) & (h >= 3 * s) & (w <= 0.3 * (bbox[2] - bbox[0]))]
    if len(bars):  # a shaded header cell holds words; a bar holds at most a number
        alpha = np.asarray([any(ch.isalpha() for ch in w[4]) for w in info.words], bool)
        if alpha.any():
            cx, cy = info.cx[alpha], info.cy[alpha]
            has_text = ((bars[:, 0, None] <= cx) & (bars[:, 2, None] >= cx)
                        & (bars[:, 1, None] <= cy) & (bars[:, 3, None] >= cy)).any(1)
            bars = bars[~has_text]
    if len(bars) < 2:
        return False
    heights = np.sort(bars[:, 3] - bars[:, 1])
    return bool(heights[-1] - heights[0] > 2 * s)


def extract_borderless(info: PageInfo) -> list[tuple[Table, FillStats]]:
    """All borderless tables on the page (frame coordinates)."""
    if len(info.words) < 4:
        return []
    idx = np.arange(len(info.words))
    gutter = find_gutter(info, idx)
    lines = build_lines(info, idx, gutter)
    left = float(info.wbox[idx, 0].min())
    content_w = float(info.wbox[idx, 2].max() - left)
    rules = info.merged_rules(join_gap=4.0)
    if gutter is None:
        regions = find_regions(lines, info.body_size, content_w, left, rules)
    else:  # two-column page: each text column on its own, and the full-width lines together
        regions = []
        for side in ("F", "L", "R"):
            part = [l for l in lines if l.side == side]
            if part:
                p_left = min(l.x0 for l in part)
                regions += find_regions(part, info.body_size, content_w, p_left, rules)
        regions.sort(key=lambda r: (r.header or r.lines)[0].y0)
    out = []
    floor = info.rect[1]
    for reg in regions:
        res = assemble_region(info, reg, rules, info.fills, lines, floor)
        if res is not None and not chart_like(info, res[0].bbox):
            out.append(res)
            floor = res[0].bbox[3]
    return out
