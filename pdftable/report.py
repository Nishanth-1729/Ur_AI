"""PDF report of the extracted tables: one readable, printable document.

Every table of the canonical JSON is laid out as a real table - cell boxes,
column widths from the cell contents, wrapped text, merged cells (rowspan /
colspan), shaded header rows that repeat on continuation pages, numbers
right-aligned - in source-page / table order, behind a cover page with a
summary and a linked index.  Tables flagged ``needs_review`` carry a red banner
with the reasons and, when the source PDF is available, a picture of the
source region so the reconstruction can be checked by eye.

The report renders the extraction faithfully; it does not repair it.  JSON
stays the source of truth.

Text is drawn with a font chain (a broad Latin font, a symbol-rich font, a CJK
font, the PDF base font) and per-character fallback, so "₹", "△", "⁽¹⁾" and
Japanese text all print; widths are measured with the same fonts that draw.
"""
from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pymupdf

from .audit import audit_document
from .models import DocumentResult, Table

# ---------------------------------------------------------------------------
# fonts
# ---------------------------------------------------------------------------

_CANDIDATES = {
    "regular": ["/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
                "/System/Library/Fonts/Supplemental/Arial.ttf"],
    "bold": ["/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf",
             "/System/Library/Fonts/Supplemental/Arial Bold.ttf"],
    "symbols": ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/seguisym.ttf"],
}


def _load(paths: list[str]) -> Optional[pymupdf.Font]:
    for p in paths:
        if Path(p).is_file():
            try:
                return pymupdf.Font(fontfile=p)
            except Exception:  # unreadable / unsupported font file: try the next one
                continue
    return None


class FontChain:
    """Fonts tried in order for every character; widths come from the font that draws."""

    def __init__(self, bold: bool = False):
        primary = _load(_CANDIDATES["bold" if bold else "regular"])
        base = pymupdf.Font("hebo" if bold else "helv")
        self.fonts = [f for f in (primary, _load(_CANDIDATES["symbols"]), pymupdf.Font("cjk"), base) if f]
        self._pick: dict[str, pymupdf.Font] = {}
        self._advance: dict[pymupdf.Font, dict[str, float]] = {f: {} for f in self.fonts}

    def _length(self, f: pymupdf.Font, text: str, size: float) -> float:
        """``f.text_length(text, size)`` - the sum of the glyph advances times the size - from
        advances cached per character: layout measures millions of strings, and every
        uncached character is a round trip into MuPDF."""
        adv = self._advance[f]
        rc = 0
        for ch in text:
            a = adv.get(ch)
            if a is None:
                a = adv[ch] = f.text_length(ch, fontsize=1)
            rc += a
        return rc * size

    def font_for(self, ch: str) -> pymupdf.Font:
        f = self._pick.get(ch)
        if f is None:
            cp = ord(ch)
            f = next((f for f in self.fonts if f.has_glyph(cp)), self.fonts[0])
            self._pick[ch] = f
        return f

    def runs(self, text: str) -> list[tuple[pymupdf.Font, str]]:
        out: list[tuple[pymupdf.Font, str]] = []
        for ch in text:
            f = self.font_for(ch) if not ch.isspace() else (out[-1][0] if out else self.fonts[0])
            if out and out[-1][0] is f:
                out[-1] = (f, out[-1][1] + ch)
            else:
                out.append((f, ch))
        return out

    def width(self, text: str, size: float) -> float:
        """Sum over ``runs`` of ``_length``, without building the runs (layout calls this for
        every candidate line); the arithmetic and its order are the same."""
        total = rc = 0
        cur = None
        for ch in text:
            f = self.font_for(ch) if not ch.isspace() else (cur if cur is not None else self.fonts[0])
            if f is not cur:
                if cur is not None:
                    total += rc * size
                cur, rc = f, 0
            a = self._advance[f].get(ch)
            if a is None:
                a = self._advance[f][ch] = f.text_length(ch, fontsize=1)
            rc += a
        return total + rc * size if cur is not None else total

    def draw(self, tw: pymupdf.TextWriter, x: float, y: float, text: str, size: float) -> None:
        for f, t in self.runs(text):
            tw.append((x, y), t, font=f, fontsize=size)
            x += self._length(f, t, size)


_CJK = re.compile(r"[\u3000-\u9fff\uac00-\ud7af\uff00-\uffef]")
_NUMERIC = re.compile(r"^[\s(（\[]*[-–−+△▲]?[₹$€£¥]?\s?[\d.,]+[%％]?[)）\]]?[*#⁰¹²³⁴⁵⁶⁷⁸⁹⁽⁾]*$|^[-–—−－]{1,2}$")


def wrap(chain: FontChain, text: str, width: float, size: float) -> list[str]:
    """Greedy word wrap; explicit line breaks kept; words longer than the width (or CJK
    runs, which have no spaces) are broken between characters."""
    lines: list[str] = []
    for para in (text or "").split("\n"):
        tokens = re.findall(r"\S+\s*", para) if not _CJK.search(para) else list(para)
        cur = ""
        for tok in tokens:
            if chain.width((cur + tok).rstrip(), size) <= width or not cur:
                if chain.width(tok.rstrip(), size) > width:  # break an over-long token
                    for ch in tok:
                        if cur and chain.width((cur + ch).rstrip(), size) > width:
                            lines.append(cur.rstrip())
                            cur = ""
                        cur += ch
                else:
                    cur += tok
            else:
                lines.append(cur.rstrip())
                cur = tok
        lines.append(cur.rstrip())
    return lines or [""]


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------

A4 = (595.28, 841.89)
MARGIN = 36.0
PAD = 3.0
GRID = (0.62, 0.65, 0.70)
HEAD_FILL = (0.90, 0.93, 0.97)
TEXT = (0.10, 0.10, 0.12)
MUTED = (0.42, 0.44, 0.48)
RED = (0.78, 0.10, 0.12)


@dataclass
class _Layout:
    size: float
    widths: list[float]
    heights: list[float]
    lines: dict[int, list[str]]  # cell index -> wrapped lines
    landscape: bool


_CUT = "[… continues - full text in the JSON / CSV output]"


def _cell_lines(chain_r: FontChain, chain_b: FontChain, t: Table, widths: list[float], size: float,
                landscape: bool) -> dict[int, list[str]]:
    """Wrapped lines per cell; a cell taller than a page is cut, and says so."""
    page_h = A4[0] if landscape else A4[1]
    max_lines = max(4, int((page_h - 2 * MARGIN - 160) / (size * 1.25)))
    out = {}
    for i, c in enumerate(t.cells):
        w = sum(widths[c.col:c.col + c.colspan]) - 2 * PAD
        lines = wrap(chain_b if c.is_header else chain_r, c.text, max(w, 4), size)
        if len(lines) > max_lines:
            lines = lines[:max_lines - 1] + wrap(chain_r, _CUT, max(w, 4), size)
        out[i] = lines
    return out


def _column_needs(t: Table, chain_r: FontChain, chain_b: FontChain, size: float) -> tuple[list[float], list[float]]:
    """Per column: (minimum width = longest word, wanted width = longest line, capped)."""
    lo = [2 * PAD + 8.0] * t.n_cols
    hi = [2 * PAD + 8.0] * t.n_cols
    for c in t.cells:
        if c.colspan != 1 or not c.text:
            continue
        ch = chain_b if c.is_header else chain_r
        words = re.findall(r"\S+", c.text) if not _CJK.search(c.text) else list(c.text.replace("\n", ""))
        longest_word = max((ch.width(w, size) for w in words), default=0.0)
        longest_line = max(ch.width(l, size) for l in c.text.split("\n"))
        lo[c.col] = max(lo[c.col], longest_word + 2 * PAD)
        hi[c.col] = max(hi[c.col], min(longest_line, 120.0 if c.is_header else 230.0) + 2 * PAD)
    return lo, [max(a, b) for a, b in zip(lo, hi)]


def _fit(lo: list[float], hi: list[float], avail: float) -> Optional[list[float]]:
    """Widths within ``avail``: everything wanted if it fits, else the minimum plus the spare
    width shared in proportion to what each column still wants; None if even the minimum
    does not fit."""
    if sum(lo) > avail:
        return None
    if sum(hi) <= avail:
        return hi
    need = [b - a for a, b in zip(lo, hi)]
    return [a + (avail - sum(lo)) * n / sum(need) for a, n in zip(lo, need)]


def layout(t: Table, chain_r: FontChain, chain_b: FontChain) -> _Layout:
    """Portrait at 8 pt when the table fits with moderate wrapping; otherwise landscape at the
    largest type (8 down to 5.5 pt) whose longest words fit; as a last resort, even columns."""
    avail_p, avail_l = A4[0] - 2 * MARGIN, A4[1] - 2 * MARGIN
    choice = None
    lo, hi = _column_needs(t, chain_r, chain_b, 8.0)
    widths = _fit(lo, hi, avail_p)
    if widths is not None and sum(hi) <= 1.35 * avail_p:
        choice = (8.0, widths, False)
    else:
        for size in (8.0, 7.5, 7.0, 6.5, 6.0, 5.5):
            lo, hi = _column_needs(t, chain_r, chain_b, size)
            widths = _fit(lo, hi, avail_l)
            if widths is not None:
                choice = (size, widths, True)
                break
    if choice is None:  # very wide table: smallest type, columns squeezed evenly, text wraps
        choice = (5.0, [avail_l / t.n_cols] * t.n_cols, True)
    size, widths, landscape = choice
    lines = _cell_lines(chain_r, chain_b, t, widths, size, landscape)
    return _Layout(size, widths, _row_heights(t, lines, size), lines, landscape)


def _row_heights(t: Table, lines: dict[int, list[str]], size: float) -> list[float]:
    lh = size * 1.25
    h = [lh + 2 * PAD] * t.n_rows
    for i, c in enumerate(t.cells):
        if c.rowspan == 1:
            h[c.row] = max(h[c.row], len(lines[i]) * lh + 2 * PAD)
    for i, c in enumerate(t.cells):  # a merged cell taller than its rows stretches the last one
        if c.rowspan > 1:
            need = len(lines[i]) * lh + 2 * PAD
            have = sum(h[c.row:c.row + c.rowspan])
            if need > have:
                h[c.row + c.rowspan - 1] += need - have
    return h


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

class _Writer:
    def __init__(self, out: pymupdf.Document, name: str):
        self.out = out
        self.name = name
        self.r = FontChain()
        self.b = FontChain(bold=True)
        self.page: Optional[pymupdf.Page] = None
        self.y = 0.0

    def new_page(self, landscape: bool = False) -> pymupdf.Page:
        w, h = (A4[1], A4[0]) if landscape else A4
        self.page = self.out.new_page(width=w, height=h)
        self.y = MARGIN
        return self.page

    @property
    def bottom(self) -> float:
        return self.page.rect.height - MARGIN - 14

    def text(self, x: float, y: float, s: str, size: float, bold: bool = False, color=TEXT,
             width: Optional[float] = None, tw: Optional[pymupdf.TextWriter] = None) -> float:
        """Draw (wrapped to ``width``) text with its first baseline at ``y``; returns the y below it.
        With ``tw`` the text is only added to that writer (the caller writes it out once)."""
        chain = self.b if bold else self.r
        lines = wrap(chain, s, width, size) if width else s.split("\n")
        own = tw is None
        if own:
            tw = pymupdf.TextWriter(self.page.rect, color=color)
        for k, line in enumerate(lines):
            chain.draw(tw, x, y + k * size * 1.25, line, size)
        if own:
            tw.write_text(self.page)
        return y + len(lines) * size * 1.25


def _heading(w: _Writer, t: Table, cont: bool) -> None:
    page_w = w.page.rect.width
    label = f"Source page {t.page} · Table {t.index}" + (" (continued)" if cont else "")
    w.y = w.text(MARGIN, w.y + 10, label, 10.5, bold=True)
    if t.title and not cont:
        w.y = w.text(MARGIN, w.y + 1, t.title, 9, width=page_w - 2 * MARGIN) - 1
    meta = [f"{t.n_rows} rows × {t.n_cols} columns", f"{t.header_rows} header row(s)", t.method]
    if t.unit:
        meta.insert(0, f"Unit: {t.unit}")
    if t.rotation:
        meta.append(f"printed rotated {t.rotation}°")
    if t.reliability is not None:
        meta.append(f"reliability {t.reliability.score:.2f}")
    w.y = w.text(MARGIN, w.y + 1, "  ·  ".join(meta), 7, color=MUTED, width=page_w - 2 * MARGIN)
    if t.needs_review and not cont:
        reasons = "; ".join(t.reliability.reasons) if t.reliability and t.reliability.reasons else "flagged as uncertain"
        box_top = w.y + 3
        y_end = w.text(MARGIN + 6, box_top + 10, f"NEEDS REVIEW - {reasons}", 8, bold=True, color=RED,
                       width=page_w - 2 * MARGIN - 12)
        w.page.draw_rect(pymupdf.Rect(MARGIN, box_top, page_w - MARGIN, y_end + 1), color=RED, fill=(1, 0.94, 0.94),
                         width=0.8, overlay=False)
        w.y = y_end + 2
    w.y += 6


def _draw_rows(w: _Writer, t: Table, lay: _Layout, rows: list[int]) -> None:
    """Draw the given grid rows (header rows first) at w.y; merged cells are drawn on the first
    of their rows present in ``rows`` and span the rows that follow it on this page."""
    x0 = MARGIN
    xs = [x0]
    for wd in lay.widths:
        xs.append(xs[-1] + wd)
    ys = {}
    y = w.y
    for r in rows:
        ys[r] = y
        y += lay.heights[r]
    present = set(rows)
    lh = lay.size * 1.25
    tw_r = pymupdf.TextWriter(w.page.rect, color=TEXT)
    shapes = w.page.new_shape()
    for i, c in enumerate(t.cells):
        span_rows = [r for r in range(c.row, c.row + c.rowspan) if r in present]
        if not span_rows:
            continue
        r0 = span_rows[0]
        cy0 = ys[r0]
        cy1 = ys[span_rows[-1]] + lay.heights[span_rows[-1]]
        rect = pymupdf.Rect(xs[c.col], cy0, xs[c.col + c.colspan], cy1)
        if c.is_header:
            shapes.draw_rect(rect)
            shapes.finish(color=GRID, fill=HEAD_FILL, width=0.5)
        else:
            shapes.draw_rect(rect)
            shapes.finish(color=GRID, width=0.4)
        if not c.text:
            continue
        chain = w.b if c.is_header else w.r
        lines = lay.lines[i]
        numeric = not c.is_header and bool(_NUMERIC.match(c.text.strip()))
        block = len(lines) * lh
        top = cy0 + PAD if not c.is_header else cy0 + max(PAD, (cy1 - cy0 - block) / 2)
        for k, line in enumerate(lines):
            lw = chain.width(line, lay.size)
            if numeric:
                lx = rect.x1 - PAD - lw
            elif c.is_header and c.col > 0:
                lx = rect.x0 + max(PAD, (rect.width - lw) / 2)
            else:
                lx = rect.x0 + PAD
            chain.draw(tw_r, lx, top + lay.size + k * lh - 1.0, line, lay.size)
    shapes.draw_rect(pymupdf.Rect(xs[0], w.y, xs[-1], y))
    shapes.finish(color=(0.35, 0.37, 0.40), width=0.8)
    shapes.commit()
    tw_r.write_text(w.page)
    w.y = y


def _row_blocks(t: Table) -> list[list[int]]:
    """Body rows grouped so that a merged cell never breaks across pages."""
    blocks: list[list[int]] = []
    end = -1
    for r in range(t.header_rows, t.n_rows):
        if r <= end and blocks:
            blocks[-1].append(r)
        else:
            blocks.append([r])
        for c in t.cells:
            if c.row == r and c.rowspan > 1:
                end = max(end, r + c.rowspan - 1)
    return blocks


def draw_table(w: _Writer, t: Table, src: Optional[pymupdf.Document]) -> int:
    """Lay out one table on fresh page(s); returns the 0-based page where it starts."""
    lay = layout(t, w.r, w.b)
    w.new_page(lay.landscape)
    start = w.out.page_count - 1
    _heading(w, t, cont=False)
    head = list(range(t.header_rows))
    head_h = sum(lay.heights[r] for r in head)
    pending: list[int] = []
    for block in _row_blocks(t):
        bh = sum(lay.heights[r] for r in block)
        used = sum(lay.heights[r] for r in pending)
        if pending and w.y + head_h + used + bh > w.bottom:
            _draw_rows(w, t, lay, head + pending)
            pending = []
            w.new_page(lay.landscape)
            _heading(w, t, cont=True)
        pending += block
    _draw_rows(w, t, lay, head + pending)
    if t.needs_review and src is not None and t.bbox is not None:
        _source_crop(w, t, src, lay.landscape)
    return start


def _source_crop(w: _Writer, t: Table, src: pymupdf.Document, landscape: bool) -> None:
    """Picture of the table's region on the source page, for checking the reconstruction."""
    page = src[t.page - 1]
    clip = pymupdf.Rect(t.bbox) + (-6, -6, 6, 6)
    clip &= page.rect
    if clip.is_empty:
        return
    mat = pymupdf.Matrix(2, 2)
    if t.rotation:
        mat = mat.prerotate(t.rotation)  # show rotated tables upright
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    pw = w.page.rect.width - 2 * MARGIN
    scale = min(pw / pix.width, (w.page.rect.height - 2 * MARGIN - 60) / pix.height, 0.75)
    iw, ih = pix.width * scale, pix.height * scale
    if w.y + 24 + ih > w.bottom:
        w.new_page(landscape)
    w.y = w.text(MARGIN, w.y + 16, f"Source region on page {t.page} (for visual comparison)", 8, bold=True, color=MUTED)
    rect = pymupdf.Rect(MARGIN, w.y + 3, MARGIN + iw, w.y + 3 + ih)
    w.page.insert_image(rect, pixmap=pix)
    w.page.draw_rect(rect, color=GRID, width=0.5)
    w.y = rect.y1


def _cover(w: _Writer, doc: DocumentResult) -> None:
    w.new_page()
    name = Path(doc.source).name
    tables = doc.tables
    review = [t for t in tables if t.needs_review]
    routes: dict[str, int] = {}
    for p in doc.pages:
        routes[p.route] = routes.get(p.route, 0) + 1
    w.y = w.text(MARGIN, MARGIN + 18, "Extracted tables", 22, bold=True)
    w.y = w.text(MARGIN, w.y + 4, name, 12, color=MUTED)
    aud = audit_document(doc)
    lines = [
        f"{len(doc.pages)} page(s) processed, {len(tables)} table(s) extracted, "
        f"{len(review)} flagged NEEDS REVIEW.",
        "Routes: " + ", ".join(f"{k} {v}" for k, v in sorted(routes.items())),
    ]
    if aud["total_cells_checkable"]:
        lines.append(f"Arithmetic check: {aud['total_cells_verified']} of {aud['total_cells_checkable']} "
                     f"checkable 'Total' cells equal the sum of the rows above them "
                     f"({aud['verified_ratio']:.0%}).")
    lines += [
        f"Generated {_dt.date.today().isoformat()} from the canonical JSON; JSON / CSV remain the "
        "machine-readable outputs.  Each table starts on a new page; header rows repeat when a "
        "table continues.  Tables flagged NEEDS REVIEW show the reasons and the source region.",
    ]
    for s in lines:
        w.y = w.text(MARGIN, w.y + 8, s, 9, width=A4[0] - 2 * MARGIN)
    w.y += 14


def _index(w: _Writer, doc: DocumentResult, starts: list[int], index_pages: list[int]) -> None:
    """Index table (page · table · title · size · status) with links to each table."""
    cols = [52, 40, 300, 60, 71]
    heads = ["Source page", "Table", "Title", "Size", "Status"]
    lh = 11.0
    pi = 0
    # per index page: one shape for the cell boxes and one writer per text colour, written once
    page = {}

    def open_page(k: int) -> None:
        w.page = w.out[index_pages[k]]
        page.update(shape=w.page.new_shape(), text=pymupdf.TextWriter(w.page.rect, color=TEXT),
                    red=pymupdf.TextWriter(w.page.rect, color=RED))

    def close_page() -> None:
        page["shape"].commit()
        page["text"].write_text(w.page)
        page["red"].write_text(w.page)

    def head_row():
        x = MARGIN
        for c, h in zip(cols, heads):
            page["shape"].draw_rect(pymupdf.Rect(x, w.y, x + c, w.y + lh + 2))
            page["shape"].finish(color=GRID, fill=HEAD_FILL, width=0.4)
            w.text(x + 3, w.y + 9, h, 7.5, bold=True, tw=page["text"])
            x += c
        w.y += lh + 2

    open_page(pi)
    head_row()
    for t, start in zip(doc.tables, starts):
        if w.y + lh > w.bottom:
            close_page()
            pi += 1
            open_page(pi)
            w.y = MARGIN
            head_row()
        vals = [str(t.page), str(t.index), (t.title or "").replace("\n", " "), f"{t.n_rows}×{t.n_cols}",
                "NEEDS REVIEW" if t.needs_review else "ok"]
        x = MARGIN
        for k, (c, v) in enumerate(zip(cols, vals)):
            page["shape"].draw_rect(pymupdf.Rect(x, w.y, x + c, w.y + lh))
            page["shape"].finish(color=GRID, width=0.3)
            s = v
            while s and w.r.width(s, 7) > c - 6:
                s = s[:-2] + "…" if len(s) > 2 else ""
            w.text(x + 3, w.y + 8, s, 7, tw=page["red" if k == 4 and t.needs_review else "text"])
            x += c
        w.page.insert_link({"kind": pymupdf.LINK_GOTO, "page": start, "from": pymupdf.Rect(MARGIN, w.y, x, w.y + lh),
                            "to": pymupdf.Point(MARGIN, MARGIN)})
        w.y += lh
    close_page()


def write_report(doc: DocumentResult, path: Path, source_pdf: Optional[Path] = None) -> Path:
    out = pymupdf.open()
    w = _Writer(out, Path(doc.source).name)
    _cover(w, doc)
    tables = doc.tables
    # reserve index pages after the cover (the cover page holds the first rows)
    first_rows = int((A4[1] - MARGIN - 14 - w.y - 13) // 11)
    n_index_pages = 1 + max(0, -(-(len(tables) - first_rows) // int((A4[1] - 2 * MARGIN - 27) // 11)))
    for _ in range(n_index_pages - 1):
        out.new_page(width=A4[0], height=A4[1])
    index_pages = list(range(n_index_pages))
    cover_y = w.y
    src = None
    sp = source_pdf if source_pdf is not None else Path(doc.source)
    if sp is not None and Path(sp).is_file():
        src = pymupdf.open(sp)
    starts = []
    toc = []
    for t in tables:
        starts.append(draw_table(w, t, src))
        label = f"p{t.page} t{t.index}" + (f" - {t.title[:60]}" if t.title else "") + (" [REVIEW]" if t.needs_review else "")
        toc.append([1, label, starts[-1] + 1])
    if not tables:
        w.page = out[0]
        w.text(MARGIN, cover_y + 10, "No tables were detected in this document.", 11, bold=True)
    else:
        w.y = cover_y
        _index(w, doc, starts, index_pages)
    total = out.page_count
    for k, page in enumerate(out):
        w.page = page
        tw = pymupdf.TextWriter(page.rect, color=MUTED)
        w.text(MARGIN, page.rect.height - MARGIN + 8, f"{w.name} - extracted tables", 7, tw=tw)
        label = f"{k + 1} / {total}"
        w.text(page.rect.width - MARGIN - w.r.width(label, 7), page.rect.height - MARGIN + 8, label, 7, tw=tw)
        tw.write_text(page)
    if toc:
        out.set_toc(toc)
    try:
        out.subset_fonts()  # embed only the glyphs used (the CJK font alone is several MB)
    except Exception:  # subsetting needs fontTools; without it the full fonts stay embedded
        pass
    out.save(path, garbage=3, deflate=True, deflate_fonts=True)
    out.close()
    if src is not None:
        src.close()
    return path
