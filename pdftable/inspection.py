"""Phase 1 - page inspection with PyMuPDF.

Collects everything later phases need in one pass over the page: words with
coordinates (as numpy arrays for vectorised assignment), per-word style (bold,
size, colour), page geometry, image coverage and text-layer quality.  Vector
graphics are read lazily, once per page, and rules / separators / shading are
derived from that one read.

Two normalisations happen here so that every later phase can assume ordinary
left-to-right, top-to-bottom text:

* **reading frame** - landscape tables are often printed rotated by 90 degrees
  on a portrait page.  Their words come out as tall, narrow boxes, and any
  column/row logic applied to them produces garbage grids.  When most of the
  page's text runs vertically, the page is re-imposed into a rotated view (a
  temporary one-page PDF showing the original page turned upright) and all
  extraction - words, drawings, PyMuPDF's table finder, rendering for OCR -
  runs on that view.  ``to_source`` maps view coordinates back to the page.
* **page furniture** - running headers, footers and page numbers that repeat
  across the document are removed from the word list (see ``Furniture``),
  so they cannot be mistaken for a table row or a caption.

Words whose text direction differs from the frame (e.g. a vertical side label
on a portrait page) are dropped from the word list for the same reason.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Sequence

import numpy as np
import pymupdf

from .models import BBox

# a unit statement: "(INR crores)", "INR crore, unless ...", "in thousands", "(₹ million)", "（単位：千円）"
# - a bare "thousand" in running text ("one lakh twenty nine thousand") is not one
UNIT_RE = re.compile(r"単位|\bunits?\s*[:：]|\(\s*units?\b|\bin\s+units\b"
                     r"|\bin\s+(thousands|millions|billions|lakhs?|crores?)\b|千円|百万円|億円"
                     r"|\b(INR|USD|EUR|GBP|Rs\.?|₹|\$)\s*(in\s+)?(crores?|lakhs?|thousands?|millions?|billions?|mn|bn|'?000s?)\b"
                     r"|\(\s*(INR|USD|EUR|Rs\.?|₹|\$)?\s*(in\s+)?(crores?|lakhs?|thousands?|millions?|billions?|mn|bn)\s*\)", re.I)
# Characters that signal a broken text layer (missing ToUnicode map, etc.).
_BAD_CHARS = re.compile(r"[\ufffd\ue000-\uf8ff\x00-\x08\x0b\x0c\x0e-\x1f]")
_CTRL = re.compile(r"[\x00-\x1f\x7f]")
_BOLD_FONT = re.compile(r"bold|black|heavy|semibold|demi", re.I)


class Line:
    """Words sharing a baseline, left to right."""
    __slots__ = ("idx", "x0", "y0", "x1", "y1", "text")

    def __init__(self, idx: list[int], words: list[tuple]):
        self.idx = sorted(idx, key=lambda i: words[i][0])
        ws = [words[i] for i in self.idx]
        self.x0 = min(w[0] for w in ws)
        self.y0 = min(w[1] for w in ws)
        self.x1 = max(w[2] for w in ws)
        self.y1 = max(w[3] for w in ws)
        self.text = " ".join(w[4] for w in ws)

    @property
    def height(self) -> float:
        return self.y1 - self.y0


def group_lines(words: list[tuple], idx) -> list[Line]:
    """Cluster words into visual lines; returns lines top to bottom.  A word joins a line
    when its vertical centre lies within the line's own centre band - the band does not
    grow as words join, so a value centred beside a wrapped label (between two of its
    lines) cannot chain those lines together."""
    order = sorted(idx, key=lambda i: (words[i][1] + words[i][3]) / 2)
    lines: list[list[int]] = []
    ref_c = ref_h = 0.0
    for i in order:
        y0, y1 = words[i][1], words[i][3]
        c, h = (y0 + y1) / 2, y1 - y0
        if lines and abs(c - ref_c) <= 0.35 * min(h, ref_h):
            lines[-1].append(i)
        else:
            lines.append([i])
            ref_c, ref_h = c, h
    return [Line(l, words) for l in lines]


def join_words(words: list[tuple], idx) -> str:
    """Reading-order text of a set of words: spaces within a line, newlines between lines."""
    if not idx:
        return ""
    return "\n".join(l.text for l in group_lines(words, idx))


def phrases(line: Line, words: list[tuple], gap: float) -> list[tuple[float, float, list[int]]]:
    """Split a line at horizontal gaps wider than ``gap`` -> [(x0, x1, word_idx)]."""
    out: list[tuple[float, float, list[int]]] = []
    for i in line.idx:
        w = words[i]
        if out and w[0] - out[-1][1] <= gap:
            x0, x1, ids = out[-1]
            out[-1] = (x0, max(x1, w[2]), ids + [i])
        else:
            out.append((w[0], w[2], [i]))
    return out


class WordSet:
    """Words ``(x0, y0, x1, y1, text, ...)`` plus numpy views for vectorised lookups.
    Used for the native text layer and for OCR output alike."""

    def __init__(self, words: list[tuple], scores=None):
        self.words = words
        a = np.asarray([w[:4] for w in words], dtype=np.float64) if words else np.zeros((0, 4))
        self.wbox = a
        self.cx = (a[:, 0] + a[:, 2]) / 2
        self.cy = (a[:, 1] + a[:, 3]) / 2
        self.scores = None if scores is None else np.asarray(scores, dtype=np.float64)
        n = len(words)
        # style per word; OCR words have none, so they default to plain text of word height
        self.wsize = a[:, 3] - a[:, 1] if n else np.zeros(0)
        self.wbold = np.zeros(n, bool)
        self.wcolor = np.zeros(n, np.int64)

    @cached_property
    def median_word_height(self) -> float:
        if not len(self.wbox):
            return 10.0
        return float(np.median(self.wbox[:, 3] - self.wbox[:, 1]))

    @cached_property
    def body_size(self) -> float:
        """Most common font size, weighted by characters (the running-text size)."""
        if not len(self.words):
            return 10.0
        c: Counter = Counter()
        for s, w in zip(np.round(self.wsize, 1), self.words):
            c[float(s)] += len(w[4])
        return c.most_common(1)[0][0]

    def words_in(self, bbox: BBox, pad: float = 0.0) -> np.ndarray:
        x0, y0, x1, y1 = bbox
        m = (self.cx >= x0 - pad) & (self.cx <= x1 + pad) & (self.cy >= y0 - pad) & (self.cy <= y1 + pad)
        return np.nonzero(m)[0]


# ---------------------------------------------------------------------------
# Reading frame (rotated pages)
# ---------------------------------------------------------------------------

def text_direction(page: pymupdf.Page) -> Counter:
    """Characters per text direction: 'h' (left-to-right), 'up', 'down', 'rtl'."""
    dirs: Counter = Counter()
    for b in page.get_text("dict", flags=pymupdf.TEXTFLAGS_WORDS)["blocks"]:
        for l in b.get("lines", []):
            dx, dy = l["dir"]
            key = "h" if dx > 0.9 else "rtl" if dx < -0.9 else "up" if dy < 0 else "down"
            dirs[key] += sum(len(s["text"].strip()) for s in l["spans"])
    return dirs


def detect_rotation(page: pymupdf.Page, share: float = 0.6) -> int:
    """Clockwise degrees (0, 90, 270) by which the page's content must be turned so
    that most of its text reads left to right."""
    dirs = text_direction(page)
    total = sum(dirs.values())
    if not total:
        return 0
    if dirs["up"] >= share * total:
        return 90
    if dirs["down"] >= share * total:
        return 270
    return 0


def frame_to_page(rotation: int, width: float, height: float) -> Optional[pymupdf.Matrix]:
    """Matrix mapping reading-frame coordinates to the original page (``width`` x ``height``)."""
    if rotation == 90:  # text ran bottom-to-top: frame (x', y') = (h - y, x)
        return pymupdf.Matrix(0, -1, 1, 0, 0, height)  # x = y', y = h - x'
    if rotation == 270:  # text ran top-to-bottom: frame (x', y') = (y, w - x)
        return pymupdf.Matrix(0, 1, -1, 0, width, 0)  # x = w - y', y = x'
    return None


def _rotated_view(page: pymupdf.Page, rotation: int):
    """One-page PDF showing ``page`` turned clockwise by ``rotation`` degrees, plus the
    matrix mapping view coordinates back to the original page."""
    r = page.rect
    doc = pymupdf.open()
    view = doc.new_page(width=r.height, height=r.width)
    view.show_pdf_page(view.rect, page.parent, page.number, rotate=-rotation)
    return doc, view, frame_to_page(rotation, r.width, r.height)


def _normalized(r: Sequence[float]) -> BBox:
    """A raw rectangle tuple with its corners ordered (``pymupdf.Rect.normalize``)."""
    return min(r[0], r[2]), min(r[1], r[3]), max(r[0], r[2]), max(r[1], r[3])


def map_bbox(b: Optional[BBox], m: Optional[pymupdf.Matrix]) -> Optional[BBox]:
    if b is None or m is None:
        return b
    r = pymupdf.Rect(b) * m
    r.normalize()
    return (r.x0, r.y0, r.x1, r.y1)


# ---------------------------------------------------------------------------
# Page furniture (running headers / footers / page numbers)
# ---------------------------------------------------------------------------

_DIGITS = re.compile(r"\d+")
FURNITURE_BAND = 0.07  # share of the page height searched at the top and at the bottom


def furniture_sample(n_pages: int, sample: int = 80) -> list[int]:
    """At most ``sample`` evenly spaced pages over which repetition is counted."""
    return sorted(set(np.linspace(0, n_pages - 1, min(n_pages, sample)).astype(int).tolist()))


def band_lines(page: pymupdf.Page, band: float = FURNITURE_BAND) -> list[tuple[str, BBox]]:
    """Text lines in the top and bottom ``band`` of the page, with their boxes."""
    r = page.rect
    out = []
    for clip in (pymupdf.Rect(r.x0, r.y0, r.x1, r.y0 + band * r.height),
                 pymupdf.Rect(r.x0, r.y1 - band * r.height, r.x1, r.y1)):
        words = page.get_text("words", clip=clip)
        for ln in group_lines(words, range(len(words))):
            out.append((ln.text, (ln.x0, ln.y0, ln.x1, ln.y1)))
    return out


def _text_key(t: str, b: BBox) -> tuple:
    return _DIGITS.sub("#", t.strip()), round(b[1] / 6)


def _place_key(b: BBox) -> tuple:
    return round(b[1] / 4), round(b[3] / 4), round(b[2] / 10)


@dataclass(frozen=True)
class Furniture:
    """Running headers / footers of one document: lines in the top / bottom ``band`` of the page
    whose text (digits ignored) recurs at the same height, or that recur at the same place with
    section-specific text ("Statutory Reports: CGR" vs "Financial Statements: Standalone"), on at
    least ``need`` of the sampled pages; plus bare page numbers.  Small and picklable, so each
    worker finds the furniture of its own pages (``boxes``)."""
    counts: Counter
    places: Counter
    need: float
    band: float

    @classmethod
    def from_lines(cls, sampled: list[list[tuple[str, BBox]]], band: float = FURNITURE_BAND,
                   min_share: float = 0.2) -> "Furniture":
        """``sampled``: the ``band_lines`` of each sampled page.  A line is furniture when it
        recurs on ``min_share`` of them, and on at least 3."""
        counts = Counter(k for ls in sampled for k in {_text_key(t, b) for t, b in ls})
        places = Counter(k for ls in sampled for k in {_place_key(b) for _, b in ls})
        return cls(counts, places, max(3, min_share * len(sampled)), band)

    @classmethod
    def learn(cls, doc: pymupdf.Document, band: float = FURNITURE_BAND) -> "Furniture":
        return cls.from_lines([band_lines(doc[p], band) for p in furniture_sample(doc.page_count)], band)

    def boxes(self, page: pymupdf.Page) -> list[BBox]:
        return [b for t, b in band_lines(page, self.band)
                if self.counts[_text_key(t, b)] >= self.need or self.places[_place_key(b)] >= self.need
                or t.strip().isdigit()]


# ---------------------------------------------------------------------------
# Page inspection
# ---------------------------------------------------------------------------

_SUP = str.maketrans("0123456789(),+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁽⁾,⁺⁻")
_SUP_TEXT = re.compile(r"^[\d(),+\-]{1,6}$")


def _attach_superscripts(words: list[tuple], size: list[float]) -> tuple[list[tuple], list[float], list[int]]:
    """Footnote markers set as small raised digits ("78,72,500¹", "commerce⁽¹⁾") come out as
    separate words.  Attach each to the word it touches, as Unicode superscripts, so a value
    and its footnote mark stay one token.  Returns (words, sizes, kept original indices)."""
    if not words:
        return words, size, []
    drop: set[int] = set()
    text = [w[4] for w in words]
    box = [list(w[:4]) for w in words]
    for ln in group_lines(words, range(len(words))):
        ids = ln.idx
        for a, b in zip(ids, ids[1:]):
            if a in drop:
                continue
            ha = words[a][3] - words[a][1]
            if (size[b] <= 0.8 * size[a] and _SUP_TEXT.match(words[b][4]) and words[b][0] - box[a][2] <= 0.15 * size[a]
                    and words[b][3] < words[a][3] - 0.15 * ha):
                text[a] += words[b][4].translate(_SUP)
                box[a][2] = max(box[a][2], words[b][2])
                drop.add(b)
    keep = [i for i in range(len(words)) if i not in drop]
    out = [(*box[i], text[i], *words[i][5:]) for i in keep]
    return out, [size[i] for i in keep], keep


def _superscript_suffix(text: str, w: tuple, spans: list[dict]) -> str:
    """A footnote mark typeset without a space ("78,72,500" + small raised "1") is one
    extracted word; rewrite its trailing marker as superscript characters."""
    over = sorted((s for s in spans if s["text"].strip() and s["bbox"][0] < w[2] - 0.1 and s["bbox"][2] > w[0] + 0.1),
                  key=lambda s: s["bbox"][0])
    if len(over) < 2:
        return text
    main = max(over, key=lambda s: s["size"])
    tail = over[over.index(main) + 1:]
    sup = "".join(s["text"].strip() for s in tail)
    if (tail and all(s["size"] <= 0.8 * main["size"] and s["bbox"][3] < main["bbox"][3] - 0.1 * main["size"]
                     for s in tail) and _SUP_TEXT.match(sup) and text.endswith(sup) and len(sup) < len(text)):
        return text[:-len(sup)] + sup.translate(_SUP)
    return text


def _read_words(page: pymupdf.Page, furniture: Sequence[BBox], to_view: Optional[pymupdf.Matrix]):
    """Words of the frame page with style, dropping control characters, words running in
    another direction than the frame, and words inside page-furniture boxes."""
    tp = page.get_textpage(flags=pymupdf.TEXTFLAGS_WORDS)
    raw = tp.extractWORDS()
    blocks = tp.extractDICT()["blocks"]
    furn = [map_bbox(b, to_view) for b in furniture]
    words, size, bold, color = [], [], [], []
    for w in raw:
        text = _CTRL.sub("", w[4])
        if not text.strip():
            continue
        line = blocks[w[5]]["lines"][w[6]]
        if line["dir"][0] < 0.9:  # vertical / upside-down relative to the frame
            continue
        cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
        if any(f[0] - 1 <= cx <= f[2] + 1 and f[1] - 1 <= cy <= f[3] + 1 for f in furn):
            continue
        spans = line["spans"]
        span = next((s for s in spans if s["bbox"][0] - 0.5 <= cx <= s["bbox"][2] + 0.5 and s["text"].strip()),
                    spans[0])
        text = _superscript_suffix(text, w, spans)
        words.append((w[0], w[1], w[2], w[3], text, w[5], w[6], w[7]))
        size.append(span["size"])
        bold.append(bool(span["flags"] & 16) or bool(_BOLD_FONT.search(span["font"])))
        color.append(span["color"])
    words, size, keep = _attach_superscripts(words, size)
    return words, size, [bold[i] for i in keep], [color[i] for i in keep]


class PageInfo(WordSet):
    """Words and geometry of one page, in its reading frame (see module docstring).

    ``page`` is the page all extraction runs on (the original page, or the upright
    rotated view); ``source_page`` is always the original.  ``rotation`` is 0, 90 or
    270 and ``to_source`` maps frame coordinates to the original page (None if 0).
    """

    def __init__(self, page: pymupdf.Page, furniture: Sequence[BBox] = (), rotation: Optional[int] = None):
        self.source_page = page
        self.number = page.number + 1
        self.rotation = detect_rotation(page) if rotation is None else rotation
        self._view_doc = None  # keeps the rotated view's document alive
        self.to_source: Optional[pymupdf.Matrix] = None
        to_view = None
        if self.rotation:
            self._view_doc, page, self.to_source = _rotated_view(page, self.rotation)
            to_view = ~self.to_source
        self.page = page
        words, size, bold, color = _read_words(page, furniture, to_view)
        super().__init__(words)
        if words:
            self.wsize = np.asarray(size, dtype=np.float64)
            self.wbold = np.asarray(bold, bool)
            self.wcolor = np.asarray(color, np.int64)
        r = page.rect
        self.rect: BBox = (r.x0, r.y0, r.x1, r.y1)
        self.n_chars = sum(len(w[4]) for w in self.words)

    # ---- text layer -------------------------------------------------------
    @cached_property
    def text_quality(self) -> float:
        """Fraction of characters that are plausible text (1.0 = clean layer)."""
        if not self.n_chars:
            return 0.0
        bad = sum(len(_BAD_CHARS.findall(w[4])) for w in self.words)
        return 1.0 - bad / self.n_chars

    # ---- images -------------------------------------------------------------
    @cached_property
    def image_coverage(self) -> float:
        p = self.source_page
        r = p.rect
        pw, ph = r.width, r.height
        area = 0.0
        for info in p.get_image_info():
            x0, y0, x1, y1 = info["bbox"]
            area += max(0.0, min(x1, r.x1) - max(x0, r.x0)) * max(0.0, min(y1, r.y1) - max(y0, r.y0))
        return min(1.0, area / (pw * ph)) if pw * ph else 0.0

    @cached_property
    def page_type(self) -> str:
        has_text = self.n_chars >= 20
        if has_text and self.image_coverage < 0.5:
            return "digital"
        if has_text:
            return "mixed"
        return "scanned" if self.image_coverage >= 0.2 else "empty"

    @property
    def text_layer_ok(self) -> bool:
        return self.page_type in ("digital", "mixed") and self.text_quality >= 0.98

    # ---- vector rules -------------------------------------------------------
    @cached_property
    def _drawings(self) -> list[dict]:
        """The page's vector graphics, read once, in PyMuPDF's raw form: points and rectangles
        as plain tuples (``get_drawings`` converts every one of them to a Point / Rect object,
        which costs twice the read itself on vector-heavy pages)."""
        return self.page.get_cdrawings()

    @cached_property
    def _segments(self) -> np.ndarray:
        """Visible horizontal strokes as an (n, 3) array of [x0, x1, y], unmerged.
        Invisible (white) strokes are ignored: they are layout scaffolding, not rules."""
        segs = []
        for d in self._drawings:
            col = d.get("color") if d["type"] in ("s", "fs") else None
            fill = d.get("fill") if d["type"] in ("f", "fs") else None
            stroke_visible = col is not None and sum(col) < 2.85
            fill_visible = fill is not None and sum(fill) < 2.85
            for item in d["items"]:
                if item[0] == "l":
                    (px, py), (qx, qy) = item[1], item[2]
                    if stroke_visible and abs(py - qy) <= 1.0 and abs(px - qx) > 1:
                        segs.append((min(px, qx), max(px, qx), (py + qy) / 2))
                elif item[0] == "re":
                    x0, y0, x1, y1 = _normalized(item[1])
                    if y1 - y0 <= 3.0 and (fill_visible or stroke_visible):
                        # short pieces kept: rules are often drawn as joined segments
                        segs.append((x0, x1, y0))
                    elif stroke_visible and y1 - y0 > 3.0:  # stroked box edges
                        segs.extend([(x0, x1, y0), (x0, x1, y1)])
        return np.asarray(segs) if segs else np.zeros((0, 3))

    @cached_property
    def v_edges(self) -> np.ndarray:
        """Visible vertical separators as an (n, 3) array of [y0, y1, x]: vertical strokes,
        thin vertical rects, and the left/right edges of shaded cells (a shaded grid
        separates its columns by the edges of its cell fills)."""
        segs = []
        min_h = 0.9 * self.median_word_height  # shorter fills are text highlights, not cells
        for d in self._drawings:
            col = d.get("color") if d["type"] in ("s", "fs") else None
            fill = d.get("fill") if d["type"] in ("f", "fs") else None
            stroke = col is not None and sum(col) < 2.85
            shaded = fill is not None and sum(fill) < 2.85
            if not (stroke or shaded):
                continue
            for item in d["items"]:
                if item[0] == "l":
                    (px, py), (qx, qy) = item[1], item[2]
                    if stroke and abs(px - qx) <= 1.0 and abs(py - qy) > 2:
                        segs.append((min(py, qy), max(py, qy), (px + qx) / 2))
                elif item[0] == "re":
                    x0, y0, x1, y1 = _normalized(item[1])
                    if x1 - x0 <= 3.0 and y1 - y0 > 2:
                        segs.append((y0, y1, x0))
                    elif y1 - y0 > max(2.0, 0.0 if stroke else min_h):
                        segs.extend([(y0, y1, x0), (y0, y1, x1)])
        return np.asarray(segs) if segs else np.zeros((0, 3))

    def has_lattice(self, bbox: BBox, xs: Sequence[float]) -> bool:
        """True when visible vertical separators divide the table's columns: at least half
        of the interior column boundaries ``xs`` are drawn along most of the table height
        (strokes or shaded-cell edges, summed over rows)."""
        inner = list(xs)[1:-1]
        v = self.v_edges
        if not inner or not len(v):
            return False
        h = bbox[3] - bbox[1]
        seg_len = np.clip(np.minimum(v[:, 1], bbox[3]) - np.maximum(v[:, 0], bbox[1]), 0, None)
        hit = sum(1 for x in inner if seg_len[np.abs(v[:, 2] - x) <= 3.0].sum() >= 0.5 * h)
        return hit >= max(1, len(inner) / 2)

    @cached_property
    def fills(self) -> np.ndarray:
        """Visible filled shapes (cell shading, header pills) as an (n, 4) bbox array.
        Page-sized backgrounds and text highlights (strips shorter than a line of text, drawn
        behind single words) are excluded."""
        r = self.page.rect
        min_h = 0.9 * self.median_word_height
        out = []
        for d in self._drawings:
            f = d.get("fill") if d["type"] in ("f", "fs") else None
            x0, y0, x1, y1 = d["rect"]
            w, h = max(0, x1 - x0), max(0, y1 - y0)
            if f is None or sum(f) >= 2.95 or w < 4 or h < max(4.0, min_h):
                continue
            if w * h > 0.4 * r.width * r.height:
                continue
            out.append((x0, y0, x1, y1))
        return np.asarray(out) if out else np.zeros((0, 4))

    def merged_rules(self, join_gap: float = 1.0, min_len: float = 5.0) -> np.ndarray:
        """Collinear horizontal segments merged when the gap between them is at most
        ``join_gap`` (dotted rules and per-column rule pieces need a larger gap)."""
        segs = self._segments
        if not len(segs):
            return np.zeros((0, 3))
        rules = sorted(map(tuple, segs), key=lambda t: (round(t[2], 1), t[0]))
        merged = [list(rules[0])]
        for x0, x1, y in rules[1:]:
            m = merged[-1]
            if abs(y - m[2]) <= 0.8 and x0 <= m[1] + join_gap:
                m[1] = max(m[1], x1)
            else:
                merged.append([x0, x1, y])
        a = np.asarray(merged)
        return a[a[:, 1] - a[:, 0] > min_len]

    @cached_property
    def h_rules(self) -> np.ndarray:
        """Horizontal rules as an (n, 3) array of [x0, x1, y] (thin rects and stroked lines)."""
        return self.merged_rules()
