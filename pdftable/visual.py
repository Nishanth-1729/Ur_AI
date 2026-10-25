"""Phase 2 (fallback) - visual table recognition with PaddleOCR PP-StructureV3.

Only pages that fail native validation reach this module.  The model supplies
table regions and the cell lattice; cell *text* still comes from the PDF text
layer whenever that layer is trustworthy (OCR is lossy - e.g. 産 -> 產,
３ -> 3 on the sample), and from OCR boxes otherwise (scanned pages), with the
OCR confidence kept per cell.

The heavy model is created lazily, once per process.
"""
from __future__ import annotations

import importlib.util
import inspect
import logging
import os
import warnings
from html.parser import HTMLParser
from typing import Optional

import numpy as np
import pymupdf

from .inspection import PageInfo, WordSet
from .models import Cell, Table
from .reconstruct import FillStats, Grid, assemble, build_grid


def available() -> bool:
    return importlib.util.find_spec("paddleocr") is not None


class _HTMLTable(HTMLParser):
    """Minimal <table> parser -> rows of [text, rowspan, colspan, is_th]."""

    def __init__(self):
        super().__init__()
        self.rows: list[list[list]] = []
        self._cell: Optional[list] = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th") and self.rows is not None:
            a = dict(attrs)
            self._cell = ["", int(a.get("rowspan") or 1), int(a.get("colspan") or 1), tag == "th"]
        elif tag == "br" and self._cell is not None:
            self._cell[0] += "\n"

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            if not self.rows:
                self.rows.append([])
            self._cell[0] = self._cell[0].strip()
            self.rows[-1].append(self._cell)
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell[0] += data


def html_to_cells(html: str) -> tuple[list[Cell], int, int]:
    """Lay HTML cells (with spans) onto a grid -> (cells, n_rows, n_cols)."""
    p = _HTMLTable()
    p.feed(html)
    taken: set[tuple[int, int]] = set()
    cells: list[Cell] = []
    for r, row in enumerate(p.rows):
        c = 0
        for text, rs, cs, th in row:
            while (r, c) in taken:
                c += 1
            cells.append(Cell(r, c, text, rs, cs, None, "ocr" if text else "empty", th))
            taken |= {(rr, cc) for rr in range(r, r + rs) for cc in range(c, c + cs)}
            c += cs
    n_rows = max((rr for rr, _ in taken), default=-1) + 1
    n_cols = max((cc for _, cc in taken), default=-1) + 1
    for r in range(n_rows):
        for c in range(n_cols):
            if (r, c) not in taken:
                cells.append(Cell(r, c, "", source="empty"))
    cells.sort(key=lambda x: (x.row, x.col))
    return cells, n_rows, n_cols


class VisualRecognizer:
    _shared: dict = {}

    def __init__(self, dpi: int = 200, orientation: bool = True, device: Optional[str] = None):
        self.dpi = dpi
        key = (orientation, device)
        if key not in self._shared:
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            quiet = os.environ.get("PDFTABLE_MODEL_LOGS") != "1"
            if quiet:  # start-up chatter, not problems (set PDFTABLE_MODEL_LOGS=1 to see it)
                os.environ.setdefault("GLOG_minloglevel", "2")  # Paddle's C++ log: errors only
                warnings.filterwarnings("ignore", message="No ccache found")
            from paddleocr import PPStructureV3

            if quiet:  # PaddleX logs every model it loads at INFO level
                for name in ("paddlex", "paddleocr", "ppocr"):
                    logging.getLogger(name).setLevel(logging.WARNING)

            kw = dict(use_doc_orientation_classify=orientation, use_doc_unwarping=False,
                      use_textline_orientation=False, use_seal_recognition=False,
                      use_formula_recognition=False, use_chart_recognition=False,
                      use_region_detection=False, device=device)
            params = inspect.signature(PPStructureV3.__init__).parameters
            if not any(p.kind is p.VAR_KEYWORD for p in params.values()):
                kw = {k: v for k, v in kw.items() if k in params}
            self._shared[key] = PPStructureV3(**{k: v for k, v in kw.items() if v is not None})
        self.pipeline = self._shared[key]

    def _render(self, page: pymupdf.Page) -> np.ndarray:
        pix = page.get_pixmap(dpi=self.dpi, colorspace=pymupdf.csRGB, alpha=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, 3)
        return np.ascontiguousarray(img[:, :, ::-1])  # RGB -> BGR (OpenCV convention)

    def recognize(self, info: PageInfo) -> tuple[list[tuple[Table, FillStats]], list[str]]:
        page = info.page
        res = next(iter(self.pipeline.predict(self._render(page))))
        data = res.json
        data = data.get("res", data)
        warnings: list[str] = []
        angle = (data.get("doc_preprocessor_res") or {}).get("angle", 0)
        rotated = angle not in (None, 0, -1)
        if rotated:
            warnings.append(f"visual: page orientation corrected by {angle} deg; cell coordinates "
                            "are in the corrected image frame and text comes from OCR")

        scale = self.dpi / 72.0
        derot = page.derotation_matrix if page.rotation and not rotated else None

        def to_pdf(boxes) -> list[tuple]:
            out = []
            for b in boxes:
                x0, y0, x1, y1 = (float(v) / scale for v in b[:4])
                if derot is not None:
                    r = pymupdf.Rect(x0, y0, x1, y1) * derot
                    x0, y0, x1, y1 = r.x0, r.y0, r.x1, r.y1
                out.append((x0, y0, x1, y1))
            return out

        use_native = info.text_layer_ok and not rotated
        # page-level OCR lines let scanned pages keep titles / units above the table
        page_ocr = None
        if not use_native and not rotated:
            o = data.get("overall_ocr_res") or {}
            page_ocr = WordSet([(*bx, t, 0, 0, 0) for bx, t in zip(to_pdf(o.get("rec_boxes", [])), o.get("rec_texts", []))],
                               o.get("rec_scores") or None)
        tables: list[tuple[Table, FillStats]] = []
        for tr in data.get("table_res_list") or []:
            ocr = tr.get("table_ocr_pred") or {}
            ocr_words = WordSet([(*b, t, 0, 0, 0) for b, t in zip(to_pdf(ocr.get("rec_boxes", [])),
                                                                  ocr.get("rec_texts", []))],
                                ocr.get("rec_scores") or None)
            boxes = to_pdf(tr.get("cell_box_list") or [])
            html_cells, hr, hc = html_to_cells(tr.get("pred_html", ""))
            grid = _model_grid(boxes, info if use_native else ocr_words) if boxes else None
            if grid is not None:
                table, stats = assemble(
                    info, grid, "visual:pp-structurev3", info.rect[1],
                    words=None if use_native else ocr_words,
                    source="native_pdf_text" if use_native else "ocr",
                    context=True if use_native else page_ocr)
                if (hr, hc) != grid.shape:
                    warnings.append(f"visual: cell boxes give {grid.shape[0]}x{grid.shape[1]} grid, "
                                    f"HTML structure {hr}x{hc}")
            elif html_cells:  # no cell geometry: keep the model's HTML structure and OCR text
                table = Table(info.number, 0, None, hr, hc, html_cells,
                              header_rows=sum(1 for r in range(hr) if all(c.is_header for c in html_cells if c.row == r)),
                              method="visual:pp-structurev3-html")
                for c in table.cells:
                    c.is_header = c.row < table.header_rows
                stats = FillStats(sum(1 for c in html_cells if c.text), 0, _mean(ocr.get("rec_scores")))
            else:
                continue
            if rotated:
                table.bbox = None
                for c in table.cells:
                    c.bbox = None
            tables.append((table, stats))
        return tables, warnings


def _mean(xs) -> Optional[float]:
    return float(np.mean(xs)) if xs else None


def _model_grid(boxes: list[tuple], words: WordSet) -> Optional[Grid]:
    """Model boxes are noisy by a few pixels: tolerances scale with the cell sizes."""
    b = np.asarray(boxes)
    h = np.median(b[:, 3] - b[:, 1])
    w = np.median(b[:, 2] - b[:, 0])
    return build_grid(boxes, tol=float(min(8.0, max(2.0, 0.3 * w))), ytol=float(min(6.0, max(2.0, 0.35 * h))),
                      noisy=True, words=words)
