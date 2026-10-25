"""Phase 4 - canonical table representation.

JSON produced from these classes is the source of truth; HTML and CSV are
derived exports.  Cell text is stored exactly as printed (no translation, no
numeric normalisation): "△2,979", "△0" and "－" stay as they are.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

BBox = tuple[float, float, float, float]

SCHEMA_VERSION = "1.0"


def _r(b: Optional[BBox]) -> Optional[list[float]]:
    return None if b is None else [round(float(v), 2) for v in b]


def _bbox(v: Any) -> Optional[BBox]:
    return None if v is None else tuple(float(x) for x in v)  # type: ignore[return-value]


@dataclass(slots=True)
class Cell:
    row: int
    col: int
    text: str = ""
    rowspan: int = 1
    colspan: int = 1
    bbox: Optional[BBox] = None
    source: str = "native_pdf_text"  # native_pdf_text | ocr | empty
    is_header: bool = False
    confidence: Optional[float] = None

    def to_dict(self) -> dict:
        d = {
            "row": self.row, "col": self.col, "text": self.text,
            "rowspan": self.rowspan, "colspan": self.colspan,
            "bbox": _r(self.bbox), "source": self.source, "is_header": self.is_header,
        }
        if self.confidence is not None:
            d["confidence"] = round(self.confidence, 4)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Cell":
        return cls(
            row=int(d["row"]), col=int(d["col"]), text=d.get("text", ""),
            rowspan=int(d.get("rowspan", 1)), colspan=int(d.get("colspan", 1)),
            bbox=_bbox(d.get("bbox")), source=d.get("source", "native_pdf_text"),
            is_header=bool(d.get("is_header", False)), confidence=d.get("confidence"),
        )


@dataclass(slots=True)
class Reliability:
    score: float
    reliable: bool
    checks: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"score": round(self.score, 4), "reliable": self.reliable,
                "checks": {k: round(v, 4) for k, v in self.checks.items()},
                "reasons": self.reasons}

    @classmethod
    def from_dict(cls, d: dict) -> "Reliability":
        return cls(d["score"], d["reliable"], dict(d.get("checks", {})), list(d.get("reasons", [])))


@dataclass(slots=True)
class Table:
    page: int  # 1-based
    index: int  # order on the page, 1-based
    bbox: Optional[BBox]
    n_rows: int
    n_cols: int
    cells: list[Cell]
    header_rows: int = 0
    method: str = "native:lines"
    caption: list[str] = field(default_factory=list)  # text lines directly above, reading order
    title: Optional[str] = None
    unit: Optional[str] = None
    reliability: Optional[Reliability] = None
    needs_review: bool = False
    rotation: int = 0  # 90 / 270: printed rotated on the page (extracted in an upright frame)

    def occupancy(self) -> list[list[int]]:
        """n_rows x n_cols matrix of indices into ``cells`` (-1 = uncovered)."""
        occ = [[-1] * self.n_cols for _ in range(self.n_rows)]
        for i, c in enumerate(self.cells):
            for r in range(c.row, min(c.row + c.rowspan, self.n_rows)):
                for k in range(c.col, min(c.col + c.colspan, self.n_cols)):
                    occ[r][k] = i
        return occ

    def matrix(self, fill_spans: bool = False) -> list[list[str]]:
        """Dense text grid. Merged cells hold text at their anchor only, unless ``fill_spans``."""
        grid = [[""] * self.n_cols for _ in range(self.n_rows)]
        for c in self.cells:
            if fill_spans:
                for r in range(c.row, c.row + c.rowspan):
                    for k in range(c.col, c.col + c.colspan):
                        grid[r][k] = c.text
            else:
                grid[c.row][c.col] = c.text
        return grid

    def to_dict(self) -> dict:
        return {
            "page": self.page, "index": self.index, "bbox": _r(self.bbox),
            "n_rows": self.n_rows, "n_cols": self.n_cols, "header_rows": self.header_rows,
            "method": self.method, "title": self.title, "unit": self.unit, "caption": self.caption,
            "needs_review": self.needs_review, "rotation": self.rotation,
            "reliability": self.reliability.to_dict() if self.reliability else None,
            "cells": [c.to_dict() for c in self.cells],
        }

    @classmethod
    def from_dict(cls, d: dict, page: Optional[int] = None) -> "Table":
        """Accepts the canonical form, or the compact annotation form with a ``rows`` text grid
        (plus optional ``spans: [[row, col, rowspan, colspan], ...]``)."""
        page = int(d.get("page", page or 1))
        header_rows = int(d.get("header_rows", 0))
        if "cells" in d:
            cells = [Cell.from_dict(c) for c in d["cells"]]
            n_rows = int(d.get("n_rows") or max(c.row + c.rowspan for c in cells))
            n_cols = int(d.get("n_cols") or max(c.col + c.colspan for c in cells))
        else:
            rows = d["rows"]
            n_rows, n_cols = len(rows), max(len(r) for r in rows)
            spans = {(s[0], s[1]): (s[2], s[3]) for s in d.get("spans", [])}
            covered: set[tuple[int, int]] = set()
            for (r, c), (rs, cs) in spans.items():
                covered |= {(rr, cc) for rr in range(r, r + rs) for cc in range(c, c + cs)} - {(r, c)}
            cells = []
            for r, row in enumerate(rows):
                for c in range(n_cols):
                    if (r, c) in covered:
                        continue
                    rs, cs = spans.get((r, c), (1, 1))
                    text = row[c] if c < len(row) else ""
                    cells.append(Cell(r, c, text, rs, cs, None, "annotation", r < header_rows))
        return cls(
            page=page, index=int(d.get("index", 1)), bbox=_bbox(d.get("bbox")),
            n_rows=n_rows, n_cols=n_cols, cells=cells, header_rows=header_rows,
            method=d.get("method", "annotation"), caption=list(d.get("caption", [])),
            title=d.get("title"), unit=d.get("unit"),
            reliability=Reliability.from_dict(d["reliability"]) if d.get("reliability") else None,
            needs_review=bool(d.get("needs_review", False)), rotation=int(d.get("rotation", 0)),
        )


@dataclass(slots=True)
class PageResult:
    page: int
    width: float
    height: float
    page_type: str  # digital | scanned | mixed | empty
    route: str  # native | visual | native+visual | none
    tables: list[Table] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    rotation: int = 0  # reading frame used for extraction (see inspection.PageInfo)

    def to_dict(self) -> dict:
        return {"page": self.page, "width": round(self.width, 2), "height": round(self.height, 2),
                "page_type": self.page_type, "route": self.route, "rotation": self.rotation,
                "warnings": self.warnings,
                "elapsed_ms": round(self.elapsed_ms, 1), "tables": [t.to_dict() for t in self.tables]}

    @classmethod
    def from_dict(cls, d: dict) -> "PageResult":
        p = int(d["page"])
        return cls(p, float(d.get("width", 0)), float(d.get("height", 0)), d.get("page_type", "digital"),
                   d.get("route", "native"), [Table.from_dict(t, p) for t in d.get("tables", [])],
                   list(d.get("warnings", [])), float(d.get("elapsed_ms", 0)), int(d.get("rotation", 0)))


@dataclass(slots=True)
class DocumentResult:
    source: str
    pages: list[PageResult]
    meta: dict = field(default_factory=dict)

    @property
    def tables(self) -> list[Table]:
        return [t for p in self.pages for t in p.tables]

    def to_dict(self) -> dict:
        return {"schema_version": SCHEMA_VERSION, "source": self.source, "n_pages": len(self.pages),
                "n_tables": len(self.tables), "meta": self.meta,
                "pages": [p.to_dict() for p in self.pages]}

    @classmethod
    def from_dict(cls, d: dict) -> "DocumentResult":
        if "pages" not in d and "tables" in d:  # compact annotation file: flat list of tables
            by_page: dict[int, list[Table]] = {}
            for t in d["tables"]:
                tab = Table.from_dict(t)
                by_page.setdefault(tab.page, []).append(tab)
            n = max([int(d.get("n_pages", 0)), *by_page]) if by_page else int(d.get("n_pages", 0))
            pages = [PageResult(p, 0, 0, "digital", "annotation", by_page.get(p, [])) for p in range(1, n + 1)]
            meta = dict(d.get("meta", {}))
            if "annotated_pages" in d:  # ground truth covering a sample of pages only
                meta["annotated_pages"] = [int(p) for p in d["annotated_pages"]]
            if "splits" in d:  # named page subsets reported separately (e.g. held-out vs dev)
                meta["splits"] = {k: [int(p) for p in v] for k, v in d["splits"].items()}
            return cls(d.get("source", ""), pages, meta)
        return cls(d.get("source", ""), [PageResult.from_dict(p) for p in d["pages"]], d.get("meta", {}))
