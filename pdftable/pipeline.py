"""Page-level routing: native fast path -> reliability gate -> visual fallback.

Stage 0 learns the document's page furniture (running headers/footers) from a
sample of pages.  Stages 0 and 1 (native) are CPU-bound and independent per page,
so they run in one process pool: pages are handed out in small tasks as workers
free up (heavy pages cluster in some sections, so fixed per-worker blocks would
leave most workers idle), each worker opens a PDF once and finds the furniture of
its own pages.  Stage 2 (visual) runs in the main process for the few pages that
need it, so the heavy model is loaded exactly once.  Tables are built in each
page's reading frame (rotated pages are read upright) and mapped back to page
coordinates at the end.
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Iterable, Iterator, Optional

import numpy as np
import pymupdf

from . import visual
from .inspection import Furniture, PageInfo, band_lines, frame_to_page, furniture_sample, map_bbox
from .models import BBox, DocumentResult, PageResult, Table
from .native import covers, extract_native, overlap_ratio
from .reliability import Thresholds, assess
from .textgrid import build_lines, classify


@dataclass(frozen=True)
class Config:
    fallback: str = "auto"  # auto | always | never
    dpi: int = 200
    workers: int = 0  # 0 = one per CPU
    orientation: bool = True
    device: Optional[str] = None
    thresholds: Thresholds = field(default_factory=Thresholds)
    pages: Optional[tuple[int, ...]] = None  # 1-based pages to process (None = all)
    furniture: bool = True  # drop running headers / footers / page numbers


@dataclass(slots=True)
class _Native:
    result: PageResult
    pending: list[Table]  # native candidates that failed the checks
    reason: Optional[str]  # why the page needs the visual fallback (None = it does not)
    furniture: list[BBox]
    t: float


def stray_value_rows(info: PageInfo, tables: list[Table], min_values: int = 3) -> int:
    """Lines with ``min_values``+ numeric cells that no detected table covers (a missed table)."""
    idx = np.arange(len(info.words))
    for t in tables:
        b = t.bbox
        inside = (info.cx >= b[0] - 1) & (info.cx <= b[2] + 1) & (info.cy >= b[1] - 1) & (info.cy <= b[3] + 1)
        idx = idx[~inside[idx]]
    if len(idx) < min_values:
        return 0
    content_w = float(info.wbox[:, 2].max() - info.wbox[:, 0].min())
    n = 0
    for ln in build_lines(info, idx):
        if classify(ln, info.body_size, content_w) == "value" and len(ln.values_right()) >= min_values:
            n += 1
    return n


def _native_page(page: pymupdf.Page, cfg: Config, furniture: Optional[Furniture]) -> _Native:
    t0 = perf_counter()
    furn = furniture.boxes(page) if furniture else []
    info = PageInfo(page, furn, rotation=None if cfg.orientation else 0)
    src = page.rect
    pr = PageResult(info.number, src.width, src.height, info.page_type, "native", rotation=info.rotation)
    reason, pending = None, []
    if cfg.fallback == "always":
        reason = "forced (--fallback always)"
    elif info.page_type == "empty":
        pr.route = "none"
    elif info.page_type == "scanned":
        reason = "scanned page (no text layer)"
    elif info.text_quality < cfg.thresholds.min_text_quality:
        reason = f"unreliable text layer (quality {info.text_quality:.3f})"
    else:
        nat = extract_native(info, cfg.thresholds)
        pr.tables, pending = nat.reliable, nat.unreliable
        if pending:
            reason = "native table failed checks: " + "; ".join(
                dict.fromkeys(r for t in pending for r in t.reliability.reasons))
        elif not nat.reliable and info.page_type == "mixed":
            reason = "image-dominated page without native table"
        else:
            stray = stray_value_rows(info, nat.reliable)
            if stray >= 3:
                reason = f"{stray} numeric rows outside detected tables"
    return _Native(pr, pending, reason, furn, perf_counter() - t0)


def _native_pages(doc: pymupdf.Document, pages: list[int], cfg: Config,
                  furniture: Optional[Furniture]) -> list[_Native]:
    return [_native_page(doc[i], cfg, furniture) for i in pages]


_WORKER_DOCS: dict[str, pymupdf.Document] = {}  # per worker process: each PDF opened once


def _worker_doc(path: str) -> pymupdf.Document:
    doc = _WORKER_DOCS.get(path)
    if doc is None:
        doc = _WORKER_DOCS[path] = pymupdf.open(path)
    return doc


def _band_task(path: str, pages: list[int]) -> list[list[tuple[str, BBox]]]:
    doc = _worker_doc(path)
    return [band_lines(doc[i]) for i in pages]


def _native_task(path: str, pages: list[int], cfg: Config, furniture: Optional[Furniture]) -> list[_Native]:
    return _native_pages(_worker_doc(path), pages, cfg, furniture)


class _Visual:
    """Lazy holder so the model is only built if some page actually needs it."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._rec = None
        self.ok = cfg.fallback != "never" and visual.available()

    def get(self) -> "visual.VisualRecognizer":
        if self._rec is None:
            self._rec = visual.VisualRecognizer(self.cfg.dpi, self.cfg.orientation, self.cfg.device)
        return self._rec


def _apply_visual(doc: pymupdf.Document, n: _Native, vis: _Visual, cfg: Config) -> None:
    pr = n.result
    t0 = perf_counter()

    def keep_native(why: str) -> None:
        for t in n.pending:
            t.needs_review = True
        pr.tables += n.pending
        if n.pending:
            pr.warnings.append(f"visual fallback {why}; native candidates kept and flagged needs_review")
        else:
            pr.warnings.append(f"visual fallback {why}")

    if not vis.ok:
        keep_native("disabled (--fallback never)" if cfg.fallback == "never"
                    else "unavailable (pip install -r requirements-visual.txt)")
        return
    info = PageInfo(doc[pr.page - 1], n.furniture, rotation=pr.rotation)
    try:
        found, warnings = vis.get().recognize(info)
    except Exception as e:  # model download / runtime failure must not lose the native result
        keep_native(f"failed: {type(e).__name__}: {e}")
        return
    pr.warnings += warnings
    keep = pr.tables
    added: list[Table] = []
    for t, stats in found:
        if t.bbox is not None and any(covers(k.bbox, t.bbox) > 0.6 for k in keep):
            continue  # a reliable native table already covers (most of) this region
        ocr = any(c.source == "ocr" for c in t.cells)
        quality = (stats.mean_score if stats.mean_score is not None else 0.0) if ocr else info.text_quality
        t.reliability = assess(t, stats, quality, cfg.thresholds, ocr=ocr, page_rect=info.rect)
        t.needs_review = not t.reliability.reliable
        added.append(t)
    for u in n.pending:
        if not any(v.bbox is not None and overlap_ratio(u.bbox, v.bbox) > 0.5 for v in added):
            pr.warnings.append(f"native candidate {[round(x, 1) for x in u.bbox]} dropped: failed checks "
                               f"({'; '.join(u.reliability.reasons)}) and not confirmed by the visual model")
    pr.tables = keep + added
    pr.route = "native+visual" if keep and added else ("visual" if added or not keep else "native")
    pr.elapsed_ms += (perf_counter() - t0) * 1000


def _finalize(pr: PageResult) -> None:
    pr.tables.sort(key=lambda t: (t.bbox[1], t.bbox[0]) if t.bbox else (math.inf, 0))
    m = frame_to_page(pr.rotation, pr.width, pr.height)
    for i, t in enumerate(pr.tables, 1):
        t.index = i
        t.page = pr.page
        if m is not None:  # reading frame -> page coordinates
            t.rotation = pr.rotation
            t.bbox = map_bbox(t.bbox, m)
            for c in t.cells:
                c.bbox = map_bbox(c.bbox, m)


def _expand(inputs: Iterable[str | Path]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, inputs):
        out += sorted(p.rglob("*.pdf")) if p.is_dir() else [p]
    return out


def run(inputs: Iterable[str | Path], cfg: Config = Config()) -> Iterator[DocumentResult]:
    """Process many PDFs; yields one DocumentResult per input, in input order."""
    t0 = perf_counter()
    paths = _expand(inputs)
    selected: dict[Path, list[int]] = {}
    learn: dict[Path, list[int]] = {}  # documents whose furniture is learnt -> sampled pages
    for p in paths:
        with pymupdf.open(p) as d:
            n = d.page_count
            selected[p] = [x - 1 for x in cfg.pages if 1 <= x <= n] if cfg.pages else list(range(n))
            if cfg.furniture and n > 1:
                learn[p] = furniture_sample(n)
    total = sum(len(v) for v in selected.values())
    workers = min(cfg.workers or os.cpu_count() or 1, max(1, total))

    # ---- stages 0 + 1: furniture, then native extraction (parallel) -----------------
    furniture: dict[Path, Optional[Furniture]] = dict.fromkeys(paths)
    natives: dict[Path, list[_Native]] = {p: [] for p in paths}
    if workers > 1 and total >= 4:
        size = max(1, min(4, total // (8 * workers)))  # small tasks: idle workers pick up the next one
        with ProcessPoolExecutor(workers) as ex:
            sampled = {p: [ex.submit(_band_task, str(p), pages[s:s + 4]) for s in range(0, len(pages), 4)]
                       for p, pages in learn.items()}
            for p, fs in sampled.items():
                furniture[p] = Furniture.from_lines([lines for f in fs for lines in f.result()])
            futures = [(p, ex.submit(_native_task, str(p), sel[s:s + size], cfg, furniture[p]))
                       for p, sel in selected.items() for s in range(0, len(sel), size)]
            for p, f in futures:  # submitted in page order, collected in page order
                natives[p] += f.result()
    else:
        for p, sel in selected.items():
            with pymupdf.open(p) as doc:
                furniture[p] = Furniture.learn(doc) if p in learn else None
                natives[p] = _native_pages(doc, sel, cfg, furniture[p])

    # ---- stage 2: visual fallback for flagged pages (sequential, model loaded once)
    vis = _Visual(cfg)
    for p in paths:
        ns = natives[p]
        with pymupdf.open(p) as doc:
            for n in ns:
                n.result.elapsed_ms = n.t * 1000
                if n.reason:
                    n.result.warnings.insert(0, f"routed to visual fallback: {n.reason}")
                    _apply_visual(doc, n, vis, cfg)
                _finalize(n.result)
        meta = {
            "pymupdf": pymupdf.VersionBind,
            "visual_model": "PaddleOCR PP-StructureV3" if any("visual" in n.result.route for n in ns) else None,
            "config": asdict(cfg),
            "elapsed_s": round(perf_counter() - t0, 3),
        }
        yield DocumentResult(str(p), [n.result for n in ns], meta)


def process(path: str | Path, cfg: Config = Config()) -> DocumentResult:
    return next(run([path], cfg))
