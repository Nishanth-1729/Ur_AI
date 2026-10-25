"""Visual fallback (PP-StructureV3). Slow: ~30-40 s per page on CPU plus a one-time
model download, so it only runs with PDFTABLE_TEST_VISUAL=1."""
import os

import pymupdf
import pytest

from pdftable import Config, process
from pdftable.evaluate import evaluate_pairs
from pdftable.export import export
from pdftable.visual import available

from .conftest import SAMPLE, SAMPLE_GT

pytestmark = pytest.mark.skipif(not (available() and os.environ.get("PDFTABLE_TEST_VISUAL") == "1"),
                                reason="set PDFTABLE_TEST_VISUAL=1 with paddleocr installed")


def _summary(doc, tmp_path):
    export(doc, tmp_path, {"json"})
    return evaluate_pairs([(SAMPLE_GT, tmp_path / f"{SAMPLE.stem}.json")])["summary"]


def test_forced_visual_on_digital_page_is_exact(tmp_path):
    doc = process(SAMPLE, Config(fallback="always"))
    assert doc.pages[0].route == "visual"
    s = _summary(doc, tmp_path)
    assert s["grits_top"] == 1.0 and s["strict_exact_acc"] == 1.0


def test_scanned_page_uses_ocr(tmp_path):
    src = pymupdf.open(SAMPLE)
    scan = pymupdf.open()
    pix = src[0].get_pixmap(dpi=150, colorspace=pymupdf.csGRAY)
    scan.new_page(width=src[0].rect.width, height=src[0].rect.height).insert_image(src[0].rect, pixmap=pix)
    path = tmp_path / SAMPLE.name
    scan.save(path)
    doc = process(path, Config())
    assert doc.pages[0].page_type == "scanned" and doc.pages[0].route == "visual"
    t = doc.tables[0]
    assert (t.n_rows, t.n_cols) == (44, 3)
    assert all(c.source in ("ocr", "empty") for c in t.cells)
    assert any(c.confidence is not None for c in t.cells)
    s = _summary(doc, tmp_path / "out")
    assert s["grits_top"] == 1.0 and s["cer"] < 0.1  # structure exact; OCR content measured by CER
