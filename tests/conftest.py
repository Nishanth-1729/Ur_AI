from pathlib import Path

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "balance_sheet_after_correction.pdf"
SAMPLE_GT = ROOT / "ground_truth" / "balance_sheet_after_correction.json"


def _text(page, x, y, s, size=10):
    page.insert_text((x, y), s, fontname="japan", fontsize=size)


def _ruled(page, xs, ys, skip_vertical=()):
    """Draw a lattice; ``skip_vertical`` = {(x, row)} vertical segments to omit (merges)."""
    for y in ys:
        page.draw_line((xs[0], y), (xs[-1], y), width=0.6)
    for x in xs:
        for r in range(len(ys) - 1):
            if (x, r) not in skip_vertical:
                page.draw_line((x, ys[r]), (x, ys[r + 1]), width=0.6)


@pytest.fixture(scope="session")
def synthetic_pdf(tmp_path_factory) -> Path:
    doc = pymupdf.open()

    # page 1: two ruled tables; the first has a colspan header and blank / △ / － cells
    p = doc.new_page(width=595, height=842)
    _text(p, 50, 90, "Sales summary", 12)
    xs, ys = [50, 200, 300, 400], [100, 120, 140, 160, 180]
    _ruled(p, xs, ys, skip_vertical={(300, 0)})
    rows = [["Item", "FY2025", None], ["Sales", "100", "200"], ["Cost", "△30", "－"], ["Profit", "70", ""]]
    for r, row in enumerate(rows):
        for c, s in enumerate(row):
            if s:
                _text(p, xs[c] + 4, ys[r] + 14, s)
    _text(p, 50, 292, "Table 2 (unit: pcs)")
    xs2, ys2 = [50, 150, 250], [300, 320, 340, 360]
    _ruled(p, xs2, ys2)
    for r, row in enumerate([["Name", "Qty"], ["A", "1"], ["B", "2"]]):
        for c, s in enumerate(row):
            _text(p, xs2[c] + 4, ys2[r] + 14, s)

    # page 2: prose only - must yield no table
    p = doc.new_page(width=595, height=842)
    p.insert_textbox(pymupdf.Rect(50, 80, 545, 400),
                     "This page contains ordinary running text and no tabular content. " * 12,
                     fontname="helv", fontsize=11)

    path = tmp_path_factory.mktemp("pdf") / "synthetic.pdf"
    doc.save(path)
    return path
