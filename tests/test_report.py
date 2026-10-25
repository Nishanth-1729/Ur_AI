"""PDF report: real tables, repeated headers, review banners, CJK text."""
import numpy as np
import pymupdf
import pytest

from pdftable import Config, process
from pdftable.export import export
from pdftable.models import Cell, DocumentResult, PageResult, Reliability, Table

from .conftest import SAMPLE


def _long_table(n_rows=90):
    cells = [Cell(0, 0, "Particulars", is_header=True), Cell(0, 1, "FY24", is_header=True),
             Cell(0, 2, "FY23", is_header=True)]
    for r in range(1, n_rows):
        cells += [Cell(r, 0, f"Line item {r} with a fairly long description that has to wrap in its cell"),
                  Cell(r, 1, f"{r * 1000:,}"), Cell(r, 2, f"({r})")]
    return Table(7, 1, (40, 100, 550, 700), n_rows, 3, cells, header_rows=1, method="native:borderless",
                 title="Long statement", unit="(INR crores)", reliability=Reliability(1.0, True))


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    flagged = Table(9, 1, (40, 100, 300, 200), 3, 2,
                    [Cell(0, 0, "A", is_header=True), Cell(0, 1, "B", is_header=True),
                     Cell(1, 0, "x"), Cell(1, 1, "1"), Cell(2, 0, "y"), Cell(2, 1, "2")], header_rows=1,
                    reliability=Reliability(0.4, False, {}, ["30% of cells hold several numbers (columns merged)"]),
                    needs_review=True)
    doc = DocumentResult("synthetic.pdf", [PageResult(7, 595, 842, "digital", "native", [_long_table()]),
                                           PageResult(9, 595, 842, "digital", "native", [flagged])])
    out = tmp_path_factory.mktemp("report")
    (path,) = export(doc, out, {"pdf"})
    return pymupdf.open(path)


def test_report_has_cover_index_and_all_tables(report):
    cover = report[0].get_text()
    assert "Extracted tables" in cover and "2 table(s) extracted" in cover and "1 flagged" in cover
    toc = [t[1] for t in report.get_toc()]
    assert toc[0].startswith("p7 t1") and toc[1].startswith("p9 t1") and toc[1].endswith("[REVIEW]")
    text = "".join(p.get_text() for p in report)
    assert "Line item 89" in text and "89,000" in text and "(89)" in text


def test_long_table_continues_with_repeated_header(report):
    start = report.get_toc()[0][2] - 1
    pages = [report[i].get_text() for i in range(start, report.page_count)]
    cont = [t for t in pages if "Source page 7 · Table 1 (continued)" in t]
    assert cont, "a 90-row table must continue on another page"
    assert all("Particulars" in t and "FY24" in t for t in cont)  # header repeated


def test_cells_are_drawn_as_boxes(report):
    start = report.get_toc()[0][2] - 1
    rects = [d for d in report[start].get_drawings() if d["items"] and d["items"][0][0] == "re"]
    assert len(rects) >= 3 * 20  # one box per cell
    pix = report[start].get_pixmap(dpi=40)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
    assert (img < 200).mean() > 0.01  # the rendered page is not blank


def test_review_banner_with_reasons(report):
    start = report.get_toc()[1][2] - 1
    text = report[start].get_text()
    assert "NEEDS REVIEW" in text and "columns merged" in text


def test_report_of_japanese_sample_keeps_cjk_text(tmp_path):
    doc = process(SAMPLE, Config(fallback="never"))
    (path,) = export(doc, tmp_path, {"pdf"})
    text = "".join(p.get_text() for p in pymupdf.open(path))
    for s in ("現金及び預金", "2,002,429", "△2,979", "（単位：千円）"):
        assert s in text
