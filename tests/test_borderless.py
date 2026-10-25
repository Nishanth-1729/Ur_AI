"""Borderless engine: hairline-ruled tables, rotated pages, and pages that must give no table."""
import pymupdf
import pytest

from pdftable import Config, process


def _t(page, x, y, s, size=10, bold=False, right=False, color=(0, 0, 0)):
    font = "hebo" if bold else "helv"
    if right:
        x -= pymupdf.get_text_length(s, fontname=font, fontsize=size)
    page.insert_text((x, y), s, fontname=font, fontsize=size, color=color)


def _note_page(page, x0=50, x1=545):
    """A financial-note table: title, unit, two-line header, hairline rule under every row,
    a section row, a label wrapped over two lines, and a footnote."""
    _t(page, x0, 80, "12 Other financial assets", 14, bold=True)
    _t(page, x1, 100, "(INR crores)", 10, right=True)
    _t(page, x0, 120, "Particulars", 10, bold=True)
    for x, year in ((440, "2024"), (x1, "2023")):
        _t(page, x, 120, "As at", 10, bold=True, right=True)
        _t(page, x, 133, f"March 31, {year}", 10, bold=True, right=True)
    rows = [("Current", None, None), ("Security deposits", "1,234", "987"),
            ("Interest accrued on fixed deposits with banks and", "56", "(12)"),
            ("financial institutions", None, None), ("Other receivables", "-", "3"),
            ("Total other financial assets", "1,290", "978")]
    y = 160
    for label, a, b in rows:
        wrapped = label == "financial institutions"
        if not wrapped:
            page.draw_line((x0, y - 12), (x1, y - 12), color=(0.6, 0.6, 0.6), width=0.4)
        _t(page, x0 + (0 if label in ("Current",) else 0), y, label, 10, bold=label.startswith(("Total", "Current")))
        if a is not None:
            _t(page, 440, y, a, 10, right=True)
            _t(page, x1, y, b, 10, right=True)
        y += 12 if wrapped or label.startswith("Interest") else 16
    page.draw_line((x0, y - 12), (x1, y - 12), color=(0.6, 0.6, 0.6), width=0.4)
    _t(page, x0, y + 10, "* Security deposits are refundable on termination of the lease agreements entered "
                         "into by the Company.", 8)


@pytest.fixture(scope="module")
def borderless_doc(tmp_path_factory):
    doc = pymupdf.open()
    _note_page(doc.new_page(width=595, height=842))

    # a landscape table printed rotated on a portrait page
    land = pymupdf.open()
    _note_page(land.new_page(width=842, height=595), x1=780)
    page = doc.new_page(width=595, height=842)
    page.show_pdf_page(page.rect, land, 0, rotate=90)

    # two columns of running text: no table
    page = doc.new_page(width=595, height=842)
    para = ("The Group recognises revenue when control of the services is transferred to the customer "
            "at an amount that reflects the consideration expected in exchange for those services. ") * 6
    page.insert_textbox(pymupdf.Rect(50, 80, 290, 760), para, fontname="helv", fontsize=10, align=3)
    page.insert_textbox(pymupdf.Rect(310, 80, 545, 760), para, fontname="helv", fontsize=10, align=3)

    # a bar chart with data labels: no table
    page = doc.new_page(width=595, height=842)
    _t(page, 50, 80, "Revenue grew 27% YoY", 14, bold=True)
    _t(page, 50, 100, "INR crore", 9)
    for k, (x, h, v, fy) in enumerate([(100, 120, "4,763", "FY22"), (250, 160, "6,147", "FY23"),
                                       (400, 200, "7,792", "FY24")]):
        page.draw_rect(pymupdf.Rect(x, 400 - h, x + 50, 400), color=None, fill=(0.85, 0.2, 0.25))
        _t(page, x + 8, 400 - h - 6, v, 10)
        _t(page, x + 12, 415, fy, 9)
        _t(page, x + 8, 430, ["109%", "57%", "56%"][k], 9)
    path = tmp_path_factory.mktemp("borderless") / "borderless.pdf"
    doc.save(path)
    return process(path, Config(fallback="never", workers=1))


def _check_note_table(t):
    m = t.matrix()
    assert (t.n_cols, t.header_rows) == (3, 1)
    assert m[0] == ["Particulars", "As at\nMarch 31, 2024", "As at\nMarch 31, 2023"]
    body = m[1:]
    assert body[0] == ["Current", "", ""]  # section row kept, values empty
    assert body[1] == ["Security deposits", "1,234", "987"]
    # a wrapped label is one cell, with its values on the same row
    assert body[2] == ["Interest accrued on fixed deposits with banks and\nfinancial institutions", "56", "(12)"]
    assert body[3] == ["Other receivables", "-", "3"]
    assert body[4] == ["Total other financial assets", "1,290", "978"]
    assert len(body) == 5  # the footnote is not a row
    assert t.title == "12 Other financial assets" and t.unit == "(INR crores)"
    assert t.method == "native:borderless" and t.reliability.reliable


def test_hairline_table_rows_header_title_unit(borderless_doc):
    (t,) = borderless_doc.pages[0].tables
    assert t.rotation == 0
    _check_note_table(t)


def test_rotated_page_is_read_upright(borderless_doc):
    page = borderless_doc.pages[1]
    assert page.rotation == 90
    (t,) = page.tables
    assert t.rotation == 90
    _check_note_table(t)
    # coordinates are reported on the original (portrait) page
    x0, y0, x1, y1 = t.bbox
    assert 0 <= x0 < x1 <= page.width and 0 <= y0 < y1 <= page.height
    assert y1 - y0 > x1 - x0  # the wide table stands upright on the portrait page


def test_two_column_prose_gives_no_table(borderless_doc):
    assert borderless_doc.pages[2].tables == []


def test_bar_chart_labels_are_not_a_table(borderless_doc):
    assert borderless_doc.pages[3].tables == []
