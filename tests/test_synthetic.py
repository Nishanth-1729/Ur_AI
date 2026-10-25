"""Merged cells, several tables per page, pages without tables."""
import pytest

from pdftable import Config, process
from pdftable.export import table_html


@pytest.fixture(scope="module")
def doc(synthetic_pdf):
    return process(synthetic_pdf, Config(fallback="never", workers=1))


def test_two_tables_on_first_page_none_on_second(doc):
    p1, p2 = doc.pages
    assert [t.index for t in p1.tables] == [1, 2]
    assert p2.tables == [] and p2.route == "native" and not p2.warnings


def test_colspan_header_and_blank_cells(doc):
    t = doc.pages[0].tables[0]
    assert (t.n_rows, t.n_cols, t.header_rows) == (4, 3, 1)
    head = [c for c in t.cells if c.row == 0]
    assert [(c.col, c.colspan, c.text) for c in head] == [(0, 1, "Item"), (1, 2, "FY2025")]
    assert t.matrix()[1:] == [["Sales", "100", "200"], ["Cost", "△30", "－"], ["Profit", "70", ""]]
    assert 'colspan="2"' in table_html(t)


def test_second_table_keeps_its_caption_not_the_first_tables_text(doc):
    t = doc.pages[0].tables[1]
    assert t.matrix() == [["Name", "Qty"], ["A", "1"], ["B", "2"]]
    assert t.header_rows == 1
    assert t.unit == "Table 2 (unit: pcs)"
    assert doc.pages[0].tables[0].title == "Sales summary"


def test_parallel_matches_sequential(synthetic_pdf, doc):
    par = process(synthetic_pdf, Config(fallback="never", workers=2))
    assert [t.to_dict() for t in par.tables] == [t.to_dict() for t in doc.tables]
