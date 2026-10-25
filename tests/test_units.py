"""Grid reconstruction, HTML parsing and the evaluation metrics."""
import copy

import pytest

from pdftable.evaluate import compare_tables, grits, lcs_similarity, normalize
from pdftable.inspection import WordSet
from pdftable.models import Cell, Table
from pdftable.reconstruct import build_grid
from pdftable.visual import html_to_cells


def _table(rows, spans=()):
    return Table.from_dict({"rows": rows, "spans": spans, "header_rows": 0})


# ---- reconstruction --------------------------------------------------------

def test_build_grid_recovers_spans_and_fills_holes():
    boxes = [(0, 0, 20, 10),  # merged 1x2 header
             (0, 10, 10, 20), (10, 10, 20, 20),
             (0, 20, 10, 30)]  # (2,1) missing -> implicit empty cell
    g = build_grid(boxes)
    assert g.shape == (3, 2)
    spans = sorted((c.row, c.col, c.rowspan, c.colspan) for c in g.cells)
    assert spans == [(0, 0, 1, 2), (1, 0, 1, 1), (1, 1, 1, 1), (2, 0, 1, 1), (2, 1, 1, 1)]


def test_noisy_grid_drops_phantom_column():
    # model boxes: col 2 sometimes hugs right-aligned text and sometimes is split in two
    rows = [(0, 100, 205, 300), (0, 100, 238, 300), (0, 100, 204, 240, 300)]
    boxes = []
    for r, xs in enumerate(rows):
        boxes += [(xs[i], r * 10, xs[i + 1], r * 10 + 10) for i in range(len(xs) - 1)]
    words = WordSet([(5, 2, 50, 8, "label", 0, 0, 0), (250, 2, 295, 8, "1,000", 0, 0, 0),
                     (120, 12, 190, 18, "2,000", 0, 0, 0)])
    assert build_grid(boxes).shape == (3, 4)  # exact mode keeps every boundary
    assert build_grid(boxes, tol=6, noisy=True, words=words).shape == (3, 3)


def test_html_to_cells_handles_rowspan_and_colspan():
    cells, nr, nc = html_to_cells(
        "<table><tr><th rowspan=2>A</th><th colspan=2>B</th></tr>"
        "<tr><td>b1</td><td>b2</td></tr><tr><td>x</td><td>1</td><td>2</td></tr></table>")
    assert (nr, nc) == (3, 3)
    got = {(c.row, c.col): (c.text, c.rowspan, c.colspan) for c in cells}
    assert got[(0, 0)] == ("A", 2, 1) and got[(0, 1)] == ("B", 1, 2)
    assert got[(1, 1)] == ("b1", 1, 1) and got[(2, 2)] == ("2", 1, 1)


# ---- evaluation ------------------------------------------------------------

def test_normalize_is_whitespace_only():
    assert normalize(" (2025 年３月 31 日)\n") == "(2025年３月31日)"
    assert normalize("A  b\nc") == "A b c"
    assert normalize("△2,979") == "△2,979" and normalize("３") == "３"  # no NFKC folding


def test_lcs_similarity():
    assert lcs_similarity("abc", "abc") == 1.0
    assert lcs_similarity("", "x") == 0.0
    assert lcs_similarity("abcd", "abxd") == pytest.approx(0.75)


def test_grits_identity_and_degradation():
    gt = _table([["h", "x", "y"], ["a", "1", "2"], ["b", "3", "4"]], [[0, 1, 1, 2]])
    assert grits(gt, gt) == {"grits_top": 1.0, "grits_con": 1.0, "grits_loc": None}
    missing_row = _table([["h", "x", "y"], ["a", "1", "2"]], [[0, 1, 1, 2]])
    s = grits(gt, missing_row)
    assert 0.7 < s["grits_top"] < 1.0 and 0.6 < s["grits_con"] < 1.0
    no_span = _table([["h", "x", "y"], ["a", "1", "2"], ["b", "3", "4"]])
    assert grits(gt, no_span)["grits_top"] < 1.0


def test_strict_metric_catches_a_single_swapped_value():
    gt = _table([["k", "v"], ["a", "100"], ["b", "200"], ["c", "300"]])
    pred = copy.deepcopy(gt)
    a, b = (next(c for c in pred.cells if (c.row, c.col) == rc) for rc in [(1, 1), (2, 1)])
    a.text, b.text = b.text, a.text
    res = compare_tables(gt, pred)
    assert res["grits_top"] == 1.0 and res["grits_con"] > 0.8  # similarity stays high ...
    assert res["strict_exact"] is False  # ... but the table is not correct
    assert {(f["row"], f["expected"], f["got"]) for f in res["failures"]} == {(1, "100", "200"), (2, "200", "100")}


def test_table_json_roundtrip():
    t = Table(1, 1, (0, 0, 10, 10), 1, 2, [Cell(0, 0, "x", bbox=(0, 0, 5, 10)), Cell(0, 1, "", source="empty")])
    assert Table.from_dict(t.to_dict()).to_dict() == t.to_dict()
