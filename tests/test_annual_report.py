"""Regression on the 371-page annual report (file.pdf) against its hand-verified ground truth.

Skipped when file.pdf is not present.  The thresholds lock in the measured accuracy so a change
that breaks a table type shows up here; see ground_truth/file.json for the splits."""
from pathlib import Path

import pytest

from pdftable import Config, process
from pdftable.evaluate import evaluate_pairs
from pdftable.export import export

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / "file.pdf"
GT = ROOT / "ground_truth" / "file.json"

pytestmark = pytest.mark.skipif(not (PDF.is_file() and GT.is_file()), reason="file.pdf / its ground truth not present")


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    import json
    pages = tuple(json.loads(GT.read_text(encoding="utf-8"))["annotated_pages"])
    doc = process(PDF, Config(fallback="never", pages=pages))
    out = tmp_path_factory.mktemp("annual")
    export(doc, out, {"json"})
    return evaluate_pairs([(GT, out / "file.json")])


def test_no_spurious_tables(report):
    assert report["summary"]["detection"]["fp"] == 0


def test_dev_pages_exact(report):
    s = report["splits"]["dev"]
    assert s["strict_exact_acc"] == 1.0 and s["relations"]["f1"] == 1.0


def test_heldout_accuracy(report):
    for name, min_f1 in (("heldout_a", 0.99), ("heldout_b", 0.90)):
        assert report["splits"][name]["relations"]["f1"] >= min_f1, name


def test_rotated_page_263_is_a_real_table(report):
    (t,) = [t for d in report["documents"] for t in d["tables"] if t["page"] == 263]
    assert t["strict_exact"] and t["shape_pred"] == [17, 9, 2]


def test_shaded_header_cells_and_label_only_lines():
    """p172: tall shaded header cells must not be taken for chart bars; p226: labels printed on
    lines of their own (values centred beside them) still form the label column."""
    doc = process(PDF, Config(fallback="never", pages=(172, 226)))
    (p172,), (a, b) = doc.pages[0].tables, doc.pages[1].tables
    assert (p172.n_cols, p172.header_rows) == (12, 3)
    assert p172.matrix()[0][1] == "Equity attributable to owners of the Parent"
    for t in (a, b):
        assert (t.n_cols, t.header_rows) == (12, 2)
        assert t.matrix()[2][0] == "Defined benefit\nobligation"
