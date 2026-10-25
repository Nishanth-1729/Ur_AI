"""End-to-end checks on the supplied balance sheet (native fast path)."""
import csv
import json

import pytest

from pdftable import Config, process
from pdftable.evaluate import evaluate_pairs
from pdftable.export import export, read_json
from pdftable.models import DocumentResult

from .conftest import SAMPLE, SAMPLE_GT


@pytest.fixture(scope="module")
def doc() -> DocumentResult:
    return process(SAMPLE, Config(fallback="never"))


def test_routing_and_shape(doc):
    (page,) = doc.pages
    assert page.page_type == "digital" and page.route == "native" and not page.warnings
    (t,) = page.tables
    assert (t.n_rows, t.n_cols, t.header_rows) == (44, 3, 1)  # 1 external header row + 43 body rows
    assert t.reliability.reliable and not t.needs_review


def test_external_header_unit_and_title_attached(doc):
    t = doc.tables[0]
    m = t.matrix()
    assert m[0][0] == ""
    assert m[0][1].startswith("前連結会計年度\n(2025")
    assert m[0][2].startswith("当連結会計年度\n(2026")
    assert all(c.is_header for c in t.cells if c.row == 0)
    assert t.unit == "（単位：千円）"
    assert t.title == "（１）連結貸借対照表"
    assert t.bbox[1] == pytest.approx(115.44, abs=0.1)  # header fenced by the top rule


def test_cell_text_preserved_exactly(doc):
    m = doc.tables[0].matrix()
    rows = {r[0]: r[1:] for r in m[1:]}
    assert rows["現金及び預金"] == ["2,002,429", "608,281"]  # finder's own text was "2 002 429\n, ,"
    assert rows["貸倒引当金"] == ["△2,979", "△0"]
    assert rows["建設仮勘定"] == ["－", "4,553"]
    assert m[1] == ["資産の部", "", ""]  # blank cells are part of the table
    assert m[-1] == ["資産合計", "7,534,110", "5,135,249"]


def test_strictly_correct_against_manual_ground_truth(doc, tmp_path):
    export(doc, tmp_path, {"json"})
    report = evaluate_pairs([(SAMPLE_GT, tmp_path / "balance_sheet_after_correction.json")])
    s = report["summary"]
    assert s["detection"] == {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0,
                              "mean_iou": pytest.approx(0.9986, abs=1e-3)}
    assert s["grits_top"] == s["grits_con"] == 1.0
    assert s["cell_exact_acc"] == 1.0 and s["cer"] == 0.0
    assert s["strict_exact_acc"] == 1.0


def test_exports_roundtrip(doc, tmp_path):
    files = {p.suffix for p in export(doc, tmp_path, {"json", "html", "csv"})}
    assert files == {".json", ".html", ".csv"}
    back = read_json(tmp_path / "balance_sheet_after_correction.json")
    assert back.tables[0].matrix() == doc.tables[0].matrix()
    html = (tmp_path / "balance_sheet_after_correction.html").read_text(encoding="utf-8")
    assert "<thead>" in html and "△2,979" in html and "（単位：千円）" in html
    with (tmp_path / "balance_sheet_after_correction_p001_t01.csv").open(encoding="utf-8-sig", newline="") as f:
        assert list(csv.reader(f)) == doc.tables[0].matrix()
    raw = json.loads((tmp_path / "balance_sheet_after_correction.json").read_text(encoding="utf-8"))
    cell = raw["pages"][0]["tables"][0]["cells"][9]
    assert set(cell) >= {"row", "col", "text", "rowspan", "colspan", "bbox", "source"}
