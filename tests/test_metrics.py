"""Relationship metric and arithmetic audit."""
import copy

from pdftable.audit import audit_table, parse_number
from pdftable.evaluate import compare_tables, relations
from pdftable.models import Table


def _table(rows, header_rows=1, spans=()):
    return Table.from_dict({"rows": rows, "spans": spans, "header_rows": header_rows})


GT = _table([["INR crore", "FY23", "FY24"],
             ["Food delivery", "6,147", "7,792"],
             ["Quick commerce", "806", "2,301"],
             ["Total", "6,953", "10,093"]])


def test_relations_are_row_label_column_header_value():
    assert relations(GT)[("Food delivery", "FY24", "7,792")] == 1
    assert sum(relations(GT).values()) == 6


def test_relations_catch_a_value_under_the_wrong_header():
    pred = copy.deepcopy(GT)
    for c in pred.cells:  # header cells swapped: every value now sits under the wrong year
        if c.row == 0 and c.col in (1, 2):
            c.text = {"FY23": "FY24", "FY24": "FY23"}[c.text]
    res = compare_tables(GT, pred)
    assert res["grits_con"] > 0.8  # the same strings are all there ...
    assert res["relations"]["correct"] == 0  # ... but no value is attached to its year


def test_relations_use_merged_row_labels_and_header_paths():
    t = _table([["", "FY 2023-24", ""], ["", "Male", "Female"], ["Permanent", "34.79%", "44.44%"],
                ["", "0%", "0%"]], header_rows=2, spans=[[0, 1, 1, 2], [2, 0, 2, 1]])
    rel = relations(t)
    assert rel[("Permanent", "FY 2023-24 / Male", "34.79%")] == 1
    assert rel[("Permanent", "FY 2023-24 / Female", "0%")] == 1  # label covers the merged rows


def test_parse_number():
    assert parse_number("1,234") == 1234 and parse_number("(1,234)") == -1234
    assert parse_number("–766") == -766 and parse_number("△2,979") == -2979
    assert parse_number("-") == 0 and parse_number("78,72,500¹") == 7872500
    assert parse_number("12.5%") is None and parse_number("31-Mar-24") is None


def test_audit_verifies_totals_and_flags_a_shifted_value():
    ok = audit_table(GT)
    assert [c.status for c in ok.checks] == ["verified", "verified"]
    bad = copy.deepcopy(GT)
    next(c for c in bad.cells if (c.row, c.col) == (2, 2)).text = "2,031"  # a misread / shifted value
    assert [c.status for c in audit_table(bad).checks] == ["verified", "mismatch"]
