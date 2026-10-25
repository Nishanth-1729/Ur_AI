"""Arithmetic self-consistency audit - a correctness signal that needs no ground truth.

Financial tables check themselves: a "Total ..." row equals the sum of the rows
above it.  If a value lands in the wrong column, a row is split or merged, or
prose leaks into the grid, totals stop adding up.  For every row whose label
starts with "Total" / "Sub-total" / "Net total", and for every numeric column,
the audit tries three explanations, within the printed rounding: the line items
directly above it (back to the previous total, sub-headings passed over, rows
labelled "Less: ..." subtracted), the sum of the subtotals above it (a grand
total), and the values exactly as printed (a "Less:" row may already show the
net amount).  At least two values must add up.

A total that no explanation reproduces is a *mismatch*: either an extraction error or a genuine
inconsistency in the document (annual reports print "totals may not add up due
to rounding"), so mismatches are pointers for review, not proof of error.

This is not a substitute for ground truth: it only covers numeric tables with
totals, and a table can be wrong in ways that keep its totals intact.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .models import DocumentResult, Table

_TOTAL = re.compile(r"^\s*(sub[\s-]?|net\s+)?total\b", re.I)
_LESS = re.compile(r"^\s*less\b", re.I)
_NUM = re.compile(r"^\(?[-–−△▲]?\(?([\d,]+(?:\.\d+)?)\)?$")
_NIL = re.compile(r"^[-–—−－]{1,2}$")
_SUP = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁽⁾*#]+$")


def parse_number(text: str) -> Optional[float]:
    """Printed value -> number: "1,234" 1234, "(1,234)" / "–766" / "△2,979" negative, "-" 0.
    Percentages, dates and text give None."""
    s = _SUP.sub("", (text or "").strip()).replace(" ", "")
    if not s:
        return None
    if _NIL.match(s):
        return 0.0
    if "%" in s or "％" in s:
        return None
    m = _NUM.match(s)
    if not m:
        return None
    v = float(m.group(1).replace(",", ""))
    negative = s[0] == "(" or s[0] in "-–−△▲" or (len(s) > 1 and s[1] in "-–−△▲")
    return -v if negative else v


@dataclass
class TotalCheck:
    row: int
    col: int
    label: str
    total: float
    status: str  # verified | mismatch | not_checkable (fewer than two values above it in its section)
    block: Optional[tuple[int, int]] = None  # rows summed (inclusive), when verified


@dataclass
class TableAudit:
    page: int
    index: int
    checks: list[TotalCheck] = field(default_factory=list)

    def count(self, status: str) -> int:
        return sum(c.status == status for c in self.checks)


def audit_table(t: Table, max_block: int = 40) -> TableAudit:
    out = TableAudit(t.page, t.index)
    m = t.matrix()
    labels = [row[0].replace("\n", " ") if row else "" for row in m]
    nums = [[parse_number(x) for x in row] for row in m]
    for r in range(t.header_rows, t.n_rows):
        if not _TOTAL.match(labels[r]):
            continue
        for c in range(1, t.n_cols):
            total = nums[r][c]
            if total is None:
                continue
            # (a) line items: values in this column above the total, back to the previous total
            # (sub-headings such as "Statutory dues :" are passed over)
            items: list[tuple[int, float]] = []
            for k in range(r - 1, max(t.header_rows, r - max_block) - 1, -1):
                if _TOTAL.match(labels[k]) and len(items) >= 2:
                    break
                v = nums[k][c]
                if v is not None:
                    items.append((k, -abs(v) if _LESS.match(labels[k]) and v > 0 else v))
            # (b) subtotals: a grand total ("Total assets") adds up the totals above it
            subtotals = [(k, nums[k][c]) for k in range(r - 1, max(t.header_rows, r - max_block) - 1, -1)
                         if _TOTAL.match(labels[k]) and nums[k][c] is not None]
            if len(items) < 2 and len(subtotals) < 2:
                out.checks.append(TotalCheck(r, c, labels[r], total, "not_checkable"))
                continue
            # (c) as printed: a "Less:" row may already show the net amount in this column
            raw = [(k, nums[k][c]) for k, _ in items]
            found = None
            for cand in (items, subtotals, raw):
                acc = 0.0
                for n, (k, v) in enumerate(cand, 1):
                    acc += v
                    tol = max(1.0, 0.5 * n, 0.005 * abs(total))  # printed values are rounded
                    if n >= 2 and abs(acc - total) <= tol:
                        found = (k, r - 1)
                        break
                if found:
                    break
            out.checks.append(TotalCheck(r, c, labels[r], total, "verified" if found else "mismatch", found))
    return out


def audit_document(doc: DocumentResult) -> dict:
    tables = [audit_table(t) for t in doc.tables]
    checks = [c for a in tables for c in a.checks]
    checkable = [c for c in checks if c.status != "not_checkable"]
    audited = [a for a in tables if a.count("verified") + a.count("mismatch")]
    ok = sum(c.status == "verified" for c in checkable)
    return {
        "source": doc.source,
        "tables": len(doc.tables),
        "tables_with_checkable_totals": len(audited),
        "total_cells": len(checks),
        "total_cells_checkable": len(checkable),
        "total_cells_verified": ok,
        "verified_ratio": ok / len(checkable) if checkable else None,
        "tables_fully_verified": sum(1 for a in audited if not a.count("mismatch")),
        "mismatches": [{"page": a.page, "table": a.index, "row": c.row, "col": c.col,
                        "label": c.label, "total": c.total}
                       for a in tables for c in a.checks if c.status == "mismatch"],
    }


def format_audit(rep: dict, max_lines: int = 30) -> str:
    r = rep["verified_ratio"]
    lines = [
        f"{rep['tables']} table(s); {rep['tables_with_checkable_totals']} with total rows that can be checked "
        f"(at least two values above them)",
        f"total cells that add up: {rep['total_cells_verified']} / {rep['total_cells_checkable']}"
        + (f" ({r:.1%})" if r is not None else "")
        + f"; {rep['total_cells'] - rep['total_cells_checkable']} not checkable",
        f"tables whose every checkable total adds up: {rep['tables_fully_verified']} / {rep['tables_with_checkable_totals']}",
    ]
    for mm in rep["mismatches"][:max_lines]:
        lines.append(f"  MISMATCH p{mm['page']} t{mm['table']} row {mm['row']} col {mm['col']}: "
                     f"{mm['label'][:60]!r} = {mm['total']:g}")
    if len(rep["mismatches"]) > max_lines:
        lines.append(f"  ... {len(rep['mismatches']) - max_lines} more")
    return "\n".join(lines)
