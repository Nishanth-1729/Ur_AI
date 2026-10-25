"""Regressions found on other documents (skipped when the PDF is not present)."""
from pathlib import Path

import pytest

from pdftable import Config, process

ROOT = Path(__file__).resolve().parent.parent
TATA = ROOT / "Tata.pdf"


@pytest.mark.skipif(not TATA.is_file(), reason="Tata.pdf not present")
def test_header_strip_lattice_does_not_replace_the_whole_table():
    """Tata p20: the shaded header bars form a tiny ruled 2x2 grid inside the borderless
    table.  The fragment must not win over (or block) the reading of the whole table."""
    doc = process(TATA, Config(fallback="never", pages=(20,)))
    (t,) = doc.tables
    assert (t.n_rows, t.n_cols, t.header_rows) == (39, 4, 2)
    m = t.matrix()
    assert m[0][:2] == ["Company", "Revenue"] and m[1][1:] == ["FY20", "FY25", "FY26"]
    assert m[3] == ["Tata Consultancy Services", "1,61,541", "2,59,286", "2,71,423"]
    assert m[-1] == ["Tata Group Aggregate", "7,89,057", "15,07,079", "16,24,030"]
