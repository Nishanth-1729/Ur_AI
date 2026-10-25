"""Hybrid PDF-to-table parser: PyMuPDF fast path, reliability routing,
PP-StructureV3 visual fallback, canonical JSON output and GriTS evaluation."""
import os
import sys

# When PyMuPDF's optional ONNX layout analyser (pymupdf.layout) is installed it is imported
# eagerly: ~0.7 s and ~12 CPU-seconds per process, paid again by every worker.  It is not used
# here, so its import is blocked and PyMuPDF behaves as if it were not installed (this must
# happen before PyMuPDF is first imported).
if "pymupdf" not in sys.modules:
    sys.modules.setdefault("pymupdf.layout", None)  # type: ignore[arg-type]
    os.environ.setdefault("PYMUPDF_SUGGEST_LAYOUT_ANALYZER", "0")

from .models import Cell, DocumentResult, PageResult, Table  # noqa: E402
from .pipeline import Config, process, run  # noqa: E402

__all__ = ["Cell", "Config", "DocumentResult", "PageResult", "Table", "process", "run"]
__version__ = "1.1.0"
