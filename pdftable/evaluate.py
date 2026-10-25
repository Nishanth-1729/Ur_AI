"""Phase 5 - correctness definition and evaluation against manual ground truth.

Correctness definition (a table is *strictly correct* when all hold):
  1. it is detected - matched to a ground-truth table with bbox IoU >= 0.5
     (or by order when the annotation has no bbox) - and no spurious table exists;
  2. the grid has the same number of rows, columns and header rows;
  3. every cell has the same position, rowspan and colspan;
  4. every cell's text equals the reference after ``normalize`` (blank cells included);
  5. the unit / title, when annotated, are retained.

``normalize`` is the only text normalisation and is meaning-preserving:
  * runs of whitespace (including line breaks inside a cell) -> one space, trimmed;
  * whitespace adjacent to CJK / full-width characters is removed (CJK
    typesetting has no word spaces; such gaps are rendering artefacts).
Nothing else changes: "△2,979", "－", full-width digits, punctuation are compared as is.

Metrics: detection precision / recall / mean IoU; GriTS-Top, -Con, -Loc
(Smock et al., 2023: factored 2-D most-similar-substructure alignment);
exact cell-text accuracy; character error rate (CER); strict exact-table accuracy.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from .export import read_json
from .models import DocumentResult, Table
from .textgrid import is_value

_CJK = r"[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]"
_WS = re.compile(r"\s+")
_CJK_GAP = re.compile(rf"(?<={_CJK})\s+|\s+(?={_CJK})")


def normalize(text: Optional[str]) -> str:
    return _CJK_GAP.sub("", _WS.sub(" ", text or "").strip())


# ---------------------------------------------------------------------------
# GriTS
# ---------------------------------------------------------------------------

def _lcs_len(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if ca == cb else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


@lru_cache(maxsize=1 << 16)
def lcs_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    return 2.0 * _lcs_len(a, b) / (len(a) + len(b))


def _iou4(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """IoU of every true box (Rt,Ct,4) with every predicted box (Rp,Cp,4) -> (Rt,Rp,Ct,Cp)."""
    t = T[:, None, :, None, :]
    p = P[None, :, None, :, :]
    iw = np.clip(np.minimum(t[..., 2], p[..., 2]) - np.maximum(t[..., 0], p[..., 0]), 0, None)
    ih = np.clip(np.minimum(t[..., 3], p[..., 3]) - np.maximum(t[..., 1], p[..., 1]), 0, None)
    inter = iw * ih
    union = (t[..., 2] - t[..., 0]) * (t[..., 3] - t[..., 1]) + (p[..., 2] - p[..., 0]) * (p[..., 3] - p[..., 1]) - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _inner(R: np.ndarray) -> np.ndarray:
    """Best monotone column alignment score for every (true row, pred row) pair at once."""
    Rt, Rp, Ct, Cp = R.shape
    prev = np.zeros((Cp + 1, Rt, Rp))
    for a in range(Ct):
        cur = np.zeros_like(prev)
        for b in range(1, Cp + 1):
            cur[b] = np.maximum(np.maximum(prev[b - 1] + R[:, :, a, b - 1], cur[b - 1]), prev[b])
        prev = cur
    return prev[Cp]


def _outer(S: np.ndarray) -> list[tuple[int, int]]:
    """1-D weighted LCS over rows using pairwise scores ``S``; returns aligned index pairs."""
    n, m = S.shape
    D = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        Di, Dp, Si = D[i], D[i - 1], S[i - 1]
        for j in range(1, m + 1):
            Di[j] = max(Dp[j - 1] + Si[j - 1], Di[j - 1], Dp[j])
    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        if np.isclose(D[i, j], D[i - 1, j - 1] + S[i - 1, j - 1]) and S[i - 1, j - 1] > 0:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif np.isclose(D[i, j], D[i - 1, j]):
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def grits_from_rewards(R: np.ndarray) -> tuple[float, float, float]:
    """(fscore, precision, recall) given the 4-D reward tensor R[t_row, p_row, t_col, p_col]."""
    Rt, Rp, Ct, Cp = R.shape
    rows = _outer(_inner(R))
    cols = _outer(_inner(R.transpose(2, 3, 0, 1)))
    if not rows or not cols:
        return 0.0, 0.0, 0.0
    ti, pi = map(np.array, zip(*rows))
    tj, pj = map(np.array, zip(*cols))
    score = float(R[ti[:, None], pi[:, None], tj[None, :], pj[None, :]].sum())
    nt, npred = Rt * Ct, Rp * Cp
    return 2 * score / (nt + npred), score / npred, score / nt


def _position_arrays(t: Table):
    occ = np.asarray(t.occupancy())
    rr, cc = np.indices(occ.shape)
    cells = t.cells
    span = np.asarray([[c.col, c.row, c.col + c.colspan, c.row + c.rowspan] for c in cells], float)[occ]
    rel = span - np.stack([cc, rr, cc, rr], axis=-1)
    texts = np.asarray([normalize(c.text) for c in cells], dtype=object)[occ]
    loc = None
    if all(c.bbox is not None for c in cells):
        loc = np.asarray([c.bbox for c in cells], float)[occ]
    return rel, texts, loc


def grits(gt: Table, pred: Table) -> dict[str, Optional[float]]:
    g_rel, g_txt, g_loc = _position_arrays(gt)
    p_rel, p_txt, p_loc = _position_arrays(pred)
    top = grits_from_rewards(_iou4(g_rel, p_rel))[0]
    # content reward: LCS similarity, computed once per distinct (true, pred) string pair
    gu, gi = np.unique(g_txt.astype(str), return_inverse=True)
    pu, pi = np.unique(p_txt.astype(str), return_inverse=True)
    sim = np.array([[lcs_similarity(a, b) for b in pu] for a in gu])
    gi, pi = gi.reshape(g_txt.shape), pi.reshape(p_txt.shape)
    con = grits_from_rewards(sim[gi[:, None, :, None], pi[None, :, None, :]])[0]
    loc = grits_from_rewards(_iou4(g_loc, p_loc))[0] if g_loc is not None and p_loc is not None else None
    return {"grits_top": top, "grits_con": con, "grits_loc": loc}


# ---------------------------------------------------------------------------
# Other metrics
# ---------------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _reading_text(t: Table) -> str:
    return "\n".join("\t".join(row) for row in
                     [[normalize(x) for x in r] for r in t.matrix()])


def bbox_iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _cell_key(c) -> tuple:
    return c.row, c.col, c.rowspan, c.colspan


# ---------------------------------------------------------------------------
# Relationship metric: (row label, column header, value) triples
# ---------------------------------------------------------------------------

def relations(t: Table) -> Counter:
    """Every non-empty body cell outside the first column as a triple
    (row label, column-header path, value):

    * row label - the row's non-numeric cells to the left of the value (the stub, plus any
      descriptive columns such as "Nature of transaction"), joined;
    * column-header path - the texts of all header cells covering the column, top to bottom
      ("FY 2023-24 / Male");
    * value - the cell text.

    All three are ``normalize``-d.  A table whose values sit under the wrong header or next to
    the wrong label loses exactly those triples, regardless of how the error arose (missing
    row, extra column, shifted cell, prose mixed in), so the score reads directly as "what
    fraction of the facts in the table were extracted with the right row and column"."""
    occ = t.occupancy()
    nh = min(t.header_rows, t.n_rows)
    col_key = []
    for c in range(t.n_cols):
        parts: list[str] = []
        for r in range(nh):
            i = occ[r][c]
            txt = normalize(t.cells[i].text) if i >= 0 else ""
            if txt and (not parts or parts[-1] != txt):
                parts.append(txt)
        col_key.append(" / ".join(parts))
    out: Counter = Counter()
    for r in range(nh, t.n_rows):
        row, spanned = [], []  # anchor text (values) / text of the covering cell (labels)
        for c in range(t.n_cols):
            i = occ[r][c]
            txt = normalize(t.cells[i].text) if i >= 0 else ""
            anchor = i >= 0 and t.cells[i].row == r and t.cells[i].col == c
            row.append(txt if anchor else "")
            spanned.append(txt if i >= 0 and t.cells[i].col == c else "")
        for c in range(1, t.n_cols):
            if row[c]:
                label = " | ".join(x for x in spanned[:c] if x and not is_value(x))
                out[(label, col_key[c], row[c])] += 1
    return out


def _prf(correct: int, n_pred: int, n_gt: int) -> dict:
    p = correct / n_pred if n_pred else (1.0 if not n_gt else 0.0)
    r = correct / n_gt if n_gt else (1.0 if not n_pred else 0.0)
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "correct": correct, "n_pred": n_pred, "n_gt": n_gt}


def compare_tables(gt: Table, pred: Table, max_failures: int = 25) -> dict:
    g = {_cell_key(c): normalize(c.text) for c in gt.cells}
    p = {_cell_key(c): normalize(c.text) for c in pred.cells}
    correct = sum(1 for k, v in g.items() if p.get(k) == v)
    occ = pred.occupancy()
    failures = []
    for c in gt.cells:
        if p.get(_cell_key(c)) != normalize(c.text) and len(failures) < max_failures:
            i = occ[c.row][c.col] if c.row < pred.n_rows and c.col < pred.n_cols else -1
            got = pred.cells[i] if i >= 0 else None
            failures.append({"row": c.row, "col": c.col, "expected": c.text,
                             "got": got.text if got else None, "span": [c.rowspan, c.colspan],
                             "got_span": [got.rowspan, got.colspan] if got else None})
    shape_ok = (gt.n_rows, gt.n_cols, gt.header_rows) == (pred.n_rows, pred.n_cols, pred.header_rows)
    meta = {k: normalize(getattr(gt, k)) == normalize(getattr(pred, k)) for k in ("unit", "title") if getattr(gt, k)}
    meta_ok = all(meta.values())
    ref, hyp = _reading_text(gt), _reading_text(pred)
    rg, rp = relations(gt), relations(pred)
    res = {
        "shape_gt": [gt.n_rows, gt.n_cols, gt.header_rows],
        "shape_pred": [pred.n_rows, pred.n_cols, pred.header_rows],
        **grits(gt, pred),
        "cell_exact_acc": correct / len(g) if g else 1.0,
        "cer": levenshtein(ref, hyp) / max(1, len(ref)),
        "relations": _prf(sum((rg & rp).values()), sum(rp.values()), sum(rg.values())),
        "strict_exact": bool(shape_ok and meta_ok and g == p),
        "unit_title_ok": meta_ok,
        "unit_ok": meta.get("unit"),
        "title_ok": meta.get("title"),
        "failures": failures,
    }
    if not shape_ok:
        failures.insert(0, {"structure": f"expected {res['shape_gt']} (rows, cols, header_rows), got {res['shape_pred']}"})
    if not meta_ok:
        failures.insert(0, {"unit_title": {"expected": [gt.unit, gt.title], "got": [pred.unit, pred.title]}})
    return res


def _match(gts: list[Table], preds: list[Table], thr: float = 0.5):
    if gts and all(t.bbox is None for t in gts):  # annotation without geometry: match by order
        n = min(len(gts), len(preds))
        return [(i, i, None) for i in range(n)]
    cand = sorted(((bbox_iou(g.bbox, p.bbox), i, j) for i, g in enumerate(gts) for j, p in enumerate(preds)),
                  reverse=True)
    used_g, used_p, out = set(), set(), []
    for iou, i, j in cand:
        if iou >= thr and i not in used_g and j not in used_p:
            used_g.add(i), used_p.add(j)
            out.append((i, j, iou))
    return out


@dataclass
class Totals:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    ious: list[float] = field(default_factory=list)
    per_table: list[dict] = field(default_factory=list)
    rel_correct: int = 0  # relationship triples, end to end (missed / spurious tables included)
    rel_pred: int = 0
    rel_gt: int = 0
    pages: int = 0


def evaluate_document(gt: DocumentResult, pred: DocumentResult, totals: Totals,
                      only_pages: Optional[set[int]] = None) -> dict:
    """Ground truth may annotate only some pages (``meta["annotated_pages"]``); predictions on
    other pages are then not scored.  ``only_pages`` restricts scoring further (a split)."""
    pred_pages = {p.page: p.tables for p in pred.pages}
    gt_pages = {p.page: p.tables for p in gt.pages}
    annotated = gt.meta.get("annotated_pages")
    pages = sorted(annotated) if annotated else sorted(set(gt_pages) | set(pred_pages))
    if only_pages is not None:
        pages = [p for p in pages if p in only_pages]
    doc = {"source": pred.source or gt.source, "pages": pages, "tables": [], "missed": [], "spurious": []}
    totals.pages += len(pages)
    for pg in pages:
        gts, preds = gt_pages.get(pg, []), pred_pages.get(pg, [])
        matches = _match(gts, preds)
        mg, mp = {m[0] for m in matches}, {m[1] for m in matches}
        totals.tp += len(matches)
        totals.fn += len(gts) - len(mg)
        totals.fp += len(preds) - len(mp)
        doc["missed"] += [{"page": pg, "index": gts[i].index} for i in range(len(gts)) if i not in mg]
        doc["spurious"] += [{"page": pg, "index": preds[j].index} for j in range(len(preds)) if j not in mp]
        # end-to-end facts are pooled per page (no table matching needed): a fact counts when the
        # page's output contains it, whatever the table boundaries, and a missed or spurious
        # table simply contributes its facts to recall or precision
        rg, rp = Counter(), Counter()
        for t in gts:
            rg.update(relations(t))
        for t in preds:
            rp.update(relations(t))
        totals.rel_correct += sum((rg & rp).values())
        totals.rel_gt += sum(rg.values())
        totals.rel_pred += sum(rp.values())
        for i, j, iou in matches:
            r = {"page": pg, "gt_index": gts[i].index, "pred_index": preds[j].index,
                 "bbox_iou": iou, **compare_tables(gts[i], preds[j])}
            if iou is not None:
                totals.ious.append(iou)
            totals.per_table.append(r)
            doc["tables"].append(r)
    return doc


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def summarize(t: Totals) -> dict:
    n_gt = t.tp + t.fn
    pt = t.per_table
    return {
        "detection": {"tp": t.tp, "fp": t.fp, "fn": t.fn,
                      "precision": t.tp / (t.tp + t.fp) if t.tp + t.fp else 1.0,
                      "recall": t.tp / n_gt if n_gt else 1.0,
                      "mean_iou": _mean(t.ious)},
        "matched_tables": len(pt),
        "grits_top": _mean([r["grits_top"] for r in pt]),
        "grits_con": _mean([r["grits_con"] for r in pt]),
        "grits_loc": _mean([r["grits_loc"] for r in pt]),
        "cell_exact_acc": _mean([r["cell_exact_acc"] for r in pt]),
        "cer": _mean([r["cer"] for r in pt]),
        # end-to-end: triples of missed tables count as not found, of spurious tables as wrong
        "relations": _prf(t.rel_correct, t.rel_pred, t.rel_gt),
        "unit_acc": _mean([float(r["unit_ok"]) for r in pt if r["unit_ok"] is not None]),
        "title_acc": _mean([float(r["title_ok"]) for r in pt if r["title_ok"] is not None]),
        # end-to-end: missed tables count as incorrect, spurious tables are reported in detection
        "strict_exact_acc": sum(r["strict_exact"] for r in pt) / n_gt if n_gt else None,
        "pages": t.pages,
    }


def evaluate_pairs(pairs: list[tuple[Path, Path]]) -> dict:
    loaded = [(read_json(g), read_json(p)) for g, p in pairs]
    totals = Totals()
    docs = [evaluate_document(g, p, totals) for g, p in loaded]
    report = {"summary": summarize(totals), "documents": docs}
    names = sorted({k for g, _ in loaded for k in g.meta.get("splits", {})})
    if names:
        report["splits"] = {}
        for name in names:
            t = Totals()
            for g, p in loaded:
                if name in g.meta.get("splits", {}):
                    evaluate_document(g, p, t, set(g.meta["splits"][name]))
            report["splits"][name] = summarize(t)
    return report


def pair_files(gt: Path, pred: Path) -> list[tuple[Path, Path]]:
    """Pair ground-truth and prediction JSON files by stem (files or directories)."""
    if gt.is_file() and pred.is_file():
        return [(gt, pred)]
    gts = {f.stem: f for f in ([gt] if gt.is_file() else gt.rglob("*.json"))}
    preds = {f.stem: f for f in ([pred] if pred.is_file() else pred.rglob("*.json"))}
    return [(gts[s], preds[s]) for s in sorted(gts) if s in preds]


def _summary_lines(s: dict) -> list[str]:
    d = s["detection"]
    f = lambda v: "n/a" if v is None else f"{v:.4f}"  # noqa: E731
    rel = s["relations"]
    return [
        f"Scope       {s['pages']} annotated page(s), {d['tp'] + d['fn']} reference table(s)",
        f"Detection   P={f(d['precision'])} R={f(d['recall'])} meanIoU={f(d['mean_iou'])} "
        f"(tp={d['tp']} fp={d['fp']} fn={d['fn']})",
        f"Relations   P={f(rel['precision'])} R={f(rel['recall'])} F1={f(rel['f1'])} "
        f"({rel['correct']} of {rel['n_gt']} reference (row label, column header, value) triples; "
        f"{rel['n_pred']} predicted)",
        f"GriTS       Top={f(s['grits_top'])} Con={f(s['grits_con'])} Loc={f(s['grits_loc'])}",
        f"Cells       exact-text acc={f(s['cell_exact_acc'])}  CER={f(s['cer'])}",
        f"Captions    unit acc={f(s['unit_acc'])}  title acc={f(s['title_acc'])}",
        f"Strict      exact-table acc={f(s['strict_exact_acc'])} over {d['tp'] + d['fn']} reference tables",
    ]


def format_summary(report: dict) -> str:
    lines = _summary_lines(report["summary"])
    for name, s in report.get("splits", {}).items():
        lines += ["", f"[split: {name}]"] + ["  " + x for x in _summary_lines(s)]
    lines.append("")
    for doc in report["documents"]:
        for t in doc["tables"]:
            if not t["strict_exact"]:
                lines.append(f"  FAIL {Path(doc['source']).name} p{t['page']} t{t['pred_index']}: "
                             + json.dumps(t["failures"][:5], ensure_ascii=False))
        for m in doc["missed"]:
            lines.append(f"  MISSED {Path(doc['source']).name} p{m['page']} t{m['index']}")
        for m in doc["spurious"]:
            lines.append(f"  SPURIOUS {Path(doc['source']).name} p{m['page']} t{m['index']}")
    return "\n".join(lines)
