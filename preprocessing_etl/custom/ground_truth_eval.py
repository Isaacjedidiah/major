"""Ground-truth evaluation — compare the pipeline's extraction against HUMAN
LABELS to produce credible accuracy metrics (precision / recall / accuracy),
overall and by element type (table vs chart).

This is the strongest evidence for a team: 'here is what a person confirmed is
correct, here is what the pipeline produced, here is the match rate.'

WORKFLOW
  1. A human labels a sample of documents -> ground-truth rows:
       {document, field_name, value, page, element_type}
     (element_type optional; 'table' / 'figure' lets us report by type.)
  2. Run the pipeline on those same documents -> extracted claims.
  3. compare_to_ground_truth(claims, truth) matches them and returns the metrics.

MATCHING
  A claim matches a truth row when their VALUES match (normalised) on the SAME
  page. Value is the unambiguous anchor; field names differ in format between the
  human and the canonical_metric, so value+page is the primary key and field is a
  soft secondary signal. Chart values may be matched within a tolerance (they are
  estimates); table values require an exact normalised match.
"""
from __future__ import annotations
import re
from collections import defaultdict


def _norm_value(v) -> str:
    """Normalise a value for comparison: drop commas, currency, %, spaces; lower."""
    s = str(v if v is not None else "").lower()
    s = re.sub(r"[,£$€%\s]", "", s)
    return s


def _as_number(v):
    try:
        return float(re.sub(r"[,£$€%\s]", "", str(v)))
    except (ValueError, TypeError):
        return None


def _values_match(extracted, truth, element_type=None, chart_tol=0.02):
    """Exact (normalised) match; for chart/figure values allow a small relative
    tolerance since those are vision estimates."""
    if _norm_value(extracted) == _norm_value(truth):
        return True
    if element_type in ("figure", "chart"):
        a, b = _as_number(extracted), _as_number(truth)
        if a is not None and b is not None and b != 0:
            return abs(a - b) / abs(b) <= chart_tol
    return False


def compare_to_ground_truth(claims, truth_rows, chart_tol=0.02):
    """claims: pipeline Claim objects (need .value, .page, optional .element_type,
              .field_name/.canonical_metric).
    truth_rows: list of dicts {document?, field_name?, value, page, element_type?}.
    Returns precision / recall / accuracy overall and by element_type, plus the
    specific misses (false negatives) and spurious (false positives) for review."""

    # index truth by page -> list of rows (unmatched to start)
    truth_by_page = defaultdict(list)
    for t in truth_rows:
        truth_by_page[t.get("page")].append({**t, "_matched": False})

    tp = []     # (claim, truth)
    fp = []     # claims with no truth match
    for c in claims:
        page = getattr(c, "page", None)
        etype = getattr(c, "element_type", None)
        cval = getattr(c, "value", None)
        hit = None
        for t in truth_by_page.get(page, []):
            if t["_matched"]:
                continue
            if _values_match(cval, t.get("value"), etype or t.get("element_type"), chart_tol):
                hit = t
                break
        if hit:
            hit["_matched"] = True
            tp.append((c, hit))
        else:
            fp.append(c)

    fn = [t for rows in truth_by_page.values() for t in rows if not t["_matched"]]

    def _rate(n, d):
        return round(n / d, 4) if d else None

    n_tp, n_fp, n_fn = len(tp), len(fp), len(fn)
    precision = _rate(n_tp, n_tp + n_fp)
    recall = _rate(n_tp, n_tp + n_fn)
    f1 = _rate(2 * (precision or 0) * (recall or 0),
               (precision or 0) + (recall or 0)) if (precision and recall) else None

    # by element type
    by_type = {}
    types = set()
    for c in claims:
        types.add(getattr(c, "element_type", None) or "unknown")
    for t in truth_rows:
        types.add(t.get("element_type") or "unknown")
    for ty in types:
        ttp = sum(1 for (c, t) in tp
                  if (getattr(c, "element_type", None) or "unknown") == ty)
        tfp = sum(1 for c in fp
                  if (getattr(c, "element_type", None) or "unknown") == ty)
        tfn = sum(1 for t in fn if (t.get("element_type") or "unknown") == ty)
        by_type[ty] = {
            "true_positive": ttp, "false_positive": tfp, "false_negative": tfn,
            "precision": _rate(ttp, ttp + tfp), "recall": _rate(ttp, ttp + tfn)}

    return {
        "overall": {
            "ground_truth_total": len(truth_rows),
            "extracted_total": len(claims),
            "true_positive": n_tp, "false_positive": n_fp, "false_negative": n_fn,
            "precision": precision, "recall": recall, "f1": f1,
            "value_accuracy": precision,   # of extracted, fraction correct
        },
        "by_element_type": by_type,
        "misses": [{"page": t.get("page"), "field": t.get("field_name"),
                    "value": t.get("value")} for t in fn],           # we SHOULD have got these
        "spurious": [{"page": getattr(c, "page", None),
                      "field": getattr(c, "field_name", None),
                      "value": getattr(c, "value", None)} for c in fp],  # we extracted, not in truth
    }


def ground_truth_report(result: dict) -> str:
    """Readable report of the ground-truth comparison."""
    o = result["overall"]
    lines = ["=== GROUND-TRUTH EVALUATION ===", ""]
    lines += [
        f"ground-truth values:  {o['ground_truth_total']}",
        f"extracted values:     {o['extracted_total']}",
        f"correct (matched):    {o['true_positive']}",
        f"missed (not extracted): {o['false_negative']}   <- recall gap (coverage)",
        f"spurious (wrong/extra): {o['false_positive']}   <- precision gap",
        "",
        f"PRECISION: {(o['precision'] or 0)*100:.1f}%   (of what we extracted, how much is correct)",
        f"RECALL:    {(o['recall'] or 0)*100:.1f}%   (of what should be extracted, how much we got)",
        f"F1:        {(o['f1'] or 0)*100:.1f}%" if o['f1'] else "F1: n/a",
        "",
        "BY ELEMENT TYPE:"]
    for ty, m in result["by_element_type"].items():
        lines.append(f"  {ty:10s} precision={_pct(m['precision'])} recall={_pct(m['recall'])} "
                     f"(TP={m['true_positive']} FP={m['false_positive']} FN={m['false_negative']})")
    if result["misses"]:
        lines += ["", f"MISSED ({len(result['misses'])}) — should have extracted:"]
        for m in result["misses"][:10]:
            lines.append(f"    p{m['page']} {m['field']}={m['value']}")
    return "\n".join(lines)


def _pct(x):
    return f"{x*100:.0f}%" if x is not None else "n/a"
