"""Comprehensive extraction evaluation — ACCURACY, COVERAGE, ROBUSTNESS.

Built to an agreed spec:

ACCURACY (vs human-labelled ground truth, confusion matrix)
  * ground truth CSV rows: document, page, element_type, field_name, value,
    should_extract, [notes]
  * a claim matches a ground-truth row when ALL of: same DOCUMENT, same PAGE,
    FIELD matches (fuzzy: the human label's words appear in the canonical_metric),
    and VALUE matches — EXACT for table/text, within +-tol for figure (charts).
  * positives (should_extract=TRUE): matched -> TP, unmatched -> FN.
  * decoys (should_extract=FALSE): matched by a claim -> FP, else -> TN.
  * confusion matrix + precision / recall / accuracy / F1, overall & per type.

COVERAGE (structural completeness)
  * PAGE coverage  = distinct pages with claims / total pages (human-counted).
  * ELEMENT coverage = elements that produced >=1 claim / elements ai_parse found.

ROBUSTNESS (same document run twice)
  * fingerprint each claim (document, page, canonical_metric, value); compare the
    two runs. EXACT reproducibility for table/text; WITHIN-TOLERANCE for figures.
"""
from __future__ import annotations
import re
from collections import defaultdict, Counter


# ---------------------------------------------------------------------------
# normalisation helpers
# ---------------------------------------------------------------------------
def _norm_value(v) -> str:
    return re.sub(r"[,£$€%\s]", "", str(v if v is not None else "").lower())


def _as_number(v):
    try:
        return float(re.sub(r"[,£$€%\s]", "", str(v)))
    except (ValueError, TypeError):
        return None


def _norm_doc(name) -> str:
    """Document name without extension, lowercased, for matching."""
    s = str(name or "").lower().strip()
    return re.sub(r"\.(pptx|pdf|docx|xlsx)$", "", s)


def _field_words(text) -> set:
    return set(re.sub(r"[^a-z0-9 ]", " ", str(text or "").lower()).split())


def _field_matches(human_label, canonical) -> bool:
    """Fuzzy: the human label's words (mostly) appear in the canonical_metric."""
    hw = _field_words(human_label)
    if not hw:
        return True
    cw = _field_words(canonical)
    overlap = len(hw & cw)
    return overlap >= max(1, int(round(0.6 * len(hw))))   # >=60% of the human words


def _value_matches(extracted, truth, element_type, chart_tol):
    if _norm_value(extracted) == _norm_value(truth):
        return True
    if element_type in ("figure", "chart"):
        a, b = _as_number(extracted), _as_number(truth)
        if a is not None and b is not None and b != 0:
            return abs(a - b) / abs(b) <= chart_tol
    return False


def _claim_field(c):
    return getattr(c, "canonical_metric", None) or getattr(c, "field_name", "") or ""


def _claim_doc(c):
    return getattr(c, "source_document", None) or getattr(c, "source_document_id", "") or ""


# ---------------------------------------------------------------------------
# ACCURACY
# ---------------------------------------------------------------------------
def evaluate_accuracy(claims, truth_rows, chart_tol=0.03):
    """truth_rows: list of dicts (document,page,element_type,field_name,value,
    should_extract). Returns confusion matrix + metrics overall and per type."""
    positives = [t for t in truth_rows if str(t.get("should_extract", "")).strip().lower()
                 in ("true", "1", "yes", "y")]
    decoys = [t for t in truth_rows if t not in positives]

    # index claims by (doc, page) for quick lookup; track used so one claim can't
    # satisfy two truth rows.
    claims_by_dp = defaultdict(list)
    for c in claims:
        claims_by_dp[(_norm_doc(_claim_doc(c)), getattr(c, "page", None))].append(
            {"claim": c, "used": False})

    tp, fn = [], []
    for t in positives:
        key = (_norm_doc(t.get("document")), t.get("page"))
        et = (t.get("element_type") or "").lower()
        hit = None
        for rec in claims_by_dp.get(key, []):
            if rec["used"]:
                continue
            c = rec["claim"]
            if _field_matches(t.get("field_name"), _claim_field(c)) and \
               _value_matches(getattr(c, "value", None), t.get("value"), et, chart_tol):
                hit = rec
                break
        if hit:
            hit["used"] = True
            tp.append((hit["claim"], t))
        else:
            fn.append(t)

    # decoys: a claim matching a decoy -> FP; else TN
    fp, tn = [], []
    for t in decoys:
        key = (_norm_doc(t.get("document")), t.get("page"))
        matched = False
        for rec in claims_by_dp.get(key, []):
            c = rec["claim"]
            if _field_matches(t.get("field_name"), _claim_field(c)) and \
               _value_matches(getattr(c, "value", None), t.get("value"),
                              (t.get("element_type") or "").lower(), chart_tol):
                matched = True
                break
        (fp if matched else tn).append(t)

    def _metrics(ntp, nfp, nfn, ntn):
        prec = ntp / (ntp + nfp) if (ntp + nfp) else None
        rec = ntp / (ntp + nfn) if (ntp + nfn) else None
        acc = (ntp + ntn) / (ntp + ntn + nfp + nfn) if (ntp + ntn + nfp + nfn) else None
        f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
        return {"TP": ntp, "FP": nfp, "FN": nfn, "TN": ntn,
                "precision": _r(prec), "recall": _r(rec),
                "accuracy": _r(acc), "f1": _r(f1)}

    overall = _metrics(len(tp), len(fp), len(fn), len(tn))

    # per element type
    by_type = {}
    types = {(t.get("element_type") or "unknown").lower() for t in truth_rows}
    for ty in types:
        ttp = sum(1 for (_, t) in tp if (t.get("element_type") or "").lower() == ty)
        tfn = sum(1 for t in fn if (t.get("element_type") or "").lower() == ty)
        tfp = sum(1 for t in fp if (t.get("element_type") or "").lower() == ty)
        ttn = sum(1 for t in tn if (t.get("element_type") or "").lower() == ty)
        by_type[ty] = _metrics(ttp, tfp, tfn, ttn)

    return {
        "overall": overall,
        "by_element_type": by_type,
        "missed": [{"document": t.get("document"), "page": t.get("page"),
                    "field": t.get("field_name"), "value": t.get("value")} for t in fn],
        "false_positives": [{"document": t.get("document"), "page": t.get("page"),
                             "field": t.get("field_name"), "value": t.get("value")} for t in fp],
    }


# ---------------------------------------------------------------------------
# PAGE COUNT — read the RAW file (independent of ai_parse) so coverage catches
# EVERY skip, including pages ai_parse itself missed.
# ---------------------------------------------------------------------------
def count_pages(path):
    """Total pages/slides in the raw document, read directly from the file (NOT
    from ai_parse). PDF -> pdfplumber; PPTX -> python-pptx. Returns None for
    unsupported types (page coverage is then skipped for that doc)."""
    local = path
    if path.startswith("/Volumes"):
        local = path.replace("/Volumes", "/dbfs/Volumes")
    elif path.startswith("dbfs:"):
        local = path.replace("dbfs:", "/dbfs")
    low = path.lower()
    try:
        if low.endswith(".pdf"):
            import pdfplumber
            with pdfplumber.open(local) as pdf:
                return len(pdf.pages)
        if low.endswith((".pptx", ".ppt", ".potx")):
            from pptx import Presentation
            return len(Presentation(local).slides)
    except Exception:
        return None
    return None


def total_pages_for(submissions):
    """Build {document: total_pages} automatically from the raw files.
    document key = filename without extension (matches _norm_doc)."""
    import os
    out = {}
    for s in submissions:
        path = s.get("path") if isinstance(s, dict) else s
        if not path:
            continue
        name = os.path.basename(path)
        key = re.sub(r"\.(pptx|pdf|docx|xlsx|potx|ppt)$", "", name, flags=re.I)
        n = count_pages(path)
        if n is not None:
            out[key] = n
    return out


# ---------------------------------------------------------------------------
# COVERAGE
# ---------------------------------------------------------------------------
def evaluate_coverage(chunk_rows, total_pages_by_doc):
    """COVERAGE from the CHUNKS (search index) rows, which capture EVERY processed
    page -- including pages with text but NO values (silver/gold skip those). This
    makes coverage measure 'did we PROCESS every page', not 'did every page have a
    value'. A narrative/cover page has a chunk (so it counts as processed), while a
    genuinely SKIPPED page has no chunk (a real coverage gap).

    chunk_rows: list of dicts {source_document, page} (filter to the eval's
                entity_ref BEFORE passing in -- this is the entity-referenced set).
    total_pages_by_doc: {document: total_pages} -- the denominator (what should
                        exist). Human-counted or ai_parse's page count.
    """
    processed = defaultdict(set)   # doc -> set of pages that produced a chunk
    for r in chunk_rows:
        d = _norm_doc(r.get("source_document"))
        pg = r.get("page")
        if pg is not None:
            processed[d].add(pg)

    page_cov = {}
    for doc, total in total_pages_by_doc.items():
        nd = _norm_doc(doc)
        got = len(processed.get(nd, set()))
        page_cov[doc] = {
            "total_pages": total,
            "pages_processed": got,
            "coverage": _r(got / total) if total else None,
            "missing_pages": (total - got) if total else None,
        }

    tot = sum(v["total_pages"] for v in page_cov.values())
    got = sum(v["pages_processed"] for v in page_cov.values())
    return {
        "page_coverage": page_cov,
        "overall_page_coverage": _r(got / tot) if tot else None,
        "total_pages": tot, "pages_processed": got,
        "source": "chunks (every processed page, incl. value-less text)",
    }


# ---------------------------------------------------------------------------
# ROBUSTNESS (two runs)
# ---------------------------------------------------------------------------
def evaluate_robustness(claims_run_a, claims_run_b, chart_tol=0.03):
    """Compare two runs. EXACT reproducibility for table/text; WITHIN-TOLERANCE
    for figures. Fingerprint = (doc, page, canonical_metric, value)."""
    def _fp(c):
        return (_norm_doc(_claim_doc(c)), getattr(c, "page", None),
                _claim_field(c).lower().strip(),
                getattr(c, "element_type", None) or "", getattr(c, "value", None))

    A = [_fp(c) for c in claims_run_a]
    B = [_fp(c) for c in claims_run_b]

    # exact set comparison (ignores value tolerance)
    sa = Counter((d, p, f, et, _norm_value(v)) for (d, p, f, et, v) in A)
    sb = Counter((d, p, f, et, _norm_value(v)) for (d, p, f, et, v) in B)
    stable_exact = sum((sa & sb).values())
    only_a = sum((sa - sb).values())
    only_b = sum((sb - sa).values())

    # tolerance-aware: for figures, a value within tol counts as stable
    b_remaining = list((sb - sa).elements())
    a_remaining = list((sa - sb).elements())
    tol_recovered = 0
    for (d, p, f, et, v) in list(a_remaining):
        if et in ("figure", "chart"):
            av = _as_number(v)
            for (d2, p2, f2, et2, v2) in list(b_remaining):
                if (d, p, f) == (d2, p2, f2) and et2 == et:
                    bv = _as_number(v2)
                    if av is not None and bv is not None and bv != 0 and abs(av-bv)/abs(bv) <= chart_tol:
                        tol_recovered += 1
                        b_remaining.remove((d2, p2, f2, et2, v2))
                        break

    total = stable_exact + only_a + only_b
    stable_tol = stable_exact + tol_recovered

    # per type exact
    def _type_rate(ty):
        at = Counter((d,p,f,_norm_value(v)) for (d,p,f,et,v) in A if et==ty)
        bt = Counter((d,p,f,_norm_value(v)) for (d,p,f,et,v) in B if et==ty)
        st = sum((at & bt).values()); tot = st + sum((at-bt).values()) + sum((bt-at).values())
        return {"stable": st, "total": tot, "reproducibility": _r(st/tot) if tot else None}

    types = {et for (_,_,_,et,_) in A} | {et for (_,_,_,et,_) in B}
    return {
        "claims_run_a": len(claims_run_a), "claims_run_b": len(claims_run_b),
        "exact_reproducibility": _r(stable_exact / total) if total else None,
        "within_tolerance_reproducibility": _r(stable_tol / total) if total else None,
        "stable_exact": stable_exact, "only_in_a": only_a, "only_in_b": only_b,
        "by_element_type_exact": {ty: _type_rate(ty) for ty in types},
    }


def _r(x, n=4):
    return round(x, n) if isinstance(x, (int, float)) else x


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------
def full_report(accuracy=None, coverage=None, robustness=None) -> str:
    L = ["=" * 64, "  EXTRACTION EVALUATION — Accuracy · Coverage · Robustness", "=" * 64]
    if accuracy:
        o = accuracy["overall"]
        L += ["", "ACCURACY (vs human ground truth)",
              f"  Confusion:  TP={o['TP']}  FP={o['FP']}  FN={o['FN']}  TN={o['TN']}",
              f"  Precision:  {_p(o['precision'])}   (of extracted, how much correct)",
              f"  Recall:     {_p(o['recall'])}   (of labelled, how much we got)",
              f"  Accuracy:   {_p(o['accuracy'])}   (reals extracted + decoys ignored)",
              f"  F1:         {_p(o['f1'])}",
              "  By element type:"]
        for ty, m in accuracy["by_element_type"].items():
            L.append(f"    {ty:8s} P={_p(m['precision'])} R={_p(m['recall'])} "
                     f"Acc={_p(m['accuracy'])}  (TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']})")
        if accuracy["missed"]:
            L.append(f"  MISSED ({len(accuracy['missed'])}):")
            for m in accuracy["missed"][:8]:
                L.append(f"    {m['document']} p{m['page']} {m['field']}={m['value']}")
    if coverage:
        L += ["", "COVERAGE (pages PROCESSED, from chunks — incl. value-less text)",
              f"  Overall PAGE coverage: {_p(coverage['overall_page_coverage'])} "
              f"({coverage['pages_processed']}/{coverage['total_pages']} pages)"]
        for doc, v in coverage["page_coverage"].items():
            flag = "  <-- MISSING PAGES (skipped)" if v["missing_pages"] else ""
            L.append(f"    {doc}: {v['pages_processed']}/{v['total_pages']} "
                     f"({_p(v['coverage'])}){flag}")
    if robustness:
        r = robustness
        L += ["", "ROBUSTNESS (same document, two runs)",
              f"  Run A: {r['claims_run_a']} claims   Run B: {r['claims_run_b']} claims",
              f"  EXACT reproducibility:          {_p(r['exact_reproducibility'])}",
              f"  WITHIN-TOLERANCE reproducibility: {_p(r['within_tolerance_reproducibility'])}",
              f"  (stable={r['stable_exact']} only_A={r['only_in_a']} only_B={r['only_in_b']})",
              "  By element type (exact):"]
        for ty, m in r["by_element_type_exact"].items():
            L.append(f"    {ty:8s} {_p(m['reproducibility'])}  ({m['stable']}/{m['total']})")
    L += ["", "=" * 64]
    return "\n".join(L)


def _p(x):
    return f"{x*100:.1f}%" if isinstance(x, (int, float)) else "n/a"
