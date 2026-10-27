"""Evaluation metrics for the content-extraction pipeline.

Four axes, chosen so the evaluation catches the REAL failure modes of an
EXTRACTION system — not just the ones a generative-AI eval looks for:

  1. COVERAGE   — did we extract EVERYTHING? (pages AND modality/element types).
                  This is the axis a hallucination-only eval misses: a page
                  skipped silently produces NO hallucination, so a faithfulness
                  metric would never flag it. For extraction, MISSING content is
                  the primary risk, so coverage leads.
  2. FAITHFULNESS — is what we extracted grounded in the source (not fabricated)?
                  Matters most for the ESTIMATIVE parts (chart/vision reads),
                  which are flagged llm_estimated.
  3. SPEED      — throughput: documents/pages per second, wall-clock.
  4. COST       — LLM spend (chart/vision calls) per document/page.

The module is reporting-only: it reads what the run produced (BatchResult, the
parsed elements, the claims, timing, token usage) and returns structured metrics
+ a readable table. It never changes extraction behaviour.
"""
from __future__ import annotations
from collections import Counter
from typing import Optional


# ---------------------------------------------------------------------------
# 1. COVERAGE — pages and modality
# ---------------------------------------------------------------------------
def coverage_metrics(parsed_elements, claims, pages_in_doc=None) -> dict:
    """Coverage for ONE document.

    parsed_elements : the ParsedElement list ai_parse returned (what SHOULD be
                      extractable).
    claims          : the claims actually produced.
    pages_in_doc    : total pages in the document (from ai_parse's pages array);
                      if None, inferred from the elements' page numbers.

    Catches the '20 skipped pages' failure: compares pages that HAVE elements
    (and claims) against the document's total page count, and reports any page
    that produced no claims.
    """
    # ---- PAGE coverage ----
    el_pages = {getattr(e, "page", None) for e in parsed_elements
                if getattr(e, "page", None) is not None}
    claim_pages = {getattr(c, "page", None) for c in claims
                   if getattr(c, "page", None) is not None}
    total_pages = pages_in_doc if pages_in_doc else (max(el_pages) + 1 if el_pages else 0)

    pages_with_elements = len(el_pages)
    pages_with_claims = len(claim_pages)
    # pages ai_parse saw content on but we produced NO claims for (a real gap)
    pages_missing_claims = sorted(p for p in el_pages if p not in claim_pages)
    # pages ai_parse did not report elements for at all (possible skipped pages)
    pages_no_elements = ([p for p in range(total_pages) if p not in el_pages]
                         if pages_in_doc else [])

    page_cov = (pages_with_claims / total_pages) if total_pages else 0.0

    # ---- MODALITY / element-type coverage ----
    el_by_type = Counter(getattr(e, "el_type", "?") for e in parsed_elements)
    # which element types produced at least one claim?
    claim_src_pages = {(getattr(c, "page", None)) for c in claims}
    # map claims back to element types via element_type on the claim where present
    claim_by_type = Counter(getattr(c, "element_type", None) or "unknown" for c in claims)
    # element types ai_parse found but that produced NO claims of that type
    types_with_elements = set(el_by_type)
    types_with_claims = {t for t in claim_by_type if t and t != "unknown"}
    types_missing = sorted(t for t in types_with_elements
                           if t not in types_with_claims
                           and t not in ("page_header", "page_footer", "page_number"))

    return {
        "total_pages": total_pages,
        "pages_with_elements": pages_with_elements,
        "pages_with_claims": pages_with_claims,
        "pages_missing_claims": pages_missing_claims,     # had content, no claims
        "pages_no_elements": pages_no_elements,           # ai_parse saw nothing
        "page_coverage": round(page_cov, 4),
        "elements_by_type": dict(el_by_type),
        "claims_by_type": {k: v for k, v in claim_by_type.items()},
        "modality_types_missing": types_missing,          # a type extracted nothing
        "n_elements": len(parsed_elements),
        "n_claims": len(claims),
    }


# ---------------------------------------------------------------------------
# 2. FAITHFULNESS — grounded vs estimated
# ---------------------------------------------------------------------------
def faithfulness_metrics(claims, grounded_fn=None) -> dict:
    """Faithfulness for ONE document's claims.

    Splits claims by provenance tier:
      * parsed / rule-based  -> deterministic, grounded by construction
      * llm_estimated        -> vision/chart estimates (the hallucination-risk set)
    If grounded_fn(claim) is provided (e.g. a check that the value appears in the
    source text), it is applied to report a grounded fraction; otherwise we report
    the tier split, which is the honest signal we have without ground-truth labels.
    """
    tiers = Counter(str(getattr(c, "citation_tier", "") or "unknown") for c in claims)
    n = len(claims) or 1
    estimated = sum(v for k, v in tiers.items() if "estimated" in k.lower())
    parsed = sum(v for k, v in tiers.items()
                 if "parsed" in k.lower() or "rule" in k.lower())

    out = {
        "n_claims": len(claims),
        "by_tier": dict(tiers),
        "grounded_parsed_frac": round(parsed / n, 4),      # deterministic share
        "llm_estimated_frac": round(estimated / n, 4),     # hallucination-risk share
    }
    if grounded_fn is not None:
        checked = [c for c in claims]
        grounded = sum(1 for c in checked if grounded_fn(c))
        out["grounded_verified_frac"] = round(grounded / (len(checked) or 1), 4)
    return out


# ---------------------------------------------------------------------------
# 3. SPEED
# ---------------------------------------------------------------------------
def speed_metrics(elapsed_seconds: float, documents: int,
                  total_pages: int = 0, claims: int = 0) -> dict:
    """Throughput for the run."""
    el = max(elapsed_seconds, 1e-9)
    return {
        "elapsed_seconds": round(elapsed_seconds, 2),
        "docs_per_min": round(documents / el * 60, 2),
        "pages_per_min": round(total_pages / el * 60, 2) if total_pages else None,
        "seconds_per_doc": round(el / documents, 2) if documents else None,
        "seconds_per_page": round(el / total_pages, 3) if total_pages else None,
        "claims_per_min": round(claims / el * 60, 1) if claims else None,
    }


# ---------------------------------------------------------------------------
# 4. COST
# ---------------------------------------------------------------------------
def cost_metrics(sonnet_calls: int = 0, input_tokens: int = 0,
                 output_tokens: int = 0, documents: int = 0, total_pages: int = 0,
                 in_rate_per_m: float = 3.00, out_rate_per_m: float = 15.00,
                 ai_parse_pages: int = 0, ai_parse_rate_per_page: float = 0.0) -> dict:
    """LLM + ai_parse cost for the run. Rates default to tier2 (Sonnet) pricing;
    override with your actual rates."""
    llm_cost = (input_tokens / 1e6) * in_rate_per_m + (output_tokens / 1e6) * out_rate_per_m
    parse_cost = ai_parse_pages * ai_parse_rate_per_page
    total = llm_cost + parse_cost
    return {
        "sonnet_calls": sonnet_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "llm_cost_usd": round(llm_cost, 4),
        "ai_parse_cost_usd": round(parse_cost, 4),
        "total_cost_usd": round(total, 4),
        "cost_per_doc_usd": round(total / documents, 4) if documents else None,
        "cost_per_page_usd": round(total / total_pages, 5) if total_pages else None,
    }


# ---------------------------------------------------------------------------
# AGGREGATE — one evaluation across a run
# ---------------------------------------------------------------------------
def evaluate_run(per_doc_coverage: list, result=None, elapsed_seconds: float = 0.0,
                 sonnet_calls: int = 0, input_tokens: int = 0, output_tokens: int = 0,
                 all_claims: Optional[list] = None, cost_kwargs: Optional[dict] = None,
                 speed_kwargs: Optional[dict] = None) -> dict:
    """Combine per-document coverage + run-level faithfulness/speed/cost into one
    evaluation. per_doc_coverage is a list of coverage_metrics() dicts (one per
    document). result is the BatchResult (for document/claim totals)."""
    docs = getattr(result, "documents", len(per_doc_coverage)) if result else len(per_doc_coverage)
    total_pages = sum(d.get("total_pages", 0) for d in per_doc_coverage)
    pages_with_claims = sum(d.get("pages_with_claims", 0) for d in per_doc_coverage)
    pages_missing = sum(len(d.get("pages_missing_claims", [])) for d in per_doc_coverage)
    pages_no_elements = sum(len(d.get("pages_no_elements", [])) for d in per_doc_coverage)
    claims_total = getattr(result, "claims_total", 0) if result else (
        len(all_claims) if all_claims else sum(d.get("n_claims", 0) for d in per_doc_coverage))

    agg_coverage = {
        "documents": docs,
        "total_pages": total_pages,
        "pages_with_claims": pages_with_claims,
        "page_coverage": round(pages_with_claims / total_pages, 4) if total_pages else 0.0,
        "pages_missing_claims_total": pages_missing,     # had content, no claims
        "pages_no_elements_total": pages_no_elements,    # possibly skipped pages
    }

    faithfulness = faithfulness_metrics(all_claims or [])
    speed = speed_metrics(elapsed_seconds, docs, total_pages, claims_total,
                          **(speed_kwargs or {}))
    cost = cost_metrics(sonnet_calls=sonnet_calls, input_tokens=input_tokens,
                        output_tokens=output_tokens, documents=docs,
                        total_pages=total_pages, **(cost_kwargs or {}))

    return {"coverage": agg_coverage, "faithfulness": faithfulness,
            "speed": speed, "cost": cost}


def evaluation_table(ev: dict, label: str = "run") -> str:
    """Readable table of the evaluation."""
    lines = [f"=== EVALUATION — {label} ===", ""]
    cov = ev["coverage"]
    lines += ["COVERAGE (did we extract everything?)",
              f"  documents:            {cov['documents']}",
              f"  total pages:          {cov['total_pages']}",
              f"  pages with claims:    {cov['pages_with_claims']}",
              f"  PAGE COVERAGE:        {cov['page_coverage']*100:.1f}%",
              f"  pages w/ content but NO claims: {cov['pages_missing_claims_total']}  <-- gap",
              f"  pages ai_parse saw NOTHING on:  {cov['pages_no_elements_total']}  <-- possible skip",
              ""]
    f = ev["faithfulness"]
    lines += ["FAITHFULNESS (is it grounded?)",
              f"  claims:               {f['n_claims']}",
              f"  grounded/parsed:      {f['grounded_parsed_frac']*100:.1f}%",
              f"  llm_estimated:        {f['llm_estimated_frac']*100:.1f}%  (hallucination-risk set)",
              ""]
    s = ev["speed"]
    lines += ["SPEED",
              f"  elapsed:              {s['elapsed_seconds']}s",
              f"  docs/min:             {s['docs_per_min']}",
              f"  seconds/page:         {s['seconds_per_page']}",
              ""]
    c = ev["cost"]
    lines += ["COST",
              f"  sonnet calls:         {c['sonnet_calls']}",
              f"  total cost:           ${c['total_cost_usd']}",
              f"  cost/doc:             ${c['cost_per_doc_usd']}",
              f"  cost/page:            ${c['cost_per_page_usd']}",
              ""]
    return "\n".join(lines)
