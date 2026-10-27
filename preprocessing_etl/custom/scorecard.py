"""Per-run pipeline SCORECARD — coverage, quality, robustness and cost.

Produces an objective, comparable snapshot after each pipeline run, so two
approaches (or two runs) can be evaluated on measured outcomes rather than on
design preference. Every number here is computed from the run's own output —
the Gold/silver/quarantine/audit tables and the BatchResult — so it is
reproducible and can be re-run for any pipeline over the same documents.

Honest scope of each dimension:
  * COVERAGE   — how much was extracted (counts). Objective.
  * QUALITY    — PROXIES for correctness: grounding rate, confidence, citation
                 tier mix. These are strong signals, not a substitute for a
                 manual ground-truth check, which should accompany them.
  * ROBUSTNESS — how gracefully it handled the inputs: quarantine rate, review
                 rate, coverage across document formats.
  * COST       — LLM spend + ai_parse DBU estimate, per document and per metric.
"""

from __future__ import annotations

from typing import Optional


def build_scorecard(result, storage=None, prefix: str = "",
                    ai_parse_cost_per_page: float = 0.005,
                    pages_processed: Optional[int] = None) -> dict:
    """Compute the four-dimension scorecard from a BatchResult (and, if given,
    the storage layer to read gold/review/quarantine counts).

    Returns a flat dict of metrics; pass it to ``scorecard_table`` to render.
    """
    docs = max(1, getattr(result, "documents", 0) or 0)
    claims = getattr(result, "claims_total", 0) or 0
    gold = getattr(result, "gold_total", 0) or 0
    review = getattr(result, "review_total", 0) or 0
    quar = getattr(result, "quarantined", 0) or 0
    indexed = getattr(result, "indexed_chunks", 0) or 0
    llm_cost = getattr(result, "total_cost_usd", 0.0) or 0.0

    total_docs = docs + quar          # attempted = processed + failed
    # --- COVERAGE ---
    coverage = {
        "documents_processed": docs,
        "total_metrics": claims,
        "gold_metrics": gold,
        "metrics_per_document": round(claims / docs, 1),
        "indexed_chunks": indexed,
    }
    # --- QUALITY (proxies) ---
    quality = {
        "gold_rate_pct": round(100 * gold / claims, 1) if claims else 0.0,
        "review_rate_pct": round(100 * review / claims, 1) if claims else 0.0,
        # filled from the tables when storage is provided (see below)
        "grounded_pct": None,
        "avg_confidence": None,
        "parsed_tier_pct": None,
    }
    # --- ROBUSTNESS ---
    robustness = {
        "quarantine_rate_pct": round(100 * quar / total_docs, 1) if total_docs else 0.0,
        "success_rate_pct": round(100 * docs / total_docs, 1) if total_docs else 0.0,
        "failures": len(getattr(result, "failures", []) or []),
    }
    # --- COST ---
    pages = pages_processed or docs        # fall back to docs if pages unknown
    ai_parse_cost = pages * ai_parse_cost_per_page
    total_cost = llm_cost + ai_parse_cost
    cost = {
        "llm_cost_usd": round(llm_cost, 4),
        "ai_parse_cost_usd": round(ai_parse_cost, 4),
        "total_cost_usd": round(total_cost, 4),
        "cost_per_document_usd": round(total_cost / docs, 4),
        "cost_per_metric_usd": round(total_cost / claims, 6) if claims else 0.0,
    }

    # enrich QUALITY from the gold tables if storage is available
    if storage is not None:
        try:
            _fill_quality_from_gold(quality, storage, prefix)
        except Exception:
            pass

    return {"coverage": coverage, "quality": quality,
            "robustness": robustness, "cost": cost}


def _fill_quality_from_gold(quality: dict, storage, prefix: str) -> None:
    """Read grounding / confidence / tier stats from the gold rows (best-effort)."""
    rows = []
    try:
        for layer in ("gold",):
            data = storage.read_jsonl(layer)
            if data:
                rows.extend(data)
    except Exception:
        rows = []
    if not rows:
        return
    confs = [r.get("confidence") for r in rows if isinstance(r.get("confidence"), (int, float))]
    grounded = sum(1 for r in rows if str(r.get("grounding", "")).lower() == "grounded")
    parsed = sum(1 for r in rows if str(r.get("citation_tier", "")).lower() == "parsed")
    n = len(rows)
    quality["grounded_pct"] = round(100 * grounded / n, 1) if n else None
    quality["avg_confidence"] = round(sum(confs) / len(confs), 3) if confs else None
    quality["parsed_tier_pct"] = round(100 * parsed / n, 1) if n else None


def scorecard_table(scorecard: dict, label: str = "this run") -> str:
    """Render the scorecard as a readable text table (for the notebook / an
    email). Two scorecards can be rendered side by side to compare approaches."""
    lines = [f"===== PIPELINE SCORECARD — {label} =====", ""]
    for section in ("coverage", "quality", "robustness", "cost"):
        lines.append(section.upper())
        for k, v in scorecard[section].items():
            lines.append(f"  {k:28} {v}")
        lines.append("")
    return "\n".join(lines)


def compare_scorecards(a: dict, b: dict, label_a="A", label_b="B") -> str:
    """Render two scorecards side by side for an evidence-based comparison."""
    lines = [f"{'metric':32} {label_a:>16} {label_b:>16}", "-" * 66]
    for section in ("coverage", "quality", "robustness", "cost"):
        lines.append(section.upper())
        for k in a[section]:
            va, vb = a[section].get(k), b[section].get(k)
            lines.append(f"  {k:30} {str(va):>16} {str(vb):>16}")
    return "\n".join(lines)
