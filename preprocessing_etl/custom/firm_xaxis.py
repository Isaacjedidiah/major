"""FIRM-SPECIFIC x-axis period resolution for charts with poor x-axis labelling.

Scoped to named firms only (FIRM_XAXIS). For a firm NOT listed, this does
nothing and the normal (working) chart path is used — so global success is
unaffected.

The problem it fixes (one company's charts): values are read fine but the
PERIOD each value belongs to is wrong, because the x-axis labels are sparse /
have a varying start month / miss the first or last label. Approach:
  * group chart elements into OBSERVATION SLOTS by x-position (one slot = one
    period on the axis),
  * align printed labels to slots, rejecting labels that don't cleanly fit,
  * anchor the timeline to a label that carries a YEAR, then reconstruct one
    period per slot by the monthly step — filling missing first/last/middle
    labels and rolling the year over correctly,
  * VALIDATE the reconstruction against every printed label; if any printed
    label disagrees, return valid=False so the caller can flag for review.

resolve_axis_labels() is the standalone, tested resolver. apply_to_claims()
wires it into the pipeline: it maps each Sonnet chart claim to its resolved
period (via the element positions), for the configured firm only.
"""
from __future__ import annotations
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
from statistics import median


# ---------------------------------------------------------------------------
# Firms to apply x-axis resolution to, keyed by entity_id or entity_name
# (case-insensitive, substring-tolerant). Each value is config for that firm.
# Empty -> no firm affected (global behaviour unchanged).
# ---------------------------------------------------------------------------
FIRM_XAXIS: dict = {
    # "IkeBank": {"monthly_step": 1, "x_tolerance": 8},
}


_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}


def _firm_config(firm_id=None, firm_name=None):
    """Return the x-axis config for this firm, or None if not configured."""
    if not FIRM_XAXIS:
        return None
    cands = [str(x).strip().lower().replace(" ", "") for x in (firm_id, firm_name) if x]
    for key, cfg in FIRM_XAXIS.items():
        k = str(key).strip().lower().replace(" ", "")
        if k and any(k in c or c in k for c in cands):
            return cfg or {}
    return None


def resolve_axis_labels(detected_bars, detected_labels, monthly_step=1,
                        x_tolerance=8):
    """Resolve each chart element's period. Returns (id_to_period, valid).
    Returns (id_to_period, status):
      status "geometry" -> periods resolved by geometry, use them
      status "normal"   -> not a date/numeric sparse-axis case (text/pie/
                           categorical) -> caller should use the NORMAL Sonnet
                           chart extraction, NOT quarantine
      status "flag"     -> a date/numeric case that failed validation -> review
    """
    if not detected_bars:
        return {}, "normal"

    elements = sorted(detected_bars, key=lambda b: float(b.get('x_center', 0)))

    # --- STEP 1: cluster markers at the SAME x into raw groups (stacked bars,
    #     a bar + its dot, overlapping series share ONE x-position). ---
    raw = []
    for element in elements:
        x = float(element.get("x_center", 0))
        if raw and abs(x - raw[-1]["x_center"]) <= x_tolerance:
            raw[-1]["elements"].append(element)
            raw[-1]["x_center"] = sum(float(e.get("x_center", 0))
                                      for e in raw[-1]["elements"]) / len(raw[-1]["elements"])
        else:
            raw.append({"x_center": x, "elements": [element]})

    # --- STEP 2: derive the EVEN grid step from the gaps between raw groups.
    #     Using the MEDIAN positive gap makes the step robust to a missing
    #     position (a gap that is 2x the step) or an extra marker. ---
    typical_gap = None
    if len(raw) >= 2:
        gaps = [raw[i+1]["x_center"] - raw[i]["x_center"] for i in range(len(raw)-1)]
        pos = [g for g in gaps if g > 0]
        typical_gap = median(pos) if pos else None

    # --- STEP 3: place each raw group on an EVEN GRID. slot_index = how many
    #     steps it sits from the FIRST group. A missing period leaves a GAP in
    #     the indices (so the month maths stays correct); two groups a double-gap
    #     apart get indices 0 and 2, NOT 0 and 1. This fixes:
    #       * leading unlabeled period dropped (it still gets its own index),
    #       * off-by-N from phantom/merged slots (index comes from position),
    #       * stacked bars / dots sharing a position (same index). ---
    slots = []
    if typical_gap and typical_gap > 0:
        x0 = raw[0]["x_center"]
        used = {}
        for g in raw:
            idx = int(round((g["x_center"] - x0) / typical_gap))
            if idx in used:
                # two groups collapsed onto one index: keep separate by nudging
                # the later one forward (defensive; rare).
                idx = max(used) + 1
            used[idx] = True
            slots.append({"x_center": g["x_center"], "elements": g["elements"],
                          "slot_index": idx})
    else:
        for i, g in enumerate(raw):
            slots.append({"x_center": g["x_center"], "elements": g["elements"],
                          "slot_index": i})

    # align each printed label to its nearest slot. A label's x can sit a little
    # off its bar (multi-word labels, centring differences), so we use a GENEROUS
    # threshold and NEVER drop a label that carries a YEAR -- losing the year
    # anchor is worse than a slightly-off alignment, and is the main cause of
    # "years not added". A label landing on an already-claimed slot takes the
    # next-nearest free slot instead of overwriting.
    def _has_year(txt):
        t = re.sub(r"\s+", " ", str(txt).replace("'", "").replace("`", "").strip())
        return bool(re.search(r"\b\d{2,4}\b", t)) or bool(re.search(r"\b\d{2}$", t))

    assignments = {}
    # place year-bearing labels FIRST (they anchor the timeline) then the rest,
    # each nearest-first, skipping slots already taken.
    ordered = sorted(detected_labels,
                     key=lambda l: (not _has_year(l.get("text", "")),
                                    float(l.get("x_center", 0))))
    for label in ordered:
        if not slots:
            break
        lx = float(label.get("x_center", 0))
        text = label.get("text", "")
        is_year = _has_year(text)
        # candidate slots nearest-first
        cands = sorted(slots, key=lambda s: abs(s["x_center"] - lx))
        placed = False
        for cand in cands:
            if cand["slot_index"] in assignments:
                continue                      # slot already has a label
            # reject only NON-year labels that are implausibly far (>0.6 gap)
            if typical_gap and not is_year:
                if abs(cand["x_center"] - lx) / typical_gap > 0.6:
                    break
            assignments[cand["slot_index"]] = text
            placed = True
            break
        # a year label that found no free slot still anchors on its nearest slot
        if not placed and is_year:
            best = cands[0]
            assignments[best["slot_index"]] = text

    def _parse_date(text):
        cleaned = re.sub(r"\s+", " ", str(text).replace("'", "").replace("`", "").strip())
        for fmt in ("%B %Y", "%b %Y", "%B %y", "%b %y", "%B-%Y", "%b-%Y",
                    "%B-%y", "%b-%y", "%Y-%m"):
            try:
                p = datetime.strptime(cleaned, fmt)
                return {"month": p.month, "year": p.year, "had_year": True}
            except ValueError:
                continue
        m = _MONTHS.get(cleaned.lower().rstrip("."))
        if m:
            return {"month": m, "year": None, "had_year": False}
        return None

    def _parse_number(text):
        """A plain integer/number x-label (e.g. 'months spent' = 1,2,3)."""
        cleaned = str(text).strip().rstrip(".")
        m = re.fullmatch(r"-?\d+(?:\.\d+)?", cleaned)
        return float(cleaned) if m else None

    # classify the axis by the MAJORITY of printed labels: date vs numeric.
    date_parses, num_parses = [], []
    for slot_index, text in assignments.items():
        d = _parse_date(text)
        if d:
            date_parses.append((slot_index, d, str(text).strip()))
        else:
            n = _parse_number(text)
            if n is not None:
                num_parses.append((slot_index, n, str(text).strip()))

    is_date_axis = len(date_parses) >= len(num_parses) and len(date_parses) > 0
    is_num_axis = (not is_date_axis) and len(num_parses) > 0

    # ---------- DATE AXIS (what worked: anchor to a year, reconstruct, validate) ----------
    if is_date_axis:
        year_anchors = [(i, d) for (i, d, _) in date_parses if d["had_year"]]
        if not year_anchors:
            # MONTHS but NO year on the axis. The axis can still be MISPLACED
            # (values attached to the wrong month), so we DO use geometry to fix
            # the month SEQUENCE by position — but we do NOT invent a year. We
            # anchor on the first month label and reconstruct month names only;
            # the YEAR comes from the chart title downstream (it's already in the
            # canonical_metric). Output = month name (no year).
            date_parses.sort(key=lambda x: x[0])
            anchor_slot, a0, _ = date_parses[0]
            anchor_month = a0["month"]   # 1-12

            slot_mname = {}
            for slot in slots:
                diff = slot["slot_index"] - anchor_slot
                mnum = ((anchor_month - 1) + diff * monthly_step) % 12 + 1
                slot_mname[slot["slot_index"]] = datetime(2000, mnum, 1).strftime("%B")

            # validate month sequence against every printed month label
            valid = True
            for (i, d, _) in date_parses:
                if slot_mname[i] != datetime(2000, d["month"], 1).strftime("%B"):
                    valid = False

            out = {}
            for slot in slots:
                for e in slot["elements"]:
                    out[e["id"]] = slot_mname[slot["slot_index"]]
            # status "geometry_month" -> periods are MONTH-ONLY; caller keeps the
            # title/year from the canonical_metric and uses these months.
            return out, ("geometry_month" if valid else "flag")
        year_anchors.sort(key=lambda x: x[0])
        anchor_slot, a = year_anchors[0]
        anchor_date = datetime(a["year"], a["month"], 1)

        slot_period = {}
        for slot in slots:
            diff = slot["slot_index"] - anchor_slot
            inferred = anchor_date + relativedelta(months=(diff * monthly_step))
            slot_period[slot["slot_index"]] = inferred.strftime("%B %Y")

        valid = True
        for (i, d, _) in date_parses:
            dd = datetime.strptime(slot_period[i], "%B %Y")
            if dd.month != d["month"] or (d["had_year"] and dd.year != d["year"]):
                valid = False

        out = {}
        for slot in slots:
            for e in slot["elements"]:
                out[e["id"]] = slot_period[slot["slot_index"]]
        return out, ("geometry" if valid else "flag")

    # ---------- NUMERIC AXIS (same pattern: anchor to a number, reconstruct, validate) ----------
    if is_num_axis:
        num_parses.sort(key=lambda x: x[0])
        # derive the numeric step per slot from the two furthest-apart number labels
        (i0, n0, _), (i1, n1, _) = num_parses[0], num_parses[-1]
        span = i1 - i0
        num_step = ((n1 - n0) / span) if span else 1
        if num_step == 0:
            num_step = 1
        anchor_slot, anchor_num = i0, n0

        def _fmt(v):
            return str(int(v)) if float(v).is_integer() else str(v)

        slot_val = {}
        for slot in slots:
            v = anchor_num + (slot["slot_index"] - anchor_slot) * num_step
            slot_val[slot["slot_index"]] = _fmt(v)

        valid = True
        for (i, n, _) in num_parses:
            if slot_val[i] != _fmt(n):
                valid = False

        out = {}
        for slot in slots:
            for e in slot["elements"]:
                out[e["id"]] = slot_val[slot["slot_index"]]
        return out, ("geometry" if valid else "flag")

    # ---------- neither date nor numeric ----------
    # Not an axis the geometry is for (text labels, pie slices, categorical).
    # These do NOT have the sparse date/number alignment problem — the normal
    # Sonnet chart prompt reads them correctly. Signal "normal" so the caller
    # uses the standard extraction instead of quarantining.
    return {}, "normal"


# ---------------------------------------------------------------------------
# Pipeline hook — firm-scoped. For a CONFIGURED firm, re-reads each chart figure
# to capture element + label POSITIONS, resolves the period per slot, and sets
# each matching chart claim's period (as_at_date / canonical_metric suffix).
# For an unconfigured firm this is a no-op -> global behaviour unchanged.
# ---------------------------------------------------------------------------
_XAXIS_READ_SCHEMA = {
    "name": "record_chart",
    "description": "Record a chart's bars and x-axis labels with positions.",
    "parameters": {
        "type": "object",
        "properties": {
            "bars": {"type": "array", "items": {"type": "object", "properties": {
                "id": {"type": "integer"}, "series": {"type": "string"},
                "value": {"type": "string"}, "x_center": {"type": "number"}},
                "required": ["id", "value", "x_center"]}},
            "labels": {"type": "array", "items": {"type": "object", "properties": {
                "text": {"type": "string"}, "x_center": {"type": "number"}},
                "required": ["text", "x_center"]}},
        }, "required": ["bars", "labels"]},
}

_XAXIS_READ_PROMPT = (
    "Read this chart image. Record EVERY bar/point and EVERY printed x-axis label "
    "with its horizontal PIXEL position (x_center). For a bar's value use the "
    "number PRINTED on/beside it exactly. Include bars even with no printed label."
)


def resolve_periods_for_figure(img_b64, run_async_fn, llm_client,
                               model_key="tier2", monthly_step=1, x_tolerance=8):
    """For ONE figure image: read positions, resolve each element's period.
    Returns (value_to_period: {(series,value): period}, valid). Best-effort."""
    try:
        resp = run_async_fn(llm_client.extract(
            model_key, _XAXIS_READ_PROMPT, "", _XAXIS_READ_SCHEMA,
            image_base64=img_b64))
        ta = getattr(resp, "tool_arguments", None)
        if not isinstance(ta, dict):
            return {}, False
        bars = ta.get("bars", [])
        labels = ta.get("labels", [])
        id_to_period, status = resolve_axis_labels(
            bars, labels, monthly_step=monthly_step, x_tolerance=x_tolerance)
        # map (series,value) -> period so claims can be matched
        vp = {}
        for b in bars:
            period = id_to_period.get(b.get("id"))
            if period and period != "FLAG_REVIEW":
                vp[(str(b.get("series", "")), str(b.get("value", "")))] = period
        return vp, status      # status: "geometry" | "normal" | "flag"
    except Exception:
        return {}, "normal"    # on error, fall back to normal extraction


# ===========================================================================
# ORDER-BASED period resolution (robust to Sonnet's position non-determinism).
# Instead of pixel x-positions (which Sonnet reads inconsistently run-to-run),
# this works from the SLOT ORDER: Sonnet returns observations grouped by x-axis
# position IN ORDER (slot 0, 1, 2 ... left to right), and says which slot carries
# each printed label. The geometry anchors on a labelled slot's INDEX and fills
# every slot's period by counting from the anchor. Order is stable across runs,
# so the output is consistent.
# ===========================================================================
def resolve_periods_by_order(slots_in, monthly_step=1):
    """slots_in: list of {index, label} in left-to-right order (one per x-axis
    position). label is the printed x-axis label for that slot, or "" if none.
    Returns {slot_index: period_string}, status.

    status: "geometry" (full dates, year anchored), "geometry_month" (month-only,
    no year on axis), "flag" (validation failed), "normal" (not a date/number axis).
    """
    if not slots_in:
        return {}, "normal"
    # order + reindex 0..n-1 (trust the given order; ignore any pixel data)
    slots = sorted(slots_in, key=lambda s: s.get("index", 0))
    for i, s in enumerate(slots):
        s["_i"] = i

    def _parse_date(text):
        cleaned = re.sub(r"\s+", " ", str(text).replace("'", "").replace("`", "").strip())
        for fmt in ("%B %Y", "%b %Y", "%B %y", "%b %y", "%B-%Y", "%b-%Y",
                    "%B-%y", "%b-%y", "%Y-%m"):
            try:
                p = datetime.strptime(cleaned, fmt)
                return {"month": p.month, "year": p.year, "had_year": True}
            except ValueError:
                continue
        m = _MONTHS.get(cleaned.lower().rstrip("."))
        if m:
            return {"month": m, "year": None, "had_year": False}
        return None

    def _parse_number(text):
        cleaned = str(text).strip().rstrip(".")
        return float(cleaned) if re.fullmatch(r"-?\d+(?:\.\d+)?", cleaned) else None

    # classify labelled slots as date vs numeric
    date_anchors, num_anchors = [], []
    for s in slots:
        txt = str(s.get("label", "")).strip()
        if not txt:
            continue
        d = _parse_date(txt)
        if d:
            date_anchors.append((s["_i"], d, txt))
            continue
        n = _parse_number(txt)
        if n is not None:
            num_anchors.append((s["_i"], n, txt))

    is_date = len(date_anchors) >= len(num_anchors) and len(date_anchors) > 0
    is_num = (not is_date) and len(num_anchors) > 0

    # ---- DATE AXIS ----
    if is_date:
        year_anchors = [(i, d) for (i, d, _) in date_anchors if d["had_year"]]
        if year_anchors:
            ya = sorted(year_anchors, key=lambda x: x[0])
            anchor_i, a = ya[0]
            anchor_date = datetime(a["year"], a["month"], 1)

            # MULTI-ANCHOR consistency: with 2+ year anchors we can CHECK that the
            # slot gaps match the month gaps. If they disagree, Sonnet miscounted
            # the bars (dropped/added one) -> we cannot trust the fill -> flag.
            # This is what catches the "bad label" charts: the printed labels are
            # the skeleton; if the bars don't fit them, we say so.
            anchors_consistent = True
            if len(ya) >= 2:
                for (i2, d2) in ya[1:]:
                    slot_gap = i2 - anchor_i
                    months_gap = ((d2["year"] - a["year"]) * 12
                                  + (d2["month"] - a["month"]))
                    # expected slot gap = months_gap / monthly_step
                    if monthly_step and (months_gap % monthly_step == 0):
                        expected = months_gap // monthly_step
                        if expected != slot_gap:
                            anchors_consistent = False
                    else:
                        anchors_consistent = False

            period = {}
            for s in slots:
                diff = s["_i"] - anchor_i
                period[s["index"]] = (anchor_date
                    + relativedelta(months=diff * monthly_step)).strftime("%B %Y")
            # validate EVERY labelled slot resolves to its printed label
            valid = anchors_consistent
            for (i, d, _) in date_anchors:
                si = next(s["index"] for s in slots if s["_i"] == i)
                dd = datetime.strptime(period[si], "%B %Y")
                if dd.month != d["month"] or (d["had_year"] and dd.year != d["year"]):
                    valid = False
            # status: geometry (trust) if anchors agree AND labels validate;
            # otherwise flag (worked, but inconsistent -> do not fully trust).
            return period, ("geometry" if valid else "flag")
        # months but no year -> month-only (title supplies the year downstream)
        anchor_i, a0, _ = sorted(date_anchors, key=lambda x: x[0])[0]
        period = {}
        for s in slots:
            diff = s["_i"] - anchor_i
            mnum = ((a0["month"] - 1) + diff * monthly_step) % 12 + 1
            period[s["index"]] = datetime(2000, mnum, 1).strftime("%B")
        valid = True
        for (i, d, _) in date_anchors:
            si = next(s["index"] for s in slots if s["_i"] == i)
            if period[si] != datetime(2000, d["month"], 1).strftime("%B"):
                valid = False
        return period, ("geometry_month" if valid else "flag")

    # ---- NUMERIC AXIS ----
    if is_num:
        na = sorted(num_anchors, key=lambda x: x[0])
        (i0, n0, _), (i1, n1, _) = na[0], na[-1]
        span = i1 - i0
        step = ((n1 - n0) / span) if span else 1
        step = step or 1
        def _fmt(v): return str(int(v)) if float(v).is_integer() else str(v)
        period = {}
        for s in slots:
            v = n0 + (s["_i"] - i0) * step
            period[s["index"]] = _fmt(v)
        valid = all(_fmt(n) == period[next(s["index"] for s in slots if s["_i"] == i)]
                    for (i, n, _) in num_anchors)
        return period, ("geometry" if valid else "flag")

    return {}, "normal"
