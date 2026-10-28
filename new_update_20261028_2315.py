############################ CURRENCY & UNIT ############################
# FIX 1 — _SCALE_RE (add b for standalone billions). Find:
_SCALE_RE = re.compile(r"(?<![a-zA-Z])(mm|bn|billion|million|m|k)(?![a-zA-Z])", re.I)

# REPLACE
_SCALE_RE = re.compile(r"(?<![a-zA-Z])(mm|bn|billion|million|m|k|b)(?![a-zA-Z])", re.I)

# FIX 2 — _split_currency_unit (normalise standalone b → bn). Find:
    if pct_m:
        unit = pct_m.group(1)
    elif scale_m:
        unit = scale_m.group(1).lower().replace("million", "m").replace("billion", "bn")
    else:
        unit = None
    return currency, unit

# REPLACE
    if pct_m:
        unit = pct_m.group(1)
    elif scale_m:
        unit = scale_m.group(1).lower().replace("million", "m").replace("billion", "bn")
        if unit == "b":
            unit = "bn"   # standalone B = billions
    else:
        unit = None
    return currency, unit

# FIX 3 — rewrite _unit_from_label (handles digits like $5B, £125bn). Find the WHOLE current function:
def _unit_from_label(text):
    """Extract a unit from a HEADER/TITLE label and return (unit, cleaned_text).
    e.g. 'Revenue ($m)' -> ('$m', 'Revenue'); 'Report in £m' -> ('£m', 'Report').
    If no unit, returns (None, text unchanged). The unit is stripped from the
    text so the field_name / canonical_metric stays clean."""
    if not text:
        return None, text
    s = str(text)
    # find a parenthesised or 'in <unit>' unit token
    m = re.search(r"\((?:\s*in\s+)?\s*([£$€]\s*(?:mm|bn|billion|million|m|k)?|%|bps)\s*\)"
                  r"|\bin\s+([£$€]\s*(?:mm|bn|billion|million|m|k)?|%|bps)\b",
                  s, re.I)
    if not m:
        # also handle a trailing bare unit like 'Revenue £m'
        m2 = re.search(r"([£$€]\s*(?:mm|bn|billion|million|m|k)|%|bps)\s*$", s, re.I)
        if not m2:
            return None, text
        raw = m2.group(1)
        cleaned = _clean(s[:m2.start()])
        return _norm_label_unit(raw), (cleaned or s)
    raw = m.group(1) or m.group(2)
    cleaned = _clean((s[:m.start()] + " " + s[m.end():]))
    return _norm_label_unit(raw), (cleaned or s)

# REPLACE
def _unit_from_label(text):
    """Extract a unit token from a HEADER/TITLE label and return
    (unit_token, cleaned_text). The unit token is the currency+scale as written
    (e.g. '$mm', '£bn', '%'), stripped from the text so the field_name stays
    clean. Handles digits between the symbol and scale ('$5B', '£125bn', '($MM)')
    by locating the currency/%/bps/scale span, not an adjacency pattern.
    Returns (None, text) if no unit."""
    if not text:
        return None, text
    s = str(text)
    # locate a unit span: an optional currency, optional digits/sep, a scale OR
    # a bare %/bps. Searched anywhere (inside parens, 'in X', or trailing).
    m = re.search(
        r"[£$€]\s*[\d.,]*\s*(?:mm|bn|billion|million|m|k|b)\b"   # $5B, £125bn, $mm
        r"|[£$€]\s*(?:mm|bn|billion|million|m|k|b)\b"            # $mm (no digits)
        r"|[£$€](?=\s*[\)\s]|$)"                                # bare currency
        r"|\b\d*\s*(?:bps)\b|%",                                # 100 bps, %
        s, re.I)
    if not m:
        return None, text
    span = m.group(0)
    # derive the clean currency+scale token via the robust splitter
    cur, unit = _split_currency_unit(span)
    token = ((cur or "") + (unit or "")) or None
    if not token:
        return None, text
    # remove the matched span AND any now-empty parentheses around it
    cleaned = (s[:m.start()] + " " + s[m.end():])
    cleaned = re.sub(r"\(\s*\)", " ", cleaned)      # drop empty ()
    cleaned = _clean(cleaned)
    return token, (cleaned or s)

#################### MISSED CROPS #####################
# FILE: hybrid_extraction.py — save a miss when the crop fails. Find:
                img_b64 = crop_figure_image(el)
                if not img_b64:
                    claims.extend(desc_claims)

# REPLACE
                img_b64 = crop_figure_image(el)
                if not img_b64:
                    # crop failed -> no image to read. Record a metadata-only
                    # miss so the figure is visible for human review (the figure
                    # existed but produced no chart claims).
                    if missed_root:
                        try:
                            from .missed_crops import save_missed_meta
                            save_missed_meta(missed_root, filename,
                                             getattr(el, "page", None),
                                             getattr(el, "figure_index", figure_index),
                                             reason="crop_failed")
                        except Exception:
                            pass
                    claims.extend(desc_claims)

# FILE: missed_crops.py — add the save_missed_meta function. Find:
def iter_missed_crops(missed_root: str):
    
# REPLACE
def save_missed_meta(missed_root, document, page, figure_index,
                     reason="crop_failed"):
    """Record a metadata-only miss (no image available, e.g. the crop failed).
    Writes a small .json marker into the missed folder so the figure is visible
    for human review even without a crop image. Best-effort; never raises."""
    try:
        folder = f"{missed_root.rstrip('/')}/{_safe(document)}"
        os.makedirs(_local(folder), exist_ok=True)
        pg = page if page is not None else "x"
        name = f"page_{pg}_fig_{figure_index}.missed.json"
        meta = {"document": document, "page": page, "figure_index": figure_index,
                "reason": reason, "image": None}
        with open(_local(f"{folder}/{name}"), "w") as f:
            json.dump(meta, f)
    except Exception:
        pass


def iter_missed_crops(missed_root: str):


############################### IMAGE URL ###############################
# FILE 1: storage.py — three changes.

# 1a — add page_url_map to _stamp. Find:
def _stamp(rows: list[dict], team=None, report_type=None,
          report_date=None, entity_id=None, entity_name=None) -> list[dict]:

# REPLACE
def _stamp(rows: list[dict], team=None, report_type=None,
          report_date=None, entity_id=None, entity_name=None,
          page_url_map=None) -> list[dict]:
    

# 1b — stamp page_url in the loop. Find:

        r["extraction_date"] = now

# REPLACE
        r["extraction_date"] = now
        # page_url: the page image for this row's page, from the per-document map
        # -- SAME url for every metric on that page (consistent reference/lookup).
        if page_url_map:
            _pg = r.get("page")
            if _pg is not None and _pg in page_url_map:
                r["page_url"] = page_url_map[_pg]

# 1c — add page_url_map to the two write methods + pass it to _stamp.

# Find write_gold_metrics:
    def write_gold_metrics(self, claims: list[Claim],
                           team: Optional[str] = None,
                           report_type: Optional[str] = None,
                           report_date: Optional[str] = None,
                           entity_id: Optional[str] = None,
                           entity_name: Optional[str] = None) -> int:

# REPLACE
    def write_gold_metrics(self, claims: list[Claim],
                           team: Optional[str] = None,
                           report_type: Optional[str] = None,
                           report_date: Optional[str] = None,
                           entity_id: Optional[str] = None,
                           entity_name: Optional[str] = None,
                           page_url_map: Optional[dict] = None) -> int:

# And in that method find:
        rows = _stamp(rows, team=team, report_type=report_type,
                      report_date=report_date, entity_id=entity_id,
                      entity_name=entity_name)
        return self._write_layer("gold", rows, team, report_type)

# REPLACE
        rows = _stamp(rows, team=team, report_type=report_type,
                      report_date=report_date, entity_id=entity_id,
                      entity_name=entity_name, page_url_map=page_url_map)
        return self._write_layer("gold", rows, team, report_type)
    
# Find write_silver:
    def write_silver(self, claims: list[Claim], team: Optional[str] = None,
                     report_type: Optional[str] = None,
                     report_date: Optional[str] = None,
                     entity_id: Optional[str] = None,
                     entity_name: Optional[str] = None) -> int:
        rows = [c.model_dump(mode="json") for c in claims]
        rows = _stamp(rows, team=team, report_type=report_type,
                      report_date=report_date, entity_id=entity_id,
                      entity_name=entity_name)
        return self._write_layer("silver", rows, team, report_type)
    
# REPLACE
    def write_silver(self, claims: list[Claim], team: Optional[str] = None,
                     report_type: Optional[str] = None,
                     report_date: Optional[str] = None,
                     entity_id: Optional[str] = None,
                     entity_name: Optional[str] = None,
                     page_url_map: Optional[dict] = None) -> int:
        rows = [c.model_dump(mode="json") for c in claims]
        rows = _stamp(rows, team=team, report_type=report_type,
                      report_date=report_date, entity_id=entity_id,
                      entity_name=entity_name, page_url_map=page_url_map)
        return self._write_layer("silver", rows, team, report_type)

# FILE 2: ai_parse_pipeline.py — two changes.
# 2a — add the _build_page_url_map helper (before run_ai_parse_batch). Find:
def run_ai_parse_batch(spark, submissions, storage, *, audit=None,
                       
# Add right before it:
def _build_page_url_map(parsed, sharpen_dir=None):
    """Build {page(1-based): page_image_url} so every metric from a page can be
    stamped with the SAME page_url. Prefers ai_parse's pages[].image_uri; for the
    sharpened path uses the known sharpened page images. Returns {} if none."""
    import json as _json, os as _os
    out = {}
    # 1. sharpened path: deterministic page_<NNN>.png in the sharpen dir
    if sharpen_dir:
        try:
            from .pdf_sharpen import iter_sharpened_pages
            for _sd, _page, _img in iter_sharpened_pages(sharpen_dir):
                out[_page] = _img            # already a Volume path, 1-based page
            if out:
                return out
        except Exception:
            pass
    # 2. normal path: ai_parse pages[].image_uri, keyed by page_id (+1 -> 1-based)
    try:
        data = parsed
        if type(parsed).__name__ == "VariantVal" or hasattr(parsed, "toJson"):
            data = _json.loads(parsed.toJson())
        elif isinstance(parsed, str):
            data = _json.loads(parsed)
        elif hasattr(parsed, "asDict"):
            data = parsed.asDict(recursive=True)
        doc = (data or {}).get("document", data) if isinstance(data, dict) else {}
        pages = (data.get("pages") if isinstance(data, dict) else None) or \
                (doc.get("pages") if isinstance(doc, dict) else None) or []
        for pg in pages:
            if isinstance(pg, dict):
                pid = pg.get("id") if pg.get("id") is not None else pg.get("page_id")
                uri = pg.get("image_uri")
                if pid is not None and uri:
                    u = uri.replace("dbfs:/Volumes", "/Volumes").replace("dbfs:", "/dbfs") \
                        if isinstance(uri, str) else uri
                    out[int(pid) + 1] = u     # ai_parse page_id is 0-based -> +1
    except Exception:
        pass
    return out


def run_ai_parse_batch(spark, submissions, storage, *, audit=None,

# 2b — build the map and pass it to the writes. Find:
            entity_id = sub.get("entity_id") or ent["entity_id"]
            entity_name = sub.get("entity_name") or ent["entity_name"]

            storage.write_silver(claims, team=sub["team"],
                                 report_type=sub["report_type"],
                                 report_date=rdate, entity_id=entity_id,
                                 entity_name=entity_name)
            result.gold_total += storage.write_gold_metrics(
                clean, team=sub["team"], report_type=sub["report_type"],
                report_date=rdate, entity_id=entity_id, entity_name=entity_name)

# REPLACE
            entity_id = sub.get("entity_id") or ent["entity_id"]
            entity_name = sub.get("entity_name") or ent["entity_name"]

            # page_url: one image url per page (SAME for every metric on a page)
            _purl_map = _build_page_url_map(
                parsed, sharpen_dir=(_sharp_dir if ("_sharp_dir" in dir()
                                                    and _sharpen_on) else None))

            storage.write_silver(claims, team=sub["team"],
                                 report_type=sub["report_type"],
                                 report_date=rdate, entity_id=entity_id,
                                 entity_name=entity_name, page_url_map=_purl_map)
            result.gold_total += storage.write_gold_metrics(
                clean, team=sub["team"], report_type=sub["report_type"],
                report_date=rdate, entity_id=entity_id, entity_name=entity_name,
                page_url_map=_purl_map)
    
