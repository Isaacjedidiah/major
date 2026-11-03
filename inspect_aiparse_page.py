# =============================================================================
#  STANDALONE ai_parse page inspector — runs ai_parse and dumps the COMPLETE raw
#  output (every field, every element, every key) exactly as ai_parse presents
#  it, so you can see the full structure including however the legend/colour is
#  represented.
#
#  Supply DOC_PATH + IMG_OUT (like the shared-legend test). spark must exist.
# =============================================================================
import json
from preprocessing_etl.custom.ai_parse_pipeline import _run_ai_parse

# ---- SET THESE ----
DOC_PATH = "/Volumes/<cat>/<sch>/<vol>/docs/the_doc_with_legend.pdf"
IMG_OUT  = "/Volumes/<cat>/<sch>/<vol>/inspect_out"
# -------------------

parsed = _run_ai_parse(spark, DOC_PATH, IMG_OUT)

def _to_dict(resp):
    if type(resp).__name__ == "VariantVal" or hasattr(resp, "toJson"):
        return json.loads(resp.toJson())
    if isinstance(resp, str):
        return json.loads(resp)
    if hasattr(resp, "asDict"):
        return resp.asDict(recursive=True)
    return resp

data = _to_dict(parsed)

# 1. the ENTIRE raw response, pretty-printed (everything, nothing filtered)
print("=" * 70)
print("FULL RAW ai_parse RESPONSE (complete structure)")
print("=" * 70)
print(json.dumps(data, indent=2, default=str))

# 2. top-level keys, so you see the overall shape
print("\n" + "=" * 70)
print("TOP-LEVEL KEYS:", list(data.keys()) if isinstance(data, dict) else type(data))
doc = data.get("document", data) if isinstance(data, dict) else {}
if isinstance(doc, dict):
    print("DOCUMENT KEYS:", list(doc.keys()))
    els = doc.get("elements") or []
    print(f"ELEMENTS: {len(els)}")
    if els:
        print("KEYS PRESENT ON ELEMENT[0]:", list(els[0].keys()))
        # show the union of all keys seen across elements (some may vary)
        allkeys = set()
        for e in els:
            if isinstance(e, dict):
                allkeys.update(e.keys())
        print("ALL KEYS SEEN ACROSS ELEMENTS:", sorted(allkeys))

print("\n>>> This prints the COMPLETE raw output. Share the parts showing:")
print(">>>  - how the legend / colour is represented (any element describing it)")
print(">>>  - the full structure of a figure/chart element (all its keys)")
print(">>>  - whether elements carry coordinates and in what form")
