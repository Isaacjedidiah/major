# =============================================================================
#  INSPECT the legend extraction — see EXACTLY what read_page_legend pulls from a
#  page, so you can confirm it's extracting the RIGHT legend before any changes.
#
#  Run this on a page you KNOW has a shared legend. It prints:
#    - whether the page image was obtained
#    - the raw LLM response
#    - the parsed colour -> series map
#    - the colour-tagged claims on that page and whether each would match
# =============================================================================
import base64, io, json
from PIL import Image
from preprocessing_etl.custom.shared_legend import (
    read_page_legend, _LEGEND_SCHEMA, LEGEND_PROMPT, _find_colour)

# ---- set these ----
PAGE_IMAGE = "/Volumes/<cat>/<sch>/<vol>/.../page_003.png"   # a page with a shared legend
# run_async, client must exist in your session (the ones you pass to the pipeline)
# -------------------

def _b64(path):
    p = path.replace("/Volumes", "/Volumes")  # UC Volume path as-is
    img = Image.open(p)
    buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
    return base64.b64encode(buf.read()).decode()

print("=== 1. page image ===")
try:
    page_b64 = _b64(PAGE_IMAGE)
    print(f"  got image, {len(page_b64)} b64 chars")
except Exception as e:
    print("  FAILED to open page image:", e); raise SystemExit

print("\n=== 2. RAW legend LLM response (what the model actually returns) ===")
resp = run_async(client.extract("tier2", LEGEND_PROMPT, "", _LEGEND_SCHEMA,
                                image_base64=page_b64))
print("  tool_arguments:", getattr(resp, "tool_arguments", None))
print("  text:", (getattr(resp, "text", "") or "")[:500])

print("\n=== 3. parsed colour -> series map ===")
legend = read_page_legend(run_async, client, "tier2", page_b64)
print(f"  legend_size={len(legend)}")
for col, ser in legend.items():
    print(f"    {col}  ->  {ser}")

print("\n>>> CHECK:")
print(">>> - Is the RAW response returning a 'legend' array? (if not -> schema/prompt issue)")
print(">>> - Does the colour->series map match the ACTUAL legend on the page?")
print(">>> - Are the colour WORDS the same ones your claims use (teal vs cyan etc)?")
print(">>> If legend is right but claims don't augment -> colour-word mismatch.")
print(">>> If legend is wrong/empty -> the legend READ is the problem, not augmentation.")
