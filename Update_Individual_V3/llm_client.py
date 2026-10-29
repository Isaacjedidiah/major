"""Unified async LLM client over Databricks serving endpoints.

Databricks exposes Foundation Model APIs and External Models through a
single OpenAI-compatible surface, so one client handles every model tier
by switching the ``model`` (endpoint) string. No provider-specific classes.

The client speaks the OpenAI Chat Completions shape, including tool-calling
with forced ``tool_choice`` so extraction output is always structured JSON.

**Async:** ``complete`` and ``extract`` are coroutines using the async OpenAI
client. Concurrency is bounded by a semaphore (so many element calls run at
once without exceeding the endpoint's rate limit), and rate-limit / transient
errors are retried with exponential backoff. This is what lets the per-element
pipeline make its many small calls in overlapping waves rather than one-by-one.
"""
from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from typing import Optional

from .config import CONFIG

# Optional MLflow tracing of every model call (prompt/response/latency/tokens).
# Enabled only when MLflow is importable AND config asks for it — so the pipeline
# runs unchanged where MLflow isn't present. Databricks has MLflow built in.
def _mlflow_trace(name):
    def _decorator(fn):
        if not getattr(getattr(CONFIG, "observability", None), "mlflow_tracing", False):
            return fn
        try:
            import mlflow
            return mlflow.trace(name=name)(fn)
        except Exception:
            return fn
    return _decorator

# Bound concurrent in-flight model calls. Tune to the endpoint's rate limit;
# too high just trades waiting for throttling + retries. Overridable via config.
_MAX_CONCURRENCY = getattr(CONFIG, "max_model_concurrency", 8)
_RETRY_ATTEMPTS = 8
_RETRY_BASE_DELAY = 2.0   # seconds; doubles each attempt with jitter

# Substrings of endpoint names whose models reject temperature/top_p/top_k
# (e.g. Claude Sonnet 5 returns 400 if any are sent). Matched by substring so
# it covers versioned endpoint names like databricks-claude-sonnet-5.
_NO_SAMPLING_PARAM_MODELS = ("claude-sonnet-5",)


@dataclass
class LLMResponse:
    text: str
    tool_arguments: Optional[dict] = None
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient:
    """Thin wrapper around the Databricks OpenAI-compatible endpoint.

    Locally (no Databricks host) it raises on use unless a base URL and key
    are supplied, keeping with the no-silent-fallback principle: tests use
    the stub client in ``tests/`` rather than a hidden fake here.
    """

    def __init__(self, base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 max_concurrency: int = _MAX_CONCURRENCY):
        self._base_url = base_url or _default_base_url()
        self._api_key = api_key or CONFIG.databricks_token or CONFIG.anthropic_api_key
        self._client = None  # lazily created AsyncOpenAI client
        # One semaphore per client instance caps how many calls are in flight
        # at once across all concurrent element extractions.
        self._sem = asyncio.Semaphore(max_concurrency)

    def _ensure_client(self):
        if self._client is None:
            # openai>=1.0 async client, base_url pointing at Databricks.
            from openai import AsyncOpenAI  # imported lazily; optional at import

            if not self._base_url or not self._api_key:
                raise RuntimeError(
                    "LLMClient needs a base_url and api_key. Configure the "
                    "Databricks serving endpoint host and a secret; refusing "
                    "to fall back to a stub."
                )
            self._client = AsyncOpenAI(base_url=self._base_url,
                                       api_key=self._api_key,
                                       timeout=120.0,    # fail fast, then our retry
                                       max_retries=0)    # we handle retries below
        return self._client

    async def _create(self, **kwargs):
        """Make one chat-completion call under the concurrency cap, retrying
        rate-limit / transient errors with exponential backoff + jitter."""
        # Some Databricks foundation models reject sampling params: Claude
        # Sonnet 5 returns 400 if temperature / top_p / top_k are present. Strip
        # them for those endpoints rather than let the call fail.
        model = kwargs.get("model", "")
        if any(tok in model for tok in _NO_SAMPLING_PARAM_MODELS):
            for pname in ("temperature", "top_p", "top_k"):
                kwargs.pop(pname, None)
        client = self._ensure_client()
        last_exc = None
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                async with self._sem:
                    return await client.chat.completions.create(**kwargs)
            except Exception as exc:  # narrow at integration time to RateLimitError etc.
                last_exc = exc
                if not _is_retryable(exc) or attempt == _RETRY_ATTEMPTS - 1:
                    raise
                delay = _RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5)
                await asyncio.sleep(delay)
        raise last_exc  # pragma: no cover

    async def complete(self, model_key: str, prompt: str,
                       content: str) -> LLMResponse:
        """Plain completion, used by the query router's classification call."""
        spec = CONFIG.model(model_key)
        resp = await self._create(
            model=spec.endpoint,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": content},
            ],
            temperature=0,
            max_tokens=8192,
        )
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "completion_tokens", 0),
        )

    async def extract(self, model_key: str, prompt: str, content: str,
                      tool_schema: dict,
                      image_base64: Optional[str] = None) -> LLMResponse:
        """Forced tool-use so the model returns structured JSON, not prose.

        When ``image_base64`` is provided (a cropped-and-zoomed figure region
        from ``search.figure_preprocessor``), it is sent as a multimodal image
        block alongside the text, so the model reads the chart from a legible
        crop rather than a full page. Text-only extraction is unchanged when
        it is None.

        Not every endpoint honours forced ``tool_choice`` identically (the
        FMAPI OpenAI-compatible surface for Claude/Llama has historically
        differed from External Models). We therefore parse defensively: if
        no tool call comes back, ``tool_arguments`` is ``None`` and the
        caller escalates or quarantines rather than crashing.
        """
        spec = CONFIG.model(model_key)
        tool_name = tool_schema["name"]

        # A figure/chart element has an image but often EMPTY text content (the
        # data lives in the image). An empty text block — or an empty text-only
        # message — is rejected by the API ("messages must have non-empty
        # content"). So fall back to a minimal instruction when content is blank.
        text_part = content if (content and content.strip()) else (
            "Extract all metrics visible in this image." if image_base64
            else "Extract all metrics from this element.")

        if image_base64 is not None:
            user_content = [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                {"type": "text", "text": text_part},
            ]
        else:
            user_content = text_part

        resp = await self._create(
            model=spec.endpoint,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
            tools=[{"type": "function", "function": tool_schema}],
            tool_choice={"type": "function", "function": {"name": tool_name}},
            temperature=0,
            max_tokens=8192,
        )
        msg = resp.choices[0].message
        args = _safe_tool_args(msg)
        usage = resp.usage
        return LLMResponse(
            text=msg.content or "",
            tool_arguments=args,
            input_tokens=getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "completion_tokens", 0),
        )


def _is_retryable(exc: Exception) -> bool:
    """Whether an error is worth retrying with backoff. Rate-limit and
    transient server/network errors are; a bad request is not. Matched by
    class name so we don't hard-import openai's exception types at module load
    (kept optional). Narrow this at integration time to the real classes.
    """
    name = type(exc).__name__
    retryable_names = {"RateLimitError", "APITimeoutError", "APIConnectionError",
                       "InternalServerError", "APIError", "APIStatusError",
                       "Timeout", "TimeoutError", "ConnectionError",
                       "ReadTimeout", "ServiceUnavailableError"}
    if name in retryable_names:
        return True
    # match by HTTP status code (429 rate-limit, 5xx server/transient)
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int) and (status == 429 or 500 <= status < 600):
        return True
    # match by message for transient signals the class name may not capture
    msg = str(exc).lower()
    if any(s in msg for s in ("rate limit", "timeout", "timed out",
                              "temporarily", "503", "502", "500", "429",
                              "overloaded", "connection", "unavailable")):
        return True
    return False


def _safe_tool_args(msg) -> Optional[dict]:
    """Extract and JSON-parse tool-call arguments, tolerating malformed output.

    Returns ``None`` if there is no usable JSON — never raises. This keeps a
    single bad model response from killing a whole element's extraction.

    Two sources: a proper tool call (preferred), or — for models that return
    the JSON as plain text content instead of a tool call (e.g. reasoning
    models like gpt-oss) — the message content parsed as JSON. Without the
    content fallback, every such response yields no claims and quarantines.
    """
    import re

    # 1. preferred: a real tool call with JSON arguments
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        raw = tool_calls[0].function.arguments
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass

    # 2. fallback: the model returned JSON as text content. Reasoning models
    #    sometimes wrap it in prose, so locate the first {...claims...} object
    #    rather than assuming the whole string is JSON.
    content = getattr(msg, "content", None)
    if isinstance(content, str) and content.strip():
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        m = re.search(r"\{.*\"claims\".*\}", content, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
    return None


def _default_base_url() -> Optional[str]:
    import os

    host = os.environ.get("DATABRICKS_HOST")
    if host:
        return f"{host.rstrip('/')}/serving-endpoints"
    return None


# Aligned with the ``Claim`` contract: every optional structural field the
# reconciler / normaliser / Gold layer relies on is requestable here, so the
# model can actually return scale, netting and as_at_date (previously inert).
EXTRACTION_TOOL_SCHEMA: dict = {
    "name": "record_claims",
    "description": "Record structured regulatory claims found in the content.",
    "parameters": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field_name": {
                            "type": "string",
                            "description": "Metric name exactly as written.",
                        },
                        "value": {
                            "type": "string",
                            "description": "Value exactly as written, incl. sign.",
                        },
                        "unit": {
                            "type": "string",
                            "description": "e.g. '%', 'bps', 'GBP'. Empty if none.",
                        },
                        "reporting_basis": {
                            "type": "string",
                            "description": "e.g. 'consolidated', 'solo'. Empty if unstated.",
                        },
                        "netting": {
                            "type": "string",
                            "description": "e.g. 'net of deductions', 'gross'. Empty if unstated.",
                        },
                        "scale": {
                            "type": "string",
                            "description": "e.g. 'millions', 'thousands', 'ratio'. Empty if unstated.",
                        },
                        "as_at_date": {
                            "type": "string",
                            "description": "Reporting/reference date if stated (ISO-8601 preferred).",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "0-1 confidence this claim is correct.",
                        },
                    },
                    "required": ["field_name", "value", "confidence"],
                },
            }
        },
        "required": ["claims"],
    },
}
