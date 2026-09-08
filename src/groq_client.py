"""
Groq client wrapper — TRD.md §5.1, plus rate-limit/transient-error backoff pulled forward from
Prompt 11 (this was going to be built there, but the very first real eval run hit Groq's 8000
TPM on-demand rate limit almost immediately, so a case can't even complete once without this).

Deviation from spec: TRD.md §5.1 / PRD.md §8 pin model="llama-3.3-70b-versatile", but that model
is no longer available on this account's Groq key at all (confirmed via scripts/list_groq_models.py
-> 404 model_not_found, and no Llama chat model appears in the 14 models actually returned).
Standardized on "openai/gpt-oss-120b" instead (user-confirmed choice) as the closest available
match in capability/size for this pipeline's structured-JSON extraction/editing/validation calls.
If this key's model catalog changes again, rerun scripts/list_groq_models.py rather than guessing.

The TRD's sample code also constructs the Groq client at module import time
(`client = Groq(api_key=os.environ["GROQ_API_KEY"])`). Doing that here would mean merely
importing this module without GROQ_API_KEY set crashes immediately with a raw KeyError, at
IMPORT time rather than call time — which would break any other module that imports this one
(including test collection) even if it never actually calls the Groq API. This implementation
lazily constructs the client on first use inside call_groq() instead, and raises a clear
RuntimeError (not a KeyError) if GROQ_API_KEY is missing at that point. Everything else matches
TRD.md §5.1: temperature=0.0 default, JSON-mode + pydantic schema-validate-and-retry loop.
"""

from __future__ import annotations

import os
import re
import time
from typing import TypeVar

from groq import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    Groq,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel

MODEL = "openai/gpt-oss-120b"

_client: Groq | None = None

_TRANSIENT_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
_MAX_TRANSIENT_RETRIES = 3
_BASE_BACKOFF_SECONDS = 2.0  # 2s, 4s, 8s — a plain 1/2/4s backoff (Prompt 11's own example)
                             # isn't enough: an observed 429 asked for an 11.9s wait (8000 TPM
                             # on-demand tier), so the base is doubled and the actual
                             # Retry-After/suggested-wait value is preferred whenever present.
_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)s", flags=re.IGNORECASE)

_usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set. Set it before making any Groq "
                "API call, e.g.:\n"
                "  export GROQ_API_KEY=your-key-here    (bash)\n"
                "  $env:GROQ_API_KEY = 'your-key-here'  (PowerShell)"
            )
        _client = Groq(api_key=api_key)
    return _client


def _suggested_wait_seconds(error: Exception, attempt: int) -> float:
    response = getattr(error, "response", None)
    if response is not None:
        header_value = response.headers.get("retry-after")
        if header_value:
            try:
                return float(header_value)
            except ValueError:
                pass
    match = _RETRY_AFTER_RE.search(str(error))
    if match:
        return float(match.group(1)) + 0.5  # small safety margin
    return _BASE_BACKOFF_SECONDS * (2 ** attempt)


def _is_retryable_bad_request(error: BadRequestError) -> bool:
    """Groq's JSON-mode enforcement occasionally 400s with code="json_validate_failed" and an
    EMPTY failed_generation — i.e. the model produced no usable output on that one attempt. This
    is a stochastic generation hiccup (retrying the identical request typically succeeds), not a
    genuinely malformed request, so it's retried like a transient error. Any other BadRequestError
    (bad params, invalid model, etc.) would fail identically forever and is NOT retried."""
    body = getattr(error, "body", None) or {}
    code = (body.get("error") or {}).get("code") if isinstance(body, dict) else None
    return code == "json_validate_failed"


def _create_completion_with_backoff(client: Groq, **kwargs):
    last_error: Exception | None = None
    for attempt in range(_MAX_TRANSIENT_RETRIES + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except _TRANSIENT_ERRORS as e:
            last_error = e
            if attempt == _MAX_TRANSIENT_RETRIES:
                break
            wait = _suggested_wait_seconds(e, attempt)
            time.sleep(wait)
        except BadRequestError as e:
            if not _is_retryable_bad_request(e) or attempt == _MAX_TRANSIENT_RETRIES:
                raise
            last_error = e
            time.sleep(_BASE_BACKOFF_SECONDS * (2 ** attempt))
    raise RuntimeError(
        f"Groq call failed after {_MAX_TRANSIENT_RETRIES} retries due to a transient/rate-limit "
        f"error: {last_error}"
    )


T = TypeVar("T", bound=BaseModel)


def call_groq(
    system: str,
    user: str,
    response_model: type[T],
    temperature: float = 0.0,
    max_retries: int = 2,
) -> T:
    """Call Groq's chat completion API in JSON mode and validate the response against
    `response_model`, retrying (with the validation error appended to the prompt) up to
    `max_retries` times before raising. Transient/rate-limit errors on the API call itself are
    retried separately, with backoff, and don't consume a schema-validation retry.

    Deviation from TRD.md §5.1's sample code, added after two real failures on the first live
    eval run:
      - response_format uses `{"type": "json_schema", ...}` (Structured Outputs, grammar-
        constrained) instead of the older `{"type": "json_object"}` mode the TRD sample uses.
        Groq's own docs describe json_object as the "older" mode and say json_schema "is
        preferred for models that support it" — and in practice, json_object mode produced a
        `BadRequestError: json_validate_failed` with an EMPTY failed_generation on every single
        retry attempt for one real extraction call (not a one-off fluke), pointing at the mode
        itself rather than bad luck. `strict` is left False: pydantic's model_json_schema()
        output (Optional fields as `anyOf [X, null]`, `$defs` refs) isn't guaranteed to fall
        inside the restricted JSON-Schema subset Groq's strict mode supports.
      - reasoning_effort="low": openai/gpt-oss-120b defaults to "medium" reasoning effort, which
        can consume its entire token budget on hidden reasoning before emitting any of the final
        JSON answer — the likely root cause of the empty failed_generation above. This is a
        deterministic extraction/editing/validation task, not one needing heavy reasoning.
      - max_completion_tokens=4096: a generous but bounded ceiling so a single call can't runs
        away and blow through the account's 8000 TPM rate limit on its own.
    """
    client = _get_client()
    schema = response_model.model_json_schema()
    last_error: Exception | None = None

    for _attempt in range(max_retries + 1):
        resp = _create_completion_with_backoff(
            client,
            model=MODEL,
            temperature=temperature,
            max_completion_tokens=4096,
            reasoning_effort="low",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": schema,
                    "strict": False,
                },
            },
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

        _usage["calls"] += 1
        usage = getattr(resp, "usage", None)
        if usage is not None:
            _usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            _usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
            _usage["total_tokens"] += getattr(usage, "total_tokens", 0) or 0

        raw = resp.choices[0].message.content
        try:
            return response_model.model_validate_json(raw)
        except Exception as e:  # broad on purpose: any parse/validation failure triggers a retry
            last_error = e
            user += f"\n\nYour previous response was invalid: {e}\nReturn ONLY valid JSON matching the schema."

    raise RuntimeError(f"Groq call failed schema validation after {max_retries} retries: {last_error}")


def print_usage_summary() -> None:
    print(
        f"Groq usage: {_usage['calls']} calls, "
        f"{_usage['prompt_tokens']} prompt tokens, "
        f"{_usage['completion_tokens']} completion tokens, "
        f"{_usage['total_tokens']} total tokens"
    )
