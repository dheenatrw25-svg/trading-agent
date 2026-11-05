from __future__ import annotations

import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .paths import PROJECT_ROOT


GEMINI_MODEL = "gemini-3.0-flash"
GEMINI_REQUEST_TIMEOUT_MS = 120_000
GEMINI_MAX_ATTEMPTS = 3
GEMINI_RETRY_BACKOFF_SECONDS = 5
RETRYABLE_STATUS_CODES = {502, 503, 504}
QUOTA_ERROR_MARKERS = (
    "QUOTA",
    "DAILY LIMIT",
    "PER DAY",
    "BILLING",
    "RESOURCE_EXHAUSTED",
)


def create_client() -> genai.Client:
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment."
        )
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=GEMINI_REQUEST_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def _error_fields(exc: Exception) -> tuple[str | None, str | None, str]:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    status = getattr(exc, "status", None)
    response = getattr(exc, "response", None)
    if code is None:
        code = getattr(response, "status_code", None)
    if status is None:
        status = getattr(response, "status", None)

    message = getattr(exc, "message", None) or str(exc) or type(exc).__name__
    for key_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        api_key = os.environ.get(key_name)
        if api_key:
            message = message.replace(api_key, "[REDACTED]")

    return (
        str(code) if code is not None else None,
        str(status) if status is not None else None,
        message,
    )


def _is_quota_exhausted(exc: Exception) -> bool:
    code, status, message = _error_fields(exc)
    error_text = " ".join(part for part in (status, message) if part).upper()
    has_rate_limit_status = code == "429" or re.search(r"\b429\b", error_text)
    return bool(has_rate_limit_status) and any(
        marker in error_text for marker in QUOTA_ERROR_MARKERS
    )


def _is_retryable(exc: Exception) -> bool:
    if _is_quota_exhausted(exc):
        return False

    code, _, message = _error_fields(exc)
    if code in {str(value) for value in RETRYABLE_STATUS_CODES}:
        return True

    error_text = message.upper()
    if code == "429" and not any(
        marker in error_text for marker in QUOTA_ERROR_MARKERS
    ):
        return True
    if re.search(r"\b(429|502|503|504)\b", error_text):
        return True

    current: BaseException | None = exc
    while current is not None:
        names = {base.__name__.casefold() for base in type(current).__mro__}
        if isinstance(current, (TimeoutError, ConnectionError)) or any(
            marker in name
            for name in names
            for marker in (
                "timeout",
                "connecterror",
                "networkerror",
                "serverdisconnected",
                "remoteprotocolerror",
            )
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _describe_error(exc: Exception) -> str:
    code, status, message = _error_fields(exc)
    details = [f"type={type(exc).__name__}"]
    if code is not None:
        details.append(f"code={code}")
    if status is not None:
        details.append(f"status={status}")
    details.append(f"message={message}")
    return ", ".join(details)


def generate_content_with_retry(client: genai.Client, **request: Any) -> Any:
    for attempt in range(1, GEMINI_MAX_ATTEMPTS + 1):
        try:
            return client.models.generate_content(**request)
        except Exception as exc:
            error = _describe_error(exc)
            if not _is_retryable(exc) or attempt == GEMINI_MAX_ATTEMPTS:
                print(
                    f"[API] Gemini request failed on attempt "
                    f"{attempt}/{GEMINI_MAX_ATTEMPTS}: {error}"
                )
                raise RuntimeError(f"Gemini API request failed: {error}") from exc

            delay = GEMINI_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
            print(
                f"[API] Gemini request attempt {attempt}/{GEMINI_MAX_ATTEMPTS} "
                f"failed: {error}; retrying in {delay} seconds."
            )
            time.sleep(delay)

    raise RuntimeError("Gemini API request ended without a response.")


def response_text(response: Any, operation: str) -> str:
    if response is None:
        raise ValueError(f"Gemini {operation} returned no response.")

    prompt_feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(prompt_feedback, "block_reason", None)
    block_text = (
        str(block_reason).upper()
        if isinstance(block_reason, str)
        else ""
    )
    if any(
        marker in block_text
        for marker in (
            "SAFETY",
            "BLOCKED",
            "PROHIBITED",
            "RECITATION",
            "OTHER",
        )
    ):
        raise ValueError(
            f"Gemini {operation} response was blocked "
            f"(block_reason={block_reason})."
        )

    candidates = getattr(response, "candidates", None) or []
    if not isinstance(candidates, (list, tuple)):
        candidates = []
    for candidate in candidates:
        finish_reason = getattr(candidate, "finish_reason", None)
        finish_text = (
            str(finish_reason).upper()
            if isinstance(finish_reason, str)
            else ""
        )
        if any(
            marker in finish_text
            for marker in ("SAFETY", "BLOCKED", "PROHIBITED", "RECITATION")
        ):
            raise ValueError(
                f"Gemini {operation} response was blocked "
                f"(finish_reason={finish_reason})."
            )

    try:
        text = response.text
    except (AttributeError, ValueError) as exc:
        raise ValueError(
            f"Gemini {operation} response did not contain usable text."
        ) from exc

    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Gemini {operation} response was empty.")
    return text.strip()
