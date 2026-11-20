import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

THRESHOLD = 0.55
FILTER_MARKERS = ("content_filter", "content filter", "safety filter", "blocked by policy", "policy_violation")
REFUSAL_PATTERNS = (
    re.compile(r"\b(i (?:can't|cannot|won't|will not) (?:help|assist|provide|comply))\b", re.I),
    re.compile(r"\b(i am unable to|i'm unable to|i cannot assist)\b", re.I),
    re.compile(r"\b(that request (?:is|would be) not something i can)\b", re.I),
)
PREFIX_REFUSAL = re.compile(r"^\s*(?:sorry[,! ]|i (?:can't|cannot|won't|am unable)|as an ai\b)", re.I)
HEDGING = re.compile(r"\b(i may be wrong|i'm not certain|i cannot verify|this may be incomplete)\b", re.I)


@dataclass(frozen=True)
class Verdict:
    label: str
    score: float
    reason: str


def normalize_prompt(prompt: Any) -> str:
    if isinstance(prompt, str):
        return " ".join(prompt.split()).casefold()
    return json.dumps(prompt, sort_keys=True, ensure_ascii=False, separators=(",", ":")).casefold()


def verdict_cache_key(provider: str, model: str, prompt: Any) -> str:
    normalized = f"{provider.strip().casefold()}\0{model.strip().casefold()}\0{normalize_prompt(prompt)}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _flatten(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_flatten(part) for part in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten(part) for part in value)
    return ""


def _request_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(_flatten(item.get("content", "")) if isinstance(item, dict) else _flatten(item) for item in value)
    return _flatten(value)


def _completion_text(payload: Any) -> str:
    if isinstance(payload, dict) and isinstance(payload.get("choices"), list):
        parts: list[str] = []
        for choice in payload["choices"]:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message") or choice.get("delta") or {}
            if isinstance(message, dict):
                parts.append(_flatten(message.get("content", "")))
                parts.append(_flatten(message.get("refusal", "")))
        parts.append(_flatten(payload.get("stream_sample", "")))
        return " ".join(part for part in parts if part)
    return _flatten(payload)


def detect_verdict(payload: Any, status_code: int = 200, request_prompt: Any = None) -> Verdict:
    if status_code >= 500:
        return Verdict("unavailable", 0.0, "upstream_server_error")

    signals: list[tuple[float, str]] = []
    serialized = _flatten(payload)
    lowered = serialized.casefold()
    choices = payload.get("choices") if isinstance(payload, dict) else None

    if status_code in (400, 403) and any(marker in lowered for marker in FILTER_MARKERS):
        signals.append((1.0, "http_content_filter"))
    if isinstance(choices, list) and not choices:
        signals.append((1.0, "empty_choices"))
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            if choice.get("finish_reason") == "content_filter":
                signals.append((1.0, "finish_reason_content_filter"))
            message = choice.get("message") or choice.get("delta") or {}
            if isinstance(message, dict) and message.get("content") == "" and not message.get("tool_calls"):
                signals.append((1.0, "empty_completion_content"))

    text = _completion_text(payload)
    if PREFIX_REFUSAL.search(text):
        signals.append((0.88, "refusal_prefix"))
    if any(pattern.search(text) for pattern in REFUSAL_PATTERNS):
        signals.append((0.82, "refusal_phrase"))
    if HEDGING.search(text):
        signals.append((0.58, "hedging_language"))

    prompt_text = _request_text(request_prompt)
    request_words = len(prompt_text.split())
    response_words = len(text.split())
    if request_words >= 100 and response_words < 20:
        signals.append((0.56, "very_short_response_to_complex_request"))

    if not signals:
        return Verdict("non_refusal", 0.0, "no_refusal_signal")
    score, reason = max(signals, key=lambda signal: signal[0])
    return Verdict("refusal" if score >= THRESHOLD else "non_refusal", score, reason)
