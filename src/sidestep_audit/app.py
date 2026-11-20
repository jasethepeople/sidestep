import asyncio
import hmac
import json
import math
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx
import structlog
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from sidestep_audit.config import ProviderConfig, Settings, load_settings
from sidestep_audit.detection import Verdict, detect_verdict, verdict_cache_key
from sidestep_audit.ladder import AttemptOutcome, LadderResult, run_ladder
from sidestep_audit.storage import Store

logger = structlog.get_logger()
HOP_BY_HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-length", "content-encoding",
}


class StreamObserver:
    def __init__(self) -> None:
        self.pending = bytearray()
        self.text_parts: list[str] = []
        self.raw_parts: list[str] = []
        self.finish_reason: str | None = None
        self.saw_choices = False
        self.saw_data = False
        self.done = False
        self.has_tool_calls = False

    def feed(self, chunk: bytes) -> None:
        self.pending.extend(chunk)
        while b"\n" in self.pending:
            line, _, remainder = self.pending.partition(b"\n")
            self.pending = bytearray(remainder)
            self._line(line.rstrip(b"\r"))
        if len(self.pending) > 65536:
            self.pending = self.pending[-65536:]

    def _line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        data = line[5:].strip().decode("utf-8", errors="replace")
        self.saw_data = True
        if data == "[DONE]":
            self.done = True
            return
        if len("".join(self.raw_parts)) < 30000:
            self.raw_parts.append(data[: max(0, 30000 - len("".join(self.raw_parts)))])
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            return
        choices = event.get("choices") if isinstance(event, dict) else None
        if not isinstance(choices, list):
            return
        self.saw_choices = True
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or choice.get("message") or {}
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str):
                    self.text_parts.append(content)
                if delta.get("tool_calls"):
                    self.has_tool_calls = True
            if choice.get("finish_reason") is not None:
                self.finish_reason = str(choice["finish_reason"])

    def result(self) -> Any:
        if not self.done and self.finish_reason is None:
            return None
        if not self.saw_choices and self.done:
            return {"choices": [], "stream_sample": " ".join(self.raw_parts)}
        message: dict[str, Any] = {}
        content = "".join(self.text_parts)
        if content or not self.has_tool_calls:
            message["content"] = content
        if self.has_tool_calls:
            message["tool_calls"] = [{"type": "function"}]
        return {
            "choices": [{"message": message, "finish_reason": self.finish_reason}],
            "stream_sample": " ".join(self.raw_parts),
        }


def _response_headers(headers: httpx.Headers) -> dict[str, str]:
    return {key: value for key, value in headers.items() if key.lower() not in HOP_BY_HOP_HEADERS}


@dataclass
class BufferedStream:
    upstream: httpx.Response
    iterator: Any
    initial_frames: list[bytes]
    pending: bytes
    observer: StreamObserver
    initial_text: str
    completed: bool


def _stream_frame_text(frame: bytes) -> str:
    text: list[str] = []
    for line in frame.splitlines():
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip().decode("utf-8", errors="replace")
        if data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = event.get("choices") if isinstance(event, dict) else None
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or choice.get("message") or {}
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                text.append(delta["content"])
    return "".join(text)


def _sse_boundary(data: bytearray) -> tuple[int, int] | None:
    match = re.search(rb"\r?\n\r?\n", data)
    return (match.start(), match.end()) if match else None


async def _buffer_stream(
    upstream: httpx.Response,
    messages: Any,
    limit_tokens: int = 120,
) -> tuple[BufferedStream | None, Verdict]:
    observer = StreamObserver()
    iterator = upstream.aiter_bytes().__aiter__()
    buffered: list[bytes] = []
    pending = bytearray()
    text_parts: list[str] = []
    complete = False
    while len("".join(text_parts).split()) < limit_tokens:
        try:
            chunk = await anext(iterator)
        except StopAsyncIteration:
            complete = True
            if pending:
                frame = bytes(pending)
                buffered.append(frame)
                observer.feed(frame)
                text_parts.append(_stream_frame_text(frame))
                pending.clear()
            break
        pending.extend(chunk)
        while True:
            boundary = _sse_boundary(pending)
            if boundary is None:
                break
            start, end = boundary
            frame = bytes(pending[:end])
            del pending[:end]
            buffered.append(frame)
            observer.feed(frame)
            text_parts.append(_stream_frame_text(frame))
            if len("".join(text_parts).split()) >= limit_tokens:
                break
    initial_text = "".join(text_parts)
    verdict = detect_verdict(
        {"choices": [{"message": {"content": initial_text}}]},
        upstream.status_code,
        messages,
    )
    if upstream.status_code >= 500:
        verdict = detect_verdict({}, upstream.status_code, messages)
    if verdict.label == "refusal" or verdict.label == "unavailable":
        await upstream.aclose()
        return None, verdict
    return BufferedStream(upstream, iterator, buffered, bytes(pending), observer, initial_text, complete), verdict


def _token_estimate(value: Any) -> int:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return max(1, math.ceil(len(encoded) / 4))


def _response_cost(payload: Any, request_payload: dict[str, Any], provider: ProviderConfig) -> float:
    usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens")) if isinstance(usage, dict) else None
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens")) if isinstance(usage, dict) else None
    if not isinstance(input_tokens, (int, float)):
        input_tokens = _token_estimate(request_payload)
    if not isinstance(output_tokens, (int, float)):
        output_tokens = _token_estimate(payload)
    return (
        max(0, input_tokens) * provider.price_usd_per_1m_tokens_in
        + max(0, output_tokens) * provider.price_usd_per_1m_tokens_out
    ) / 1_000_000


def _unwrap_answer(payload: Any) -> Any:
    if not isinstance(payload, dict) or not isinstance(payload.get("choices"), list):
        return payload
    changed = json.loads(json.dumps(payload))
    for choice in changed["choices"]:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            continue
        try:
            structured = json.loads(message["content"])
        except json.JSONDecodeError:
            continue
        if isinstance(structured, dict) and isinstance(structured.get("answer"), str):
            message["content"] = structured["answer"]
    return changed


def _provider_registry(runtime: Settings) -> list[ProviderConfig]:
    if runtime.providers:
        return list(runtime.providers)
    return [ProviderConfig(
        name=runtime.provider_name or "upstream",
        base_url=runtime.upstream_base_url,
        key_env=runtime.upstream_key_env,
        model_alias="",
        filter_risk=50,
    )]


def _eligible_providers(runtime: Settings) -> list[ProviderConfig]:
    available: list[ProviderConfig] = []
    for provider in sorted(_provider_registry(runtime), key=lambda item: (item.filter_risk, item.name)):
        if provider.filter_risk < 0 or provider.filter_risk > 100:
            logger.warning("provider_skipped_invalid_filter_risk", provider=provider.name)
            continue
        if provider.key_env and not os.getenv(provider.key_env, ""):
            logger.warning("provider_skipped_missing_key", provider=provider.name, key_env=provider.key_env)
            continue
        available.append(provider)
    return available


async def _execute_ladder(request: Request, payload: dict[str, Any]) -> Response:
    runtime = request.app.state.settings
    providers = _eligible_providers(runtime)
    if not providers:
        await request.app.state.store.record_ladder([], 0, [])
        return JSONResponse(
            {"error": "no_provider_available", "attempts": []},
            status_code=502,
            headers={"X-Sidestep-Attempts": "0", "X-Sidestep-Verdict": "unavailable"},
        )

    caller_stream = payload.get("stream") is True
    original_messages = payload.get("messages", [])
    query = f"?{request.url.query}" if request.url.query else ""

    async def invoke(provider: ProviderConfig, outgoing: dict[str, Any], mutation: str, step: int) -> AttemptOutcome:
        call_payload = json.loads(json.dumps(outgoing, ensure_ascii=False))
        if not call_payload.get("model") and provider.model_alias:
            call_payload["model"] = provider.model_alias
        upstream_model = str(call_payload.get("model") or "(omitted)")
        cache_key = verdict_cache_key(provider.name, upstream_model, call_payload.get("messages", []))
        stream_upstream = call_payload.get("stream") is True and mutation != "temperature_0_n_3"
        if not stream_upstream:
            call_payload["stream"] = False
        url = f"{provider.base_url.rstrip('/')}/chat/completions{query}"
        headers = {
            "content-type": "application/json",
            "accept": "text/event-stream" if stream_upstream else request.headers.get("accept", "application/json"),
        }
        api_key = os.getenv(provider.key_env, "") if provider.key_env else ""
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        upstream_request = request.app.state.client.build_request(
            "POST", url, json=call_payload, headers=headers,
        )

        if stream_upstream:
            upstream: httpx.Response | None = None
            try:
                upstream = await request.app.state.client.send(upstream_request, stream=True)
                stream_state, verdict = await _buffer_stream(upstream, call_payload.get("messages", []), 120)
                if stream_state is None:
                    sample_payload = {"choices": [{"message": {"content": ""}}]}
                    cost = _response_cost(sample_payload, call_payload, provider)
                    await request.app.state.store.record(
                        cache_key, provider.name, upstream_model, verdict.label, verdict.score,
                        verdict.reason, upstream.status_code, caller_stream,
                    )
                    return AttemptOutcome(
                        sample_payload, b"", upstream.status_code, _response_headers(upstream.headers),
                        verdict, cost, None, cache_key, upstream_model, caller_stream,
                    )
                sample = stream_state.observer.result() or {
                    "choices": [{"message": {"content": stream_state.initial_text}}],
                }
                cost = _response_cost(sample, call_payload, provider)
                await request.app.state.store.record(
                    cache_key, provider.name, upstream_model, verdict.label, verdict.score,
                    verdict.reason, upstream.status_code, caller_stream,
                )
                return AttemptOutcome(
                    sample, b"", upstream.status_code, _response_headers(upstream.headers),
                    verdict, cost, stream_state, cache_key, upstream_model, caller_stream,
                )
            except BaseException:
                if upstream is not None:
                    await upstream.aclose()
                raise

        try:
            upstream = await request.app.state.client.send(upstream_request)
        except httpx.HTTPError as exc:
            logger.warning("upstream_attempt_unavailable", provider=provider.name, error=type(exc).__name__)
            verdict = Verdict("unavailable", 0.0, "upstream_connection_error")
            await request.app.state.store.record(
                cache_key, provider.name, upstream_model, verdict.label, verdict.score,
                verdict.reason, 502, caller_stream,
            )
            return AttemptOutcome(
                {"error": "upstream_unavailable"}, b"", 502, {}, verdict,
                0.0, None, cache_key, upstream_model, caller_stream,
            )

        response_body = upstream.content
        try:
            response_payload: Any = json.loads(response_body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            response_payload = {"raw": response_body.decode("utf-8", errors="replace")}
        if mutation == "format_lock":
            response_payload = _unwrap_answer(response_payload)
            response_body = json.dumps(response_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        verdict = detect_verdict(response_payload, upstream.status_code, call_payload.get("messages", []))
        cost = _response_cost(response_payload, call_payload, provider)
        await request.app.state.store.record(
            cache_key, provider.name, upstream_model, verdict.label, verdict.score,
            verdict.reason, upstream.status_code, caller_stream,
        )
        logger.info(
            "completion_attempt", provider=provider.name, model=upstream_model,
            mutation=mutation, verdict=verdict.label, score=verdict.score,
            status_code=upstream.status_code, stream=caller_stream,
        )
        return AttemptOutcome(
            response_payload, response_body, upstream.status_code, _response_headers(upstream.headers),
            verdict, cost, None, cache_key, upstream_model, caller_stream,
        )

    result = await run_ladder(
        payload,
        providers,
        runtime.default_provider or runtime.provider_name,
        invoke,
        runtime.max_attempts,
        runtime.max_cost_usd,
        runtime.deadline_ms,
    )
    await request.app.state.store.record_ladder(result.attempts, result.provider_switches, result.budget_stops)
    attempts = [item.as_dict() for item in result.attempts]
    if result.outcome is None:
        error = result.error or "all_upstreams_refused"
        return JSONResponse(
            {"error": error, "attempts": attempts},
            status_code=502,
            headers={
                "X-Sidestep-Attempts": str(len(result.attempts)),
                "X-Sidestep-Verdict": "refusal" if error == "all_upstreams_refused" else "unavailable",
            },
        )

    outcome = result.outcome
    response_headers = dict(outcome.headers)
    response_headers.update({
        "X-Sidestep-Attempts": str(len(result.attempts)),
        "X-Sidestep-Verdict": outcome.verdict.label,
    })
    if outcome.stream_state is not None:
        state: BufferedStream = outcome.stream_state

        async def forward_buffered_stream() -> AsyncIterator[bytes]:
            try:
                for frame in state.initial_frames:
                    yield frame
                if state.pending:
                    yield state.pending
                async for chunk in state.iterator:
                    yield chunk
            finally:
                await state.upstream.aclose()

        return StreamingResponse(
            forward_buffered_stream(), status_code=outcome.status_code,
            headers=response_headers, media_type=None,
        )
    if caller_stream:
        response_headers["content-type"] = "text/event-stream"
        return StreamingResponse(
            iter([_sse_response_body(outcome.payload, outcome.model)]),
            status_code=outcome.status_code, headers=response_headers, media_type=None,
        )
    return Response(content=outcome.body, status_code=outcome.status_code, headers=response_headers)


def _sse_response_body(payload: Any, model: str) -> bytes:
    content = ""
    if isinstance(payload, dict) and isinstance(payload.get("choices"), list) and payload["choices"]:
        choice = payload["choices"][0]
        if isinstance(choice, dict):
            message = choice.get("message") or {}
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                content = message["content"]
    event = {"id": "chatcmpl-sidestep", "object": "chat.completion.chunk", "model": model,
             "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]}
    final = {"id": "chatcmpl-sidestep", "object": "chat.completion.chunk", "model": model,
             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    return (f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            f"data: {json.dumps(final)}\n\n" "data: [DONE]\n\n").encode("utf-8")


def _available_providers(runtime: Settings) -> list[ProviderConfig]:
    providers = sorted(_provider_registry(runtime), key=lambda provider: (provider.filter_risk, provider.name))
    available: list[ProviderConfig] = []
    for provider in providers:
        if provider.key_env and not os.getenv(provider.key_env, ""):
            logger.warning("provider_skipped_missing_key", provider=provider.name, key_env=provider.key_env)
            continue
        available.append(provider)
    return available


async def _ladder_completion(request: Request, payload: dict[str, Any]) -> Response:
    runtime = request.app.state.settings
    providers = _available_providers(runtime)
    if not providers:
        await request.app.state.store.record_ladder([], 0, [])
        return JSONResponse(
            {"error": "no_provider_available", "attempts": []},
            status_code=502,
            headers={"X-Sidestep-Attempts": "0", "X-Sidestep-Verdict": "unavailable"},
        )

    messages = payload.get("messages", [])
    input_stream = payload.get("stream") is True
    query = f"?{request.url.query}" if request.url.query else ""

    async def invoke(provider: ProviderConfig, outgoing_payload: dict[str, Any], mutation: str, step: int) -> AttemptOutcome:
        call_payload = json.loads(json.dumps(outgoing_payload, ensure_ascii=False))
        if provider.model_alias:
            call_payload["model"] = provider.model_alias
        model = str(call_payload.get("model") or "(omitted)")
        if step == 8:
            call_payload["stream"] = False
        use_stream = call_payload.get("stream") is True and mutation not in {
            "decompose", "temperature_0_n_3", "format_lock",
        }
        if "stream" in call_payload or step == 8:
            call_payload["stream"] = use_stream
        cache_key = verdict_cache_key(provider.name, model, call_payload.get("messages", []))
        url = f"{provider.base_url.rstrip('/')}/chat/completions{query}"
        headers = {
            "content-type": "application/json",
            "accept": "text/event-stream" if use_stream else "application/json",
        }
        upstream_key = os.getenv(provider.key_env, "") if provider.key_env else ""
        if upstream_key:
            headers["authorization"] = f"Bearer {upstream_key}"
        request_body = json.dumps(call_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        upstream_request = request.app.state.client.build_request(
            "POST", url, content=request_body, headers=headers,
        )
        try:
            upstream = await request.app.state.client.send(upstream_request, stream=use_stream)
        except httpx.HTTPError as exc:
            logger.warning("ladder_upstream_unavailable", provider=provider.name, error=type(exc).__name__)
            verdict = Verdict("unavailable", 0.0, "upstream_connection_error")
            await request.app.state.store.record(
                cache_key, provider.name, model, verdict.label, verdict.score, verdict.reason, 502, use_stream,
            )
            return AttemptOutcome({}, b"", 502, {}, verdict, 0.0, None, cache_key, model, use_stream)

        if use_stream and "text/event-stream" in upstream.headers.get("content-type", "").lower():
            try:
                stream_state, verdict = await _buffer_stream(upstream, call_payload.get("messages", []), 120)
            except asyncio.CancelledError:
                await upstream.aclose()
                raise
            if stream_state is None:
                partial_payload = {"choices": [{"message": {"content": ""}}]}
                cost = _response_cost(partial_payload, call_payload, provider)
                await request.app.state.store.record(
                    cache_key, provider.name, model, verdict.label, verdict.score, verdict.reason,
                    upstream.status_code, True,
                )
                return AttemptOutcome({}, b"", upstream.status_code, _response_headers(upstream.headers),
                                      verdict, cost, None, cache_key, model, True)
            partial_payload = stream_state.observer.result() or {
                "choices": [{"message": {"content": stream_state.initial_text}}],
            }
            cost = _response_cost(partial_payload, call_payload, provider)
            await request.app.state.store.record(
                cache_key, provider.name, model, verdict.label, verdict.score, verdict.reason,
                upstream.status_code, True,
            )
            return AttemptOutcome(partial_payload, b"", upstream.status_code, _response_headers(upstream.headers),
                                  verdict, cost, stream_state, cache_key, model, True)

        if use_stream:
            response_body = await upstream.aread()
        else:
            response_body = upstream.content
        await upstream.aclose()
        try:
            response_payload: Any = json.loads(response_body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            response_payload = {"raw": response_body.decode("utf-8", errors="replace")}
        if mutation == "format_lock":
            response_payload = _unwrap_answer(response_payload)
            response_body = json.dumps(response_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        verdict = detect_verdict(response_payload, upstream.status_code, call_payload.get("messages", []))
        cost = _response_cost(response_payload, call_payload, provider)
        await request.app.state.store.record(
            cache_key, provider.name, model, verdict.label, verdict.score, verdict.reason,
            upstream.status_code, False,
        )
        return AttemptOutcome(
            response_payload, response_body, upstream.status_code, _response_headers(upstream.headers),
            verdict, cost, None, cache_key, model, False,
        )

    result = await run_ladder(
        payload,
        providers,
        runtime.default_provider or runtime.provider_name,
        invoke,
        max_attempts=runtime.max_attempts,
        max_cost_usd=runtime.max_cost_usd,
        deadline_ms=runtime.deadline_ms,
    )
    await request.app.state.store.record_ladder(result.attempts, result.provider_switches, result.budget_stops)
    attempt_count = len(result.attempts)
    if result.outcome is None:
        error = result.error or "all_upstreams_refused"
        verdict = "unavailable" if error in {"upstream_unavailable", "no_provider_available"} else "refusal"
        return JSONResponse(
            {"error": error, "attempts": [item.as_dict() for item in result.attempts]},
            status_code=502,
            headers={"X-Sidestep-Attempts": str(attempt_count), "X-Sidestep-Verdict": verdict},
        )

    outcome = result.outcome
    headers = dict(outcome.headers)
    headers.update({
        "X-Sidestep-Attempts": str(attempt_count),
        "X-Sidestep-Verdict": outcome.verdict.label,
    })
    if outcome.stream_state is not None:
        state = outcome.stream_state

        async def forward_buffered_stream() -> AsyncIterator[bytes]:
            try:
                for frame in state.initial_frames:
                    yield frame
                if state.pending:
                    yield state.pending
                async for chunk in state.iterator:
                    yield chunk
            finally:
                await state.upstream.aclose()

        return StreamingResponse(
            forward_buffered_stream(),
            status_code=outcome.status_code,
            headers=headers,
            media_type=None,
        )

    if input_stream:
        headers["content-type"] = "text/event-stream"
        headers.pop("content-length", None)
        body = _sse_response_body(outcome.payload, outcome.model)
        return StreamingResponse(
            iter((body,)),
            status_code=outcome.status_code,
            headers=headers,
            media_type=None,
        )
    return Response(content=outcome.body, status_code=outcome.status_code, headers=headers)


def create_app(
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    runtime_settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if not runtime_settings.token:
            raise RuntimeError("SIDESTEP_TOKEN is required; set a non-empty bearer token")
        store = Store(runtime_settings.database_path)
        await store.initialize()
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(runtime_settings.timeout_seconds),
            transport=transport,
        )
        application.state.settings = runtime_settings
        application.state.observe_only = (
            runtime_settings.observe_only if runtime_settings.observe_only is not None else settings is not None
        )
        application.state.store = store
        application.state.client = client
        yield
        await client.aclose()

    application = FastAPI(
        title="Sidestep Audit Gateway",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    async def authenticate(request: Request) -> None:
        expected = request.app.state.settings.token
        supplied = request.headers.get("authorization", "")
        if not supplied.startswith("Bearer ") or not hmac.compare_digest(
            supplied[7:].encode("utf-8"), expected.encode("utf-8")
        ):
            raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Bearer"})

    async def health() -> dict[str, str]:
        observe_only = runtime_settings.observe_only if runtime_settings.observe_only is not None else settings is not None
        return {"ok": "true", "mode": "observe-only" if observe_only else "ladder"}

    async def stats(request: Request, _: None = Depends(authenticate)) -> dict[str, Any]:
        return await request.app.state.store.stats(include_ladder=not request.app.state.observe_only)

    async def refusals(request: Request, _: None = Depends(authenticate)) -> dict[str, Any]:
        try:
            limit = int(request.query_params.get("limit", "100"))
        except ValueError:
            raise HTTPException(status_code=400, detail="limit must be an integer")
        return {"refusals": await request.app.state.store.refusals(limit)}

    async def models(request: Request, _: None = Depends(authenticate)) -> Response:
        runtime = request.app.state.settings
        providers = (
            _provider_registry(runtime)
            if request.app.state.observe_only and not runtime.providers
            else _available_providers(runtime)
        )
        if not providers:
            return JSONResponse({"error": "no_provider_available"}, status_code=502)
        provider = next((item for item in providers if item.name == runtime.default_provider), providers[0])
        url = f"{provider.base_url.rstrip('/')}/models"
        headers = {"accept": request.headers.get("accept", "application/json")}
        upstream_key = os.getenv(provider.key_env, "") if provider.key_env else ""
        if upstream_key:
            headers["authorization"] = f"Bearer {upstream_key}"
        upstream_request = request.app.state.client.build_request("GET", url, headers=headers)
        try:
            upstream = await request.app.state.client.send(upstream_request)
        except httpx.HTTPError as exc:
            logger.warning("upstream_models_unavailable", provider=provider.name, error=type(exc).__name__)
            return JSONResponse({"error": "upstream_unavailable"}, status_code=502)
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=_response_headers(upstream.headers),
        )

    async def completions(request: Request, _: None = Depends(authenticate)) -> Response:
        raw_body = await request.body()
        try:
            payload = json.loads(raw_body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise HTTPException(status_code=400, detail="Request body must be valid JSON")
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")

        runtime = request.app.state.settings
        provider = runtime.provider_name
        if not request.app.state.observe_only:
            return await _ladder_completion(request, payload)
        if runtime.providers and runtime.upstream_key_env and not os.getenv(runtime.upstream_key_env, ""):
            logger.warning(
                "provider_skipped_missing_key",
                provider=runtime.provider_name,
                key_env=runtime.upstream_key_env,
            )
            return JSONResponse(
                {"error": "no_provider_available"},
                status_code=502,
                headers={"X-Sidestep-Attempts": "0", "X-Sidestep-Verdict": "unavailable"},
            )
        model = str(payload.get("model") or "(omitted)")
        messages = payload.get("messages", [])
        fingerprint = verdict_cache_key(provider, model, messages)
        is_stream = payload.get("stream") is True
        url = f"{runtime.upstream_base_url.rstrip('/')}/chat/completions"
        if request.url.query:
            url = f"{url}?{request.url.query}"
        outgoing_headers = {
            "content-type": request.headers.get("content-type", "application/json"),
            "accept": request.headers.get("accept", "application/json"),
        }
        upstream_key = os.getenv(runtime.upstream_key_env, "")
        if upstream_key:
            outgoing_headers["authorization"] = f"Bearer {upstream_key}"
        upstream_request = request.app.state.client.build_request(
            "POST", url, content=raw_body, headers=outgoing_headers,
        )

        if is_stream:
            try:
                upstream = await request.app.state.client.send(upstream_request, stream=True)
            except httpx.HTTPError as exc:
                logger.warning("upstream_stream_unavailable", error=type(exc).__name__)
                await request.app.state.store.record(
                    fingerprint, provider, model, "unavailable", 0.0,
                    "upstream_connection_error", 502, True,
                )
                return JSONResponse(
                    {"error": "upstream_unavailable"},
                    status_code=502,
                    headers={"X-Sidestep-Attempts": "1", "X-Sidestep-Verdict": "unavailable"},
                )

            observer = StreamObserver()

            async def forward_stream() -> AsyncIterator[bytes]:
                completed = False
                try:
                    async for chunk in upstream.aiter_bytes():
                        observer.feed(chunk)
                        yield chunk
                    completed = True
                finally:
                    await upstream.aclose()
                    result = observer.result() if completed else None
                    verdict = (
                        detect_verdict(result, upstream.status_code, messages)
                        if result is not None
                        else Verdict("uncertain", 0.0, "stream_incomplete")
                    )
                    await request.app.state.store.record(
                        fingerprint, provider, model, verdict.label, verdict.score,
                        verdict.reason, upstream.status_code, True,
                    )
                    logger.info(
                        "completion_observed", provider=provider, model=model,
                        verdict=verdict.label, score=verdict.score, stream=True,
                        status_code=upstream.status_code,
                    )

            headers = _response_headers(upstream.headers)
            headers.update({"X-Sidestep-Attempts": "1", "X-Sidestep-Verdict": "pending"})
            return StreamingResponse(
                forward_stream(),
                status_code=upstream.status_code,
                headers=headers,
                media_type=None,
            )

        try:
            upstream = await request.app.state.client.send(upstream_request)
        except httpx.HTTPError as exc:
            logger.warning("upstream_completion_unavailable", error=type(exc).__name__)
            await request.app.state.store.record(
                fingerprint, provider, model, "unavailable", 0.0,
                "upstream_connection_error", 502, False,
            )
            return JSONResponse(
                {"error": "upstream_unavailable"},
                status_code=502,
                headers={"X-Sidestep-Attempts": "1", "X-Sidestep-Verdict": "unavailable"},
            )

        response_body = upstream.content
        try:
            response_payload: Any = json.loads(response_body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            response_payload = {"raw": response_body.decode("utf-8", errors="replace")}
        verdict: Verdict = detect_verdict(response_payload, upstream.status_code, messages)
        await request.app.state.store.record(
            fingerprint, provider, model, verdict.label, verdict.score,
            verdict.reason, upstream.status_code, False,
        )
        logger.info(
            "completion_observed", provider=provider, model=model,
            verdict=verdict.label, score=verdict.score, stream=False,
            status_code=upstream.status_code,
        )
        headers = _response_headers(upstream.headers)
        headers.update({"X-Sidestep-Attempts": "1", "X-Sidestep-Verdict": verdict.label})
        return Response(content=response_body, status_code=upstream.status_code, headers=headers)

    @application.middleware("http")
    async def add_sidestep_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Sidestep-Attempts", "0")
        response.headers.setdefault("X-Sidestep-Verdict", "not_evaluated")
        return response

    application.add_api_route("/healthz", health, methods=["GET"])
    application.add_api_route("/stats", stats, methods=["GET"])
    application.add_api_route("/refusals", refusals, methods=["GET"])
    application.add_api_route("/v1/models", models, methods=["GET"])
    application.add_api_route("/v1/chat/completions", completions, methods=["POST"])
    return application


app = create_app()
