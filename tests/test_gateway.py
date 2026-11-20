import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest
from httpx import ASGITransport, MockTransport

from sidestep_audit.app import create_app
from sidestep_audit.config import Settings
from sidestep_audit.detection import detect_verdict, verdict_cache_key


TOKEN = "test-bearer-token"
ROOT = Path(__file__).parent


@asynccontextmanager
async def gateway_client(handler: Callable[[httpx.Request], httpx.Response], tmp_path: Path):
    settings = Settings(
        token=TOKEN,
        database_path=str(tmp_path / "audit.sqlite3"),
        upstream_base_url="https://upstream.test/v1",
        provider_name="test-provider",
        upstream_key_env="UNSET_TEST_PROVIDER_KEY",
        timeout_seconds=5,
    )
    app = create_app(settings, MockTransport(handler))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://gateway.test"
        ) as client:
            yield client, app


def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def refusal_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "I'm sorry, I can't help with that request. I can offer safe, high-level context.",
                },
                "finish_reason": "stop",
            }],
        },
    )


@pytest.mark.asyncio
async def test_requires_bearer_token_and_health_is_public(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("unauthenticated request reached upstream")

    async with gateway_client(handler, tmp_path) as (client, _):
        assert (await client.get("/healthz")).json()["mode"] == "observe-only"
        response = await client.post("/v1/chat/completions", json={"model": "m", "messages": []})
        assert response.status_code == 401


@pytest.mark.asyncio
async def test_refusal_is_forwarded_unchanged_and_recorded_once(tmp_path: Path) -> None:
    body = {
        "model": "example-model",
        "messages": [{"role": "user", "content": "Discuss a sensitive topic at a high level."}],
    }
    raw_body = json.dumps(body, separators=(",", ":")).encode()
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.method == "POST"
        assert request.url == "https://upstream.test/v1/chat/completions"
        assert request.content == raw_body
        return refusal_response()

    async with gateway_client(handler, tmp_path) as (client, _):
        response = await client.post(
            "/v1/chat/completions", content=raw_body,
            headers={**auth_headers(), "Content-Type": "application/json"},
        )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"].startswith("I'm sorry")
        assert response.headers["X-Sidestep-Attempts"] == "1"
        assert response.headers["X-Sidestep-Verdict"] == "refusal"
        assert len(seen) == 1
        stats = (await client.get("/stats", headers=auth_headers())).json()
        assert stats == {
            "requests": 1,
            "refusals": 1,
            "uncertain": 0,
            "upstream_unavailable": 0,
            "streamed": 0,
            "refusal_rate": 1.0,
        }


@pytest.mark.asyncio
async def test_non_refusal_response_is_transparent(tmp_path: Path) -> None:
    payload = {"model": "m", "messages": [{"role": "user", "content": "Explain a benign topic."}]}
    upstream_body = {
        "id": "chatcmpl-ok",
        "choices": [{"message": {"role": "assistant", "content": "Here is a clear explanation."}, "finish_reason": "stop"}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == payload
        return httpx.Response(201, json=upstream_body, headers={"x-upstream": "test"})

    async with gateway_client(handler, tmp_path) as (client, _):
        response = await client.post("/v1/chat/completions", json=payload, headers=auth_headers())
        assert response.status_code == 201
        assert response.json() == upstream_body
        assert response.headers["x-upstream"] == "test"
        assert response.headers["X-Sidestep-Verdict"] == "non_refusal"
        assert response.headers["X-Sidestep-Attempts"] == "1"


@pytest.mark.asyncio
async def test_stream_is_forwarded_byte_for_byte_and_audited_afterward(tmp_path: Path) -> None:
    chunks = [
        {"choices": [{"delta": {"content": "I cannot assist with that request."}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    stream_body = b"".join(
        f"data: {json.dumps(event)}\n\n".encode() for event in chunks
    ) + b"data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_body)

    async with gateway_client(handler, tmp_path) as (client, _):
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "A safe question."}]},
            headers=auth_headers(),
        )
        assert response.content == stream_body
        assert response.headers["X-Sidestep-Attempts"] == "1"
        assert response.headers["X-Sidestep-Verdict"] == "pending"
        stats = (await client.get("/stats", headers=auth_headers())).json()
        assert stats["requests"] == 1
        assert stats["refusals"] == 1
        assert stats["streamed"] == 1


@pytest.mark.asyncio
async def test_upstream_failure_is_not_retried_and_is_counted(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, json={"error": {"message": "temporarily unavailable"}})

    async with gateway_client(handler, tmp_path) as (client, _):
        response = await client.post(
            "/v1/chat/completions", json={"model": "m", "messages": []}, headers=auth_headers()
        )
        assert response.status_code == 503
        assert response.json()["error"]["message"] == "temporarily unavailable"
        assert response.headers["X-Sidestep-Attempts"] == "1"
        assert response.headers["X-Sidestep-Verdict"] == "unavailable"
        assert calls == 1
        stats = (await client.get("/stats", headers=auth_headers())).json()
        assert stats["upstream_unavailable"] == 1


@pytest.mark.asyncio
async def test_models_endpoint_forwards_to_single_upstream(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url == "https://upstream.test/v1/models"
        return httpx.Response(200, json={"data": [{"id": "m"}]})

    async with gateway_client(handler, tmp_path) as (client, _):
        response = await client.get("/v1/models", headers=auth_headers())
        assert response.status_code == 200
        assert response.json() == {"data": [{"id": "m"}]}


def test_layered_detection_signals() -> None:
    assert detect_verdict(
        {"choices": [{"message": {"content": "I can't help with that."}, "finish_reason": "stop"}]}
    ).label == "refusal"
    assert detect_verdict({"error": "content_filter"}, status_code=403).reason == "http_content_filter"
    assert detect_verdict({"choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}]}).label == "refusal"
    assert detect_verdict(
        {"choices": [{"message": {"content": "Here is a concise overview."}, "finish_reason": "stop"}]}
    ).label == "non_refusal"
    assert detect_verdict({"error": "service unavailable"}, status_code=503).label == "unavailable"


def test_verdict_cache_key_is_provider_and_model_scoped() -> None:
    prompt = [{"role": "user", "content": "What is a safe high-level explanation?"}]
    key = verdict_cache_key("provider-a", "model-a", prompt)
    assert key == verdict_cache_key("provider-a", "model-a", prompt)
    assert key != verdict_cache_key("provider-b", "model-a", prompt)
    assert key != verdict_cache_key("provider-a", "model-b", prompt)
    assert key != verdict_cache_key("provider-a", "model-a", "different prompt")


def test_fixture_has_40_safety_framed_prompts_and_is_not_sent_upstream() -> None:
    cases = json.loads((ROOT / "fixtures" / "policy_edge_eval.json").read_text())
    assert len(cases) == 40
    assert len({case["id"] for case in cases}) == 40
    assert all(case["prompt"].strip() for case in cases)
