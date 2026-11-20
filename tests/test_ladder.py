import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from httpx import ASGITransport, AsyncClient, MockTransport

from sidestep_audit.app import create_app
from sidestep_audit.config import ProviderConfig, Settings
from sidestep_audit.ladder import BOTH_POSITIONS, DIRECT_SUFFIX, PERSONA_INSTRUCTION

ROOT = Path(__file__).parent
AUTH = {"Authorization": "Bearer test-token"}
REFUSAL = "I'm sorry, I cannot help with that request."


@asynccontextmanager
async def gateway_client(handler, settings: Settings):
    app = create_app(settings, transport=MockTransport(handler))
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield client, app


def make_provider(name: str, key_env: str, risk: int, *, input_price: float = 0, output_price: float = 0) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        base_url=f"https://{name}.test/v1",
        key_env=key_env,
        model_alias="test-model",
        filter_risk=risk,
        price_usd_per_1m_tokens_in=input_price,
        price_usd_per_1m_tokens_out=output_price,
    )


def settings_for(
    tmp_path: Path,
    *,
    providers=None,
    max_attempts: int = 8,
    max_cost_usd: float = 0.25,
    deadline_ms: int = 45000,
    observe_only: bool | None = False,
    upstream_base_url: str = "",
    provider_name: str = "",
    upstream_key_env: str = "",
) -> Settings:
    return Settings(
        token="test-token",
        database_path=str(tmp_path / "sidestep.sqlite3"),
        upstream_base_url=upstream_base_url,
        provider_name=provider_name,
        upstream_key_env=upstream_key_env,
        timeout_seconds=3,
        providers=providers or [
            make_provider("p1", "P1_KEY", 60),
            make_provider("p2", "P2_KEY", 5),
            make_provider("p3", "P3_KEY", 30),
        ],
        default_provider="p1",
        observe_only=observe_only,
        max_attempts=max_attempts,
        max_cost_usd=max_cost_usd,
        deadline_ms=deadline_ms,
    )


def set_provider_keys(monkeypatch) -> None:
    monkeypatch.setenv("P1_KEY", "test-p1-key")
    monkeypatch.setenv("P2_KEY", "test-p2-key")
    monkeypatch.setenv("P3_KEY", "test-p3-key")


def provider_for(request: httpx.Request) -> str:
    return {"Bearer test-p1-key": "p1", "Bearer test-p2-key": "p2", "Bearer test-p3-key": "p3"}.get(
        request.headers.get("authorization", ""), "unknown"
    )


def body_for(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content)


def ladder_step(request: httpx.Request) -> int:
    body = body_for(request)
    if body.get("temperature") == 0 and body.get("n") == 3:
        return 8
    if provider_for(request) == "p2":
        return 7
    messages = body.get("messages", [])
    text = "\n".join(str(message.get("content", "")) for message in messages if isinstance(message, dict))
    if body.get("response_format", {}).get("type") == "json_schema":
        return 4
    if PERSONA_INSTRUCTION in text:
        return 6
    if text.startswith("Identify the main facts") or text.startswith("Identify important caveats") or text.startswith("Synthesize a complete"):
        return 5
    if BOTH_POSITIONS in text:
        return 3
    if DIRECT_SUFFIX in text:
        return 2
    return 1


def completion(content: str, *, usage: dict[str, int] | None = None, choices: list[dict[str, Any]] | None = None) -> httpx.Response:
    return httpx.Response(200, json={
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "test-model",
        "choices": choices or [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        **({"usage": usage} if usage is not None else {}),
    })


@pytest.mark.asyncio
@pytest.mark.parametrize("target_step", range(1, 9))
async def test_each_ladder_step_fires_in_order(target_step: int, tmp_path: Path, monkeypatch) -> None:
    set_provider_keys(monkeypatch)
    seen: list[tuple[int, str, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        step = ladder_step(request)
        provider = provider_for(request)
        body = body_for(request)
        seen.append((step, provider, body))
        if step == target_step:
            if step == 4:
                return completion(json.dumps({"answer": "Structured answer.", "notes": "No notes."}))
            if step == 8:
                return completion("", choices=[
                    {"index": 0, "message": {"role": "assistant", "content": REFUSAL}, "finish_reason": "stop"},
                    {"index": 1, "message": {"role": "assistant", "content": "Short usable answer."}, "finish_reason": "stop"},
                    {"index": 2, "message": {"role": "assistant", "content": "The longest usable completion was selected."}, "finish_reason": "stop"},
                ])
            return completion(f"Success at ladder step {step}.")
        return completion(REFUSAL)

    settings = settings_for(tmp_path)
    async with gateway_client(handler, settings) as (client, _):
        original = {"model": "caller-model", "messages": [{"role": "user", "content": "What is two plus two?"}]}
        response = await client.post("/v1/chat/completions", json=original, headers=AUTH)

    assert response.status_code == 200
    assert response.headers["X-Sidestep-Attempts"] == str(target_step)
    assert response.headers["X-Sidestep-Verdict"] == "non_refusal"
    if target_step == 5:
        assert [entry[0] for entry in seen[:4]] == [1, 2, 3, 4]
        assert all(entry[0] == 5 for entry in seen[4:])
    else:
        assert [entry[0] for entry in seen] == [*range(1, target_step + 1)]
    if target_step == 1:
        assert seen[0][2]["messages"] == original["messages"]
        assert seen[0][2]["model"] == "test-model"
        assert "stream" not in seen[0][2]

    if target_step == 2:
        assert seen[-1][2]["messages"][-1]["content"].endswith(DIRECT_SUFFIX)
    if target_step == 3:
        assert seen[-1][2]["messages"][-1]["content"].startswith(BOTH_POSITIONS)
        assert seen[-1][2]["messages"][-1]["content"].endswith(BOTH_POSITIONS)
    if target_step == 4:
        assert response.json()["choices"][0]["message"]["content"] == "Structured answer."
        schema = seen[-1][2]["response_format"]["json_schema"]["schema"]
        assert schema["required"] == ["answer", "notes"]
    if target_step == 5:
        assert len(seen) >= 7
        assert response.json()["choices"][0]["message"]["content"].startswith("Success at ladder step 5.")
    if target_step == 6:
        assert PERSONA_INSTRUCTION in seen[-1][2]["messages"][0]["content"]
    if target_step == 7:
        assert seen[-1][1] == "p2"
        assert [item[1] for item in seen[:6]] == ["p1"] * 6
    if target_step == 8:
        assert seen[-1][1] == "p2"
        assert seen[-1][2]["temperature"] == 0 and seen[-1][2]["n"] == 3
        assert response.json()["choices"][0]["message"]["content"] == "The longest usable completion was selected."


@pytest.mark.asyncio
async def test_routes_to_next_provider_by_ascending_risk(tmp_path: Path, monkeypatch) -> None:
    set_provider_keys(monkeypatch)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        provider = provider_for(request)
        seen.append(provider)
        if provider == "p2":
            return completion("Answer from the next-lowest-risk provider.")
        return completion(REFUSAL)

    settings = settings_for(tmp_path)
    async with gateway_client(handler, settings) as (client, _):
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "caller-model", "messages": [{"role": "user", "content": "Explain this."}]},
            headers=AUTH,
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Answer from the next-lowest-risk provider."
    assert seen == ["p1"] * 6 + ["p2"]
    assert response.headers["X-Sidestep-Attempts"] == "7"


@pytest.mark.asyncio
async def test_streaming_refusal_buffer_is_discarded_before_retry(tmp_path: Path, monkeypatch) -> None:
    set_provider_keys(monkeypatch)
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = body_for(request)
        seen.append(body)
        if len(seen) == 1:
            event = {"choices": [{"delta": {"content": "I cannot provide that refused fragment."}, "finish_reason": None}]}
            stream = f"data: {json.dumps(event)}\n\n".encode() + b"data: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream)
        return completion("Clean success after refusal.")

    settings = settings_for(tmp_path)
    async with gateway_client(handler, settings) as (client, _):
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "caller-model", "stream": True, "messages": [{"role": "user", "content": "Test a stream."}]},
            headers=AUTH,
        )

    assert response.status_code == 200
    assert response.headers["X-Sidestep-Attempts"] == "2"
    assert "Clean success after refusal." in response.text
    assert "refused fragment" not in response.text
    assert response.headers["X-Sidestep-Verdict"] == "non_refusal"


@pytest.mark.asyncio
async def test_max_attempts_stops_after_cap(tmp_path: Path, monkeypatch) -> None:
    set_provider_keys(monkeypatch)
    seen = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen
        seen += 1
        return completion(REFUSAL)

    settings = settings_for(tmp_path, max_attempts=2)
    async with gateway_client(handler, settings) as (client, _):
        response = await client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Test."}]}, headers=AUTH)
        stats = (await client.get("/stats", headers=AUTH)).json()

    assert seen == 2
    assert response.status_code == 502
    assert response.json()["error"] == "all_upstreams_refused"
    assert len(response.json()["attempts"]) == 2
    assert stats["budget_stops"] == 1


@pytest.mark.asyncio
async def test_max_cost_stops_after_cost_breach(tmp_path: Path, monkeypatch) -> None:
    set_provider_keys(monkeypatch)
    seen = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen
        seen += 1
        return completion(REFUSAL, usage={"prompt_tokens": 1000, "completion_tokens": 0})

    providers = [make_provider("p1", "P1_KEY", 10, input_price=500), make_provider("p2", "P2_KEY", 20)]
    settings = settings_for(tmp_path, providers=providers, max_cost_usd=0.25)
    async with gateway_client(handler, settings) as (client, _):
        response = await client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Test."}]}, headers=AUTH)
        stats = (await client.get("/stats", headers=AUTH)).json()

    assert seen == 1
    assert response.status_code == 502
    assert response.json()["error"] == "all_upstreams_refused"
    assert stats["budget_stops"] == 1
    assert response.json()["attempts"][0]["cost_usd"] == 0.5


@pytest.mark.asyncio
async def test_deadline_stops_loop_and_returns_best_decompose_answer(tmp_path: Path, monkeypatch) -> None:
    set_provider_keys(monkeypatch)
    seen = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen
        seen += 1
        step = ladder_step(request)
        if step == 5 and seen >= 6:
            await asyncio.sleep(0.2)
        if step == 5 and seen == 5:
            return completion("First decomposition answer is usable.")
        if step == 5 and seen > 5:
            return completion("Second decomposition answer.")
        return completion(REFUSAL)

    settings = settings_for(tmp_path, deadline_ms=80)
    async with gateway_client(handler, settings) as (client, _):
        response = await client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Test."}]}, headers=AUTH)
        stats = (await client.get("/stats", headers=AUTH)).json()

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "First decomposition answer is usable."
    assert stats["budget_stops"] == 1


@pytest.mark.asyncio
async def test_all_refused_returns_full_attempt_log(tmp_path: Path, monkeypatch) -> None:
    set_provider_keys(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        if body_for(request).get("n") == 3:
            return completion("", choices=[
                {"index": index, "message": {"role": "assistant", "content": REFUSAL}, "finish_reason": "stop"}
                for index in range(3)
            ])
        return completion(REFUSAL)

    settings = settings_for(tmp_path)
    async with gateway_client(handler, settings) as (client, _):
        response = await client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Test."}]}, headers=AUTH)
        refusal_response = await client.get("/refusals", headers=AUTH)

    assert refusal_response.status_code == 200
    assert refusal_response.json()["refusals"]
    assert response.status_code == 502
    data = response.json()
    assert data["error"] == "all_upstreams_refused"
    assert [entry["step"] for entry in data["attempts"]] == list(range(1, 9))
    assert all({"mutation", "provider", "rule"} <= entry.keys() for entry in data["attempts"])
    assert data["attempts"][6]["provider"] == "p2"
    assert response.headers["X-Sidestep-Verdict"] == "refusal"


@pytest.mark.asyncio
async def test_observe_only_env_forwards_refusal_without_retry(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SIDESTEP_OBSERVE_ONLY", "1")
    monkeypatch.setenv("P1_KEY", "test-p1-key")
    seen: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.content)
        return httpx.Response(200, content=b'{"choices":[{"message":{"content":"I cannot help with that."}}]}', headers={"content-type": "application/json"})

    settings = settings_for(
        tmp_path,
        observe_only=None,
        upstream_base_url="https://p1.test/v1",
        provider_name="p1",
        upstream_key_env="P1_KEY",
    )
    async with gateway_client(handler, settings) as (client, _):
        request_body = {"model": "caller-model", "messages": [{"role": "user", "content": "Test."}]}
        response = await client.post("/v1/chat/completions", json=request_body, headers=AUTH)

    assert response.status_code == 200
    assert response.content == b'{"choices":[{"message":{"content":"I cannot help with that."}}]}'
    assert len(seen) == 1
    assert json.loads(seen[0]) == request_body
    assert response.headers["X-Sidestep-Attempts"] == "1"
    assert response.headers["X-Sidestep-Verdict"] == "refusal"


@pytest.mark.asyncio
async def test_missing_provider_keys_are_skipped_without_crash(tmp_path: Path, monkeypatch, capsys) -> None:
    for key in ("P1_KEY", "P2_KEY", "P3_KEY"):
        monkeypatch.delenv(key, raising=False)
    settings = settings_for(tmp_path)
    async with gateway_client(lambda request: completion("unused"), settings) as (client, _):
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Test."}]},
            headers=AUTH,
        )
    assert response.status_code == 502
    assert response.json()["error"] == "no_provider_available"
    assert "provider_skipped_missing_key" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_provider_without_key_is_skipped_and_next_provider_runs(tmp_path: Path, monkeypatch) -> None:
    set_provider_keys(monkeypatch)
    monkeypatch.delenv("P1_KEY", raising=False)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(provider_for(request))
        return completion("The configured fallback provider succeeded.")

    settings = settings_for(tmp_path)
    async with gateway_client(handler, settings) as (client, _):
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Test."}]},
            headers=AUTH,
        )

    assert response.status_code == 200
    assert seen == ["p2"]


@pytest.mark.asyncio
async def test_unset_observe_flag_defaults_to_ladder(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SIDESTEP_OBSERVE_ONLY", raising=False)
    monkeypatch.setenv("SIDESTEP_CONFIG", str(ROOT.parent / "config.yaml"))
    monkeypatch.setenv("SIDESTEP_TOKEN", "test-token")
    from sidestep_audit.config import load_settings

    settings = load_settings()
    assert settings.observe_only is False
    assert settings.max_attempts == 8
    assert settings.max_cost_usd == 0.25
    assert settings.deadline_ms == 45000

@pytest.mark.asyncio
async def test_observe_only_loads_registry_default_upstream(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SIDESTEP_OBSERVE_ONLY", "1")
    monkeypatch.setenv("SIDESTEP_CONFIG", str(ROOT.parent / "config.yaml"))
    monkeypatch.setenv("SIDESTEP_TOKEN", "test-token")
    from sidestep_audit.config import load_settings

    settings = load_settings()
    assert settings.observe_only is True
    assert settings.upstream_base_url == "http://mock-upstream:4100/v1"
    assert settings.provider_name == "mock-primary"
    assert settings.upstream_key_env == "MOCK_UPSTREAM_KEY"

@pytest.mark.asyncio
async def test_missing_provider_keys_are_skipped_without_crashing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("P1_KEY", raising=False)
    monkeypatch.delenv("P2_KEY", raising=False)
    monkeypatch.delenv("P3_KEY", raising=False)
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return completion("Unexpected upstream call.")

    async with gateway_client(handler, settings_for(tmp_path)) as (client, _):
        response = await client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Test."}]}, headers=AUTH)

    assert response.status_code == 502
    assert response.json() == {"error": "no_provider_available", "attempts": []}
    assert response.headers["X-Sidestep-Verdict"] == "unavailable"
    assert called is False
