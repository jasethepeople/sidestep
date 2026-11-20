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
from sidestep_audit.ladder import (
    BOTH_POSITIONS,
    DIRECT_SUFFIX,
    MUTATION_PASS,
    PERSONA_INSTRUCTION,
    plan_attempts,
)

ROOT = Path(__file__).parent
AUTH = {"Authorization": "Bearer test-token"}
REFUSAL = "I'm sorry, I cannot help with that request."


@asynccontextmanager
async def gateway_client(handler, settings: Settings):
    app = create_app(settings, transport=MockTransport(handler))
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield client, app


def make_provider(
    name: str,
    key_env: str,
    risk: int,
    *,
    input_price: float = 0,
    output_price: float = 0,
    supports_json_schema: bool = False,
    supports_n: bool = True,
) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        base_url=f"https://{name}.test/v1",
        key_env=key_env,
        model_alias="test-model",
        filter_risk=risk,
        price_usd_per_1m_tokens_in=input_price,
        price_usd_per_1m_tokens_out=output_price,
        supports_json_schema=supports_json_schema,
        supports_n=supports_n,
    )


def settings_for(
    tmp_path: Path,
    *,
    providers=None,
    max_attempts: int = 12,
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


def single_provider_settings(tmp_path: Path, **kwargs) -> Settings:
    return settings_for(
        tmp_path,
        providers=[make_provider("p1", "P1_KEY", 10, **kwargs)],
    )


def set_provider_keys(monkeypatch) -> None:
    monkeypatch.setenv("P1_KEY", "test-p1-key")
    monkeypatch.setenv("P2_KEY", "test-p2-key")
    monkeypatch.setenv("P3_KEY", "test-p3-key")


def provider_for(request: httpx.Request) -> str:
    return {
        "Bearer test-p1-key": "p1",
        "Bearer test-p2-key": "p2",
        "Bearer test-p3-key": "p3",
    }.get(request.headers.get("authorization", ""), "unknown")


def body_for(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content)


def mutation_for(request: httpx.Request) -> str:
    """Identify which mutation produced this upstream request."""
    body = body_for(request)
    if body.get("temperature") == 0 and body.get("n") == 3:
        return "temperature_0_n_3"
    if body.get("response_format", {}).get("type") == "json_schema":
        return "format_lock"
    messages = body.get("messages", [])
    if messages and isinstance(messages[0], dict) and PERSONA_INSTRUCTION in str(messages[0].get("content", "")):
        return "persona_escalation"
    text = "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
    if (
        text.startswith("Identify the main facts")
        or text.startswith("Identify important caveats")
        or text.startswith("Synthesize a complete")
    ):
        return "decompose"
    if BOTH_POSITIONS in text:
        return "both_positions"
    if DIRECT_SUFFIX in text:
        return "direct_suffix"
    return "verbatim_passthrough"


def completion(
    content: str,
    *,
    usage: dict[str, int] | None = None,
    choices: list[dict[str, Any]] | None = None,
) -> httpx.Response:
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


def log(seen: list[tuple[str, str]], request: httpx.Request) -> None:
    seen.append((provider_for(request), mutation_for(request)))


# --- plan construction -------------------------------------------------------


def test_plan_puts_every_provider_verbatim_first() -> None:
    providers = [make_provider("p1", "K1", 60), make_provider("p2", "K2", 5), make_provider("p3", "K3", 30)]
    plan = plan_attempts(providers, 12)
    assert [provider.name for provider, _ in plan[:3]] == ["p1", "p2", "p3"]
    assert [mutation for _, mutation in plan[:3]] == ["verbatim_passthrough"] * 3


def test_plan_runs_mutations_against_lowest_risk_provider_only() -> None:
    providers = [make_provider("p1", "K1", 60), make_provider("p2", "K2", 5)]
    plan = plan_attempts(providers, 12)
    tail = plan[len(providers):]
    assert tail
    assert {provider.name for provider, _ in tail} == {"p1"}


def test_plan_omits_capability_gated_mutations() -> None:
    plain = [make_provider("p1", "K1", 10)]
    capable = [make_provider("p1", "K1", 10, supports_json_schema=True, supports_n=False)]

    plain_mutations = {mutation for _, mutation in plan_attempts(plain, 20)}
    capable_mutations = {mutation for _, mutation in plan_attempts(capable, 20)}

    assert "format_lock" not in plain_mutations
    assert "temperature_0_n_3" in plain_mutations
    assert "format_lock" in capable_mutations
    assert "temperature_0_n_3" not in capable_mutations


def test_plan_ends_with_decompose() -> None:
    plan = plan_attempts([make_provider("p1", "K1", 10)], 20)
    assert plan[-1][1] == "decompose"


def test_mutation_pass_is_the_documented_order() -> None:
    assert [mutation for mutation, _ in MUTATION_PASS] == [
        "direct_suffix",
        "both_positions",
        "format_lock",
        "persona_escalation",
        "temperature_0_n_3",
    ]


# --- ladder execution -------------------------------------------------------


@pytest.mark.asyncio
async def test_ladder_hits_each_provider_before_mutating(tmp_path: Path, monkeypatch) -> None:
    set_provider_keys(monkeypatch)
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        log(seen, request)
        return completion(REFUSAL)

    async with gateway_client(handler, settings_for(tmp_path)) as (client, _):
        await client.post(
            "/v1/chat/completions",
            json={"model": "caller-model", "messages": [{"role": "user", "content": "Explain this."}]},
            headers=AUTH,
        )

    assert [provider for provider, _ in seen[:3]] == ["p1", "p2", "p3"]
    assert all(mutation == "verbatim_passthrough" for _, mutation in seen[:3])
    assert all(provider == "p1" for provider, _ in seen[3:])


@pytest.mark.asyncio
async def test_routes_to_next_provider_by_ascending_risk(tmp_path: Path, monkeypatch) -> None:
    set_provider_keys(monkeypatch)
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        log(seen, request)
        if provider_for(request) == "p2":
            return completion("Answer from the next-lowest-risk provider.")
        return completion(REFUSAL)

    async with gateway_client(handler, settings_for(tmp_path)) as (client, _):
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "caller-model", "messages": [{"role": "user", "content": "Explain this."}]},
            headers=AUTH,
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Answer from the next-lowest-risk provider."
    assert seen == [("p1", "verbatim_passthrough"), ("p2", "verbatim_passthrough")]
    assert response.headers["X-Sidestep-Attempts"] == "2"


@pytest.mark.asyncio
async def test_format_lock_answer_is_unwrapped(tmp_path: Path, monkeypatch) -> None:
    set_provider_keys(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        if mutation_for(request) == "format_lock":
            return completion(json.dumps({"answer": "Structured answer.", "notes": "No notes."}))
        return completion(REFUSAL)

    settings = single_provider_settings(tmp_path, supports_json_schema=True)
    async with gateway_client(handler, settings) as (client, _):
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "What is two plus two?"}]},
            headers=AUTH,
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Structured answer."


@pytest.mark.asyncio
async def test_longest_non_refusing_choice_is_selected(tmp_path: Path, monkeypatch) -> None:
    set_provider_keys(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        if mutation_for(request) == "temperature_0_n_3":
            return completion("", choices=[
                {"index": 0, "message": {"role": "assistant", "content": REFUSAL}, "finish_reason": "stop"},
                {"index": 1, "message": {"role": "assistant", "content": "Short usable answer."}, "finish_reason": "stop"},
                {"index": 2, "message": {"role": "assistant", "content": "The longest usable completion was selected."}, "finish_reason": "stop"},
            ])
        return completion(REFUSAL)

    async with gateway_client(handler, settings_for(tmp_path)) as (client, _):
        response = await client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "Test."}]}, headers=AUTH
        )

    assert response.json()["choices"][0]["message"]["content"] == "The longest usable completion was selected."


@pytest.mark.asyncio
async def test_streaming_refusal_buffer_is_discarded_before_retry(tmp_path: Path, monkeypatch) -> None:
    set_provider_keys(monkeypatch)
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        log(seen, request)
        if len(seen) == 1:
            event = {"choices": [{"delta": {"content": "I cannot provide that refused fragment."}, "finish_reason": None}]}
            stream = f"data: {json.dumps(event)}\n\n".encode() + b"data: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream)
        return completion("Clean success after refusal.")

    async with gateway_client(handler, settings_for(tmp_path)) as (client, _):
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
async def test_json_body_on_streaming_request_is_not_read_as_empty(tmp_path: Path, monkeypatch) -> None:
    set_provider_keys(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "Plain JSON despite stream."}}]})

    async with gateway_client(handler, settings_for(tmp_path)) as (client, _):
        response = await client.post(
            "/v1/chat/completions",
            json={"stream": True, "messages": [{"role": "user", "content": "Test."}]},
            headers=AUTH,
        )

    assert response.status_code == 200
    assert response.headers["X-Sidestep-Attempts"] == "1"
    assert "Plain JSON despite stream." in response.text


# --- transport errors advance the ladder ------------------------------------


@pytest.mark.asyncio
async def test_http_400_advances_instead_of_terminating(tmp_path: Path, monkeypatch) -> None:
    set_provider_keys(monkeypatch)
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        log(seen, request)
        if provider_for(request) == "p1":
            return httpx.Response(400, json={"error": {"message": "response_format unsupported"}})
        return completion("Answered after a 400.")

    async with gateway_client(handler, settings_for(tmp_path)) as (client, _):
        response = await client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "Test."}]}, headers=AUTH
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Answered after a 400."
    assert response.headers["X-Sidestep-Attempts"] == "2"


@pytest.mark.asyncio
async def test_http_429_is_unavailable_and_advances(tmp_path: Path, monkeypatch) -> None:
    set_provider_keys(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        if provider_for(request) == "p1":
            return httpx.Response(429, json={"error": "slow down"})
        return completion("Answered after a 429.")

    async with gateway_client(handler, settings_for(tmp_path)) as (client, _):
        response = await client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "Test."}]}, headers=AUTH
        )

    assert response.status_code == 200
    assert response.headers["X-Sidestep-Attempts"] == "2"


@pytest.mark.asyncio
async def test_http_403_with_filter_marker_is_a_refusal(tmp_path: Path, monkeypatch) -> None:
    set_provider_keys(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        if provider_for(request) == "p1":
            return httpx.Response(403, json={"error": "blocked by policy"})
        return completion("Answered after a policy block.")

    async with gateway_client(handler, settings_for(tmp_path)) as (client, _):
        response = await client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "Test."}]}, headers=AUTH
        )

    assert response.status_code == 200
    assert response.headers["X-Sidestep-Attempts"] == "2"


# --- budgets ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_attempts_stops_after_cap(tmp_path: Path, monkeypatch) -> None:
    set_provider_keys(monkeypatch)
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        log(seen, request)
        return completion(REFUSAL)

    async with gateway_client(handler, settings_for(tmp_path, max_attempts=2)) as (client, _):
        response = await client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "Test."}]}, headers=AUTH
        )
        stats = (await client.get("/stats", headers=AUTH)).json()

    assert len(seen) == 2
    assert response.status_code == 502
    assert response.json()["error"] == "all_upstreams_refused"
    assert len(response.json()["attempts"]) == 2
    assert stats["budget_stops"] == 1


@pytest.mark.asyncio
async def test_max_attempts_of_one_makes_a_single_call(tmp_path: Path, monkeypatch) -> None:
    set_provider_keys(monkeypatch)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return completion(REFUSAL)

    async with gateway_client(handler, settings_for(tmp_path, max_attempts=1)) as (client, _):
        await client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "T"}]}, headers=AUTH)

    assert calls == 1


@pytest.mark.asyncio
async def test_max_cost_stops_after_cost_breach(tmp_path: Path, monkeypatch) -> None:
    set_provider_keys(monkeypatch)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return completion(REFUSAL, usage={"prompt_tokens": 1000, "completion_tokens": 0})

    providers = [make_provider("p1", "P1_KEY", 10, input_price=500), make_provider("p2", "P2_KEY", 20)]
    settings = settings_for(tmp_path, providers=providers, max_cost_usd=0.25)
    async with gateway_client(handler, settings) as (client, _):
        response = await client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "Test."}]}, headers=AUTH
        )
        stats = (await client.get("/stats", headers=AUTH)).json()

    assert calls == 1
    assert response.status_code == 502
    assert response.json()["error"] == "all_upstreams_refused"
    assert stats["budget_stops"] == 1
    assert response.json()["attempts"][0]["cost_usd"] == 0.5


@pytest.mark.asyncio
async def test_deadline_stops_loop_and_returns_best_decompose_answer(tmp_path: Path, monkeypatch) -> None:
    set_provider_keys(monkeypatch)
    sub_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sub_calls
        if mutation_for(request) != "decompose":
            return completion(REFUSAL)
        sub_calls += 1
        if sub_calls == 1:
            return completion("First decomposition answer is usable.")
        await asyncio.sleep(0.5)
        return completion("Second decomposition answer.")

    async with gateway_client(handler, settings_for(tmp_path, deadline_ms=200)) as (client, _):
        response = await client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "Test."}]}, headers=AUTH
        )
        stats = (await client.get("/stats", headers=AUTH)).json()

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "First decomposition answer is usable."
    assert stats["budget_stops"] == 1


# --- failure surfaces -------------------------------------------------------


@pytest.mark.asyncio
async def test_all_refused_returns_full_attempt_log(tmp_path: Path, monkeypatch) -> None:
    set_provider_keys(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        if body_for(request).get("n") == 3:
            return completion("", choices=[
                {"index": i, "message": {"role": "assistant", "content": REFUSAL}, "finish_reason": "stop"}
                for i in range(3)
            ])
        return completion(REFUSAL)

    async with gateway_client(handler, settings_for(tmp_path)) as (client, _):
        response = await client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "Test."}]}, headers=AUTH
        )
        refusals = await client.get("/refusals", headers=AUTH)
        stats = (await client.get("/stats", headers=AUTH)).json()

    assert refusals.json()["refusals"]
    assert response.status_code == 502
    data = response.json()
    assert data["error"] == "all_upstreams_refused"
    assert [entry["step"] for entry in data["attempts"]] == list(range(1, 9))
    assert [entry["provider"] for entry in data["attempts"][:3]] == ["p1", "p2", "p3"]
    assert all({"mutation", "provider", "rule"} <= entry.keys() for entry in data["attempts"])
    assert stats["provider_switches"] == 3
    assert response.headers["X-Sidestep-Verdict"] == "refusal"


@pytest.mark.asyncio
async def test_all_unavailable_reports_upstream_unavailable(tmp_path: Path, monkeypatch) -> None:
    set_provider_keys(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"})

    async with gateway_client(handler, settings_for(tmp_path)) as (client, _):
        response = await client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "Test."}]}, headers=AUTH
        )

    assert response.status_code == 502
    assert response.json()["error"] == "upstream_unavailable"
    assert response.headers["X-Sidestep-Verdict"] == "unavailable"


# --- modes and provider eligibility -----------------------------------------


@pytest.mark.asyncio
async def test_observe_only_env_forwards_refusal_without_retry(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SIDESTEP_OBSERVE_ONLY", "1")
    monkeypatch.setenv("P1_KEY", "test-p1-key")
    seen: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.content)
        return httpx.Response(
            200,
            content=b'{"choices":[{"message":{"content":"I cannot help with that."}}]}',
            headers={"content-type": "application/json"},
        )

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
    async with gateway_client(lambda r: completion("unused"), settings_for(tmp_path)) as (client, _):
        response = await client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "Test."}]}, headers=AUTH
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

    async with gateway_client(handler, settings_for(tmp_path)) as (client, _):
        response = await client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "Test."}]}, headers=AUTH
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
    assert settings.max_attempts >= 8
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
    assert settings.provider_name
    assert settings.upstream_base_url


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
        response = await client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "Test."}]}, headers=AUTH
        )

    assert response.status_code == 502
    assert response.json() == {"error": "no_provider_available", "attempts": []}
    assert response.headers["X-Sidestep-Verdict"] == "unavailable"
    assert called is False
