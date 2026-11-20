import json
import os

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"ok": "true"}


def _content(body: dict[str, object], api_key: str) -> str:
    if body.get("audit_demo") == "refusal":
        return "I'm sorry, I can't help with that request. I can offer safe, high-level information instead."
    if body.get("audit_demo") == "route_to_secondary":
        if api_key == os.getenv("MOCK_UPSTREAM_KEY", "local-mock-key"):
            return "I'm sorry, I can't help with that request. I can offer safe, high-level information instead."
        if api_key == os.getenv("MOCK_SECONDARY_KEY", "local-secondary-key"):
            return "Mock secondary provider succeeded. No external model was contacted."
    if body.get("temperature") == 0 and body.get("n") == 3:
        return "Longest non-refusing mock completion selected. No external model was contacted."
    if body.get("response_format", {}).get("type") == "json_schema":
        return json.dumps({"answer": "Mock structured answer.", "notes": "Mock notes."})
    return "This is a mock response from the local test provider. No external model was contacted."


@app.get("/v1/models")
async def models() -> dict[str, object]:
    return {"object": "list", "data": [{"id": "mock-model", "object": "model", "owned_by": "sidestep-test"}]}


@app.post("/v1/chat/completions")
async def completions(request: Request):
    body = await request.json()
    authorization = request.headers.get("authorization", "")
    api_key = authorization[7:] if authorization.startswith("Bearer ") else ""
    content = _content(body, api_key)
    model = body.get("model", "mock-model")
    if body.get("stream") is True:
        async def events():
            words = content.split(" ")
            for index, word in enumerate(words):
                text = word if index == 0 else f" {word}"
                event = {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "model": model,
                         "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
                yield f"data: {json.dumps(event)}\n\n".encode()
            final = {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "model": model,
                     "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(final)}\n\n".encode()
            yield b"data: [DONE]\n\n"
        return StreamingResponse(events(), media_type="text/event-stream")
    choices = [
        {"index": 0, "message": {"role": "assistant", "content": "I'm sorry, I cannot answer."}, "finish_reason": "stop"},
        {"index": 1, "message": {"role": "assistant", "content": "A short mock answer."}, "finish_reason": "stop"},
        {"index": 2, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"},
    ] if body.get("temperature") == 0 and body.get("n") == 3 else [
        {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
    ]
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "model": model,
        "choices": choices,
    }
