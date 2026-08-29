"""Unit tests for the embedded Starlette proxy server."""

import pytest
import httpx
from hermes_antigravity.proxy.server import app, is_retryable_status


def test_is_retryable_status():
    assert is_retryable_status(429) is True
    assert is_retryable_status(500) is True
    assert is_retryable_status(502) is True
    assert is_retryable_status(503) is True
    assert is_retryable_status(504) is True
    assert is_retryable_status(400, "RESOURCE_EXHAUSTED: quota exceeded") is True
    assert is_retryable_status(400, "RATE_LIMIT_EXCEEDED") is True
    assert is_retryable_status(400, "INVALID_ARGUMENT: malformed json") is False
    assert is_retryable_status(401, "UNAUTHENTICATED") is False
    assert is_retryable_status(200, "OK") is False


@pytest.mark.asyncio
async def test_proxy_health_and_models():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Health check
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["provider"] == "antigravity"

        # 2. Models list
        resp = await client.get("/v1/models")
        assert resp.status_code == 200
        models_data = resp.json()
        assert models_data["object"] == "list"
        ids = [m["id"] for m in models_data["data"]]
        assert "gemini-3.7-flash" in ids or "antigravity/gemini-3.7-flash" in ids


@pytest.mark.asyncio
async def test_proxy_invalid_json_body():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post(
            "/v1/chat/completions",
            content="not a json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400
        assert "Invalid JSON" in resp.json().get("error", "")

