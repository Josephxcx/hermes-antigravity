"""Unit tests for the embedded Starlette proxy server, connection pooling, and 401 refresh retry."""

import json
from unittest.mock import MagicMock, patch
import httpx
import pytest

from hermes_antigravity.auth.credentials import AntigravityCredentials
from hermes_antigravity.proxy.server import (
    app,
    close_proxy_client,
    get_proxy_client,
    is_retryable_status,
)


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
async def test_proxy_client_connection_pooling():
    client1 = get_proxy_client()
    client2 = get_proxy_client()
    assert client1 is client2
    assert not client1.is_closed

    await close_proxy_client()
    assert client1.is_closed

    client3 = get_proxy_client()
    assert client3 is not client1
    assert not client3.is_closed
    await close_proxy_client()


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


@pytest.mark.asyncio
async def test_proxy_non_stream_401_in_flight_refresh_retry():
    creds_initial = AntigravityCredentials(
        access_token="stale-token",
        refresh_token="valid-refresh-token",
        expires_at=9999999999999,
        email="dev@example.com",
        project_id="test-proj",
    )
    creds_refreshed = AntigravityCredentials(
        access_token="fresh-token",
        refresh_token="valid-refresh-token",
        expires_at=9999999999999,
        email="dev@example.com",
        project_id="test-proj",
    )

    sample_google_sse = (
        'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Hello world"}]}, "finishReason": "STOP"}]}}\n\n'
    )

    request_count = 0

    class MockStreamContext:
        def __init__(self, status_code: int, content: str):
            self.status_code = status_code
            self._content = content

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def aiter_lines(self):
            for line in self._content.splitlines():
                yield line

        async def aread(self):
            return self._content.encode("utf-8")

    def mock_stream(method, url, headers=None, json=None):
        nonlocal request_count
        request_count += 1
        auth = (headers or {}).get("Authorization", "")
        if auth == "Bearer stale-token":
            return MockStreamContext(401, '{"error": {"message": "Invalid token"}}')
        return MockStreamContext(200, sample_google_sse)

    mock_client = MagicMock()
    mock_client.is_closed = False
    mock_client.stream.side_effect = mock_stream

    with patch("hermes_antigravity.proxy.server.ensure_valid_credentials", return_value=creds_initial):
        with patch("hermes_antigravity.proxy.server.refresh_access_token", return_value=creds_refreshed) as mock_refresh:
            with patch("hermes_antigravity.proxy.server.get_proxy_client", return_value=mock_client):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                    resp = await client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "gemini-3.7-flash",
                            "messages": [{"role": "user", "content": "Hi"}],
                            "stream": False,
                        },
                    )
                    assert resp.status_code == 200
                    data = resp.json()
                    assert data["choices"][0]["message"]["content"] == "Hello world"
                    assert mock_refresh.call_count == 1
                    assert request_count == 2


@pytest.mark.asyncio
async def test_proxy_streaming_401_in_flight_refresh_retry():
    creds_initial = AntigravityCredentials(
        access_token="stale-token",
        refresh_token="valid-refresh-token",
        expires_at=9999999999999,
        email="dev@example.com",
        project_id="test-proj",
    )
    creds_refreshed = AntigravityCredentials(
        access_token="fresh-token",
        refresh_token="valid-refresh-token",
        expires_at=9999999999999,
        email="dev@example.com",
        project_id="test-proj",
    )

    sample_google_sse = (
        'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Streamed text"}]}, "finishReason": "STOP"}]}}\n\n'
    )

    request_count = 0

    class MockStreamContext:
        def __init__(self, status_code: int, content: str):
            self.status_code = status_code
            self._content = content

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def aiter_lines(self):
            for line in self._content.splitlines():
                yield line

        async def aread(self):
            return self._content.encode("utf-8")

    def mock_stream(method, url, headers=None, json=None):
        nonlocal request_count
        request_count += 1
        auth = (headers or {}).get("Authorization", "")
        if auth == "Bearer stale-token":
            return MockStreamContext(401, '{"error": {"message": "Invalid credentials"}}')
        return MockStreamContext(200, sample_google_sse)

    mock_client = MagicMock()
    mock_client.is_closed = False
    mock_client.stream.side_effect = mock_stream

    with patch("hermes_antigravity.proxy.server.ensure_valid_credentials", return_value=creds_initial):
        with patch("hermes_antigravity.proxy.server.refresh_access_token", return_value=creds_refreshed) as mock_refresh:
            with patch("hermes_antigravity.proxy.server.get_proxy_client", return_value=mock_client):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                    resp = await client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "gemini-3.7-flash",
                            "messages": [{"role": "user", "content": "Hi"}],
                            "stream": True,
                        },
                    )
                    assert resp.status_code == 200
                    chunks = [line async for line in resp.aiter_lines() if line.startswith("data: ")]
                    assert any("Streamed text" in c for c in chunks)
                    assert mock_refresh.call_count == 1
                    assert request_count == 2
