"""Unit tests for the embedded in-process proxy server."""

import pytest
from aiohttp import web
from hermes_antigravity.proxy.server import AntigravityProxyServer


@pytest.mark.asyncio
async def test_proxy_health_and_models(aiohttp_client):
    proxy = AntigravityProxyServer(port=0)
    client = await aiohttp_client(proxy.app)

    # 1. Health check
    resp = await client.get("/health")
    assert resp.status == 200
    data = await resp.json()
    assert data["status"] == "healthy"
    assert data["provider"] == "antigravity"

    # 2. Models list
    resp = await client.get("/v1/models")
    assert resp.status == 200
    models_data = await resp.json()
    assert models_data["object"] == "list"
    ids = [m["id"] for m in models_data["data"]]
    assert "gemini-3.7-flash" in ids or "antigravity/gemini-3.7-flash" in ids
