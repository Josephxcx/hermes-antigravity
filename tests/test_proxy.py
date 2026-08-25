"""Unit tests for the embedded Starlette proxy server."""

import pytest
import httpx
from hermes_antigravity.proxy.server import app


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
