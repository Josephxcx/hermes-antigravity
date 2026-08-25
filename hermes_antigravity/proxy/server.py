"""Embedded in-process OpenAI-compatible proxy server for Antigravity."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

import httpx
from aiohttp import web

from hermes_antigravity.auth.oauth import ensure_valid_credentials
from hermes_antigravity.client.client import (
    DEFAULT_ENDPOINT,
    ENDPOINT_FALLBACKS,
    antigravity_headers,
    resolve_project_id,
)
from hermes_antigravity.models.models import FALLBACK_MODELS
from hermes_antigravity.stream.transformer import (
    build_gemini_request,
    transform_google_sse_to_openai,
)

logger = logging.getLogger(__name__)


class AntigravityProxyServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.host = host
        self.requested_port = port
        self.actual_port: Optional[int] = None
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self._setup_routes()

    def _setup_routes(self) -> None:
        self.app.router.add_get("/health", self.handle_health)
        self.app.router.add_get("/v1/models", self.handle_models)
        self.app.router.add_post("/v1/chat/completions", self.handle_chat_completions)

    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "healthy", "provider": "antigravity"})

    async def handle_models(self, request: web.Request) -> web.Response:
        models_data = [
            {
                "id": f"antigravity/{model_id}",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "antigravity",
            }
            for model_id in FALLBACK_MODELS
        ]
        # Also include without antigravity/ prefix for direct compatibility
        for model_id in FALLBACK_MODELS:
            models_data.append({
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "antigravity",
            })
        return web.json_response({"object": "list", "data": models_data})

    async def handle_chat_completions(self, request: web.Request) -> web.StreamResponse:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)

        try:
            creds = await ensure_valid_credentials()
        except Exception as e:
            logger.error("Authentication error in proxy: %s", e)
            return web.json_response({"error": str(e)}, status=401)

        project_id = creds.project_id or await resolve_project_id(
            creds.access_token, seed=creds.email or "antigravity-default"
        )
        runtime_model, envelope = build_gemini_request(body, project_id)

        headers = antigravity_headers(creds.access_token)
        if body.get("model", "").lower().startswith("claude-"):
            headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

        # Prepare SSE streaming response back to client
        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(request)

        async def line_generator(httpx_response: httpx.Response):
            async for line in httpx_response.aiter_lines():
                if line:
                    yield line

        try:
            model_name = body.get("model", "gemini-3.7-flash")
            success = False
            last_err_text = ""
            last_status = 500

            async with httpx.AsyncClient(timeout=120.0) as client:
                for endpoint in ENDPOINT_FALLBACKS:
                    url = f"{endpoint}/v1internal:streamGenerateContent?alt=sse"
                    async with client.stream("POST", url, headers=headers, json=envelope) as g_resp:
                        if g_resp.status_code == 200:
                            success = True
                            async for chunk in transform_google_sse_to_openai(line_generator(g_resp), model_name):
                                await response.write(chunk.encode("utf-8"))
                            break
                        else:
                            last_status = g_resp.status_code
                            err_bytes = await g_resp.aread()
                            last_err_text = err_bytes.decode("utf-8", errors="replace")
                            logger.warning("Endpoint %s failed (%d): %s", endpoint, last_status, last_err_text)
                            if last_status not in (403, 404, 429, 500, 502, 503, 504):
                                break

            if not success:
                err_chunk = {
                    "error": {
                        "message": f"Google Cloud Code Assist error ({last_status}): {last_err_text}",
                        "type": "antigravity_api_error",
                        "code": last_status,
                    }
                }
                await response.write(f"data: {json.dumps(err_chunk)}\n\n".encode("utf-8"))
                await response.write(b"data: [DONE]\n\n")

        except Exception as e:
            logger.error("Error during streaming generation: %s", e)
            err_chunk = {"error": {"message": str(e), "type": "internal_proxy_error"}}
            await response.write(f"data: {json.dumps(err_chunk)}\n\n".encode("utf-8"))
            await response.write(b"data: [DONE]\n\n")

        return response

    async def start(self) -> str:
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.requested_port)
        await self.site.start()
        # Retrieve allocated port
        sockets = self.site._server.sockets
        if sockets:
            self.actual_port = sockets[0].getsockname()[1]
        else:
            self.actual_port = self.requested_port
        base_url = f"http://{self.host}:{self.actual_port}/v1"
        logger.info("Antigravity in-process proxy started at %s", base_url)
        return base_url

    async def stop(self) -> None:
        if self.runner:
            await self.runner.cleanup()
            self.runner = None
            self.site = None


_global_proxy: Optional[AntigravityProxyServer] = None
_proxy_base_url: Optional[str] = None
_proxy_lock = asyncio.Lock()


async def get_or_start_proxy() -> str:
    """Returns the base_url of the running embedded proxy, starting it if necessary."""
    global _global_proxy, _proxy_base_url
    async with _proxy_lock:
        if _global_proxy and _proxy_base_url:
            return _proxy_base_url
        _global_proxy = AntigravityProxyServer()
        _proxy_base_url = await _global_proxy.start()
        return _proxy_base_url
