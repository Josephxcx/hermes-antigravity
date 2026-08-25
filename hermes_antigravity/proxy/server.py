"""Embedded in-process OpenAI-compatible proxy server for Antigravity using Starlette and Uvicorn."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from hermes_antigravity.auth.oauth import ensure_valid_credentials
from hermes_antigravity.client.client import (
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


async def handle_health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "healthy", "provider": "antigravity"})


async def handle_models(request: Request) -> JSONResponse:
    models_data = [
        {
            "id": f"antigravity/{model_id}",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "antigravity",
        }
        for model_id in FALLBACK_MODELS
    ]
    for model_id in FALLBACK_MODELS:
        models_data.append({
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "antigravity",
        })
    return JSONResponse({"object": "list", "data": models_data})


async def handle_chat_completions(request: Request) -> StreamingResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    try:
        creds = await ensure_valid_credentials()
    except Exception as e:
        logger.error("Authentication error in proxy: %s", e)
        return JSONResponse({"error": str(e)}, status_code=401)

    project_id = creds.project_id or await resolve_project_id(
        creds.access_token, seed=creds.email or "antigravity-default"
    )
    runtime_model, envelope = build_gemini_request(body, project_id)

    headers = antigravity_headers(creds.access_token)
    if body.get("model", "").lower().startswith("claude-"):
        headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

    async def sse_generator():
        async def line_generator(httpx_response: httpx.Response):
            async for line in httpx_response.aiter_lines():
                if line:
                    yield line

        model_name = body.get("model", "gemini-3.7-flash")
        success = False
        last_err_text = ""
        last_status = 500

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                for endpoint in ENDPOINT_FALLBACKS:
                    url = f"{endpoint}/v1internal:streamGenerateContent?alt=sse"
                    async with client.stream("POST", url, headers=headers, json=envelope) as g_resp:
                        if g_resp.status_code == 200:
                            success = True
                            async for chunk in transform_google_sse_to_openai(line_generator(g_resp), model_name):
                                yield chunk.encode("utf-8")
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
                yield f"data: {json.dumps(err_chunk)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"

        except Exception as e:
            logger.error("Error during streaming generation: %s", e)
            err_chunk = {"error": {"message": str(e), "type": "internal_proxy_error"}}
            yield f"data: {json.dumps(err_chunk)}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


routes = [
    Route("/health", handle_health, methods=["GET"]),
    Route("/v1/models", handle_models, methods=["GET"]),
    Route("/v1/chat/completions", handle_chat_completions, methods=["POST"]),
]

app = Starlette(debug=False, routes=routes)


class AntigravityProxyServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.host = host
        self.requested_port = port
        self.actual_port: Optional[int] = None
        self.server: Optional[uvicorn.Server] = None
        self.task: Optional[asyncio.Task] = None

    async def start(self) -> str:
        config = uvicorn.Config(
            app=app,
            host=self.host,
            port=self.requested_port,
            log_level="warning",
            access_log=False,
        )
        self.server = uvicorn.Server(config)

        # Start server in background asyncio task
        self.task = asyncio.create_task(self.server.serve())

        # Wait until started and bound to port
        for _ in range(50):
            if self.server.started:
                break
            await asyncio.sleep(0.05)

        # Retrieve bound socket
        for server in getattr(self.server, "servers", []):
            for socket in getattr(server, "sockets", []):
                self.actual_port = socket.getsockname()[1]
                break
            if self.actual_port:
                break

        if not self.actual_port:
            self.actual_port = self.requested_port or 51122

        base_url = f"http://{self.host}:{self.actual_port}/v1"
        logger.info("Antigravity in-process proxy started at %s", base_url)
        return base_url

    async def stop(self) -> None:
        if self.server:
            self.server.should_exit = True
            if self.task:
                await self.task
            self.server = None
            self.task = None


_global_proxy: Optional[AntigravityProxyServer] = None
_proxy_base_url: Optional[str] = None
_proxy_lock = asyncio.Lock()


async def get_or_start_proxy() -> str:
    global _global_proxy, _proxy_base_url
    async with _proxy_lock:
        if _global_proxy and _proxy_base_url:
            return _proxy_base_url
        _global_proxy = AntigravityProxyServer()
        _proxy_base_url = await _global_proxy.start()
        return _proxy_base_url
