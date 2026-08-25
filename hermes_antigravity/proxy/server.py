"""Embedded in-process OpenAI-compatible proxy server for Antigravity using Starlette and Uvicorn."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
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
DEFAULT_PROXY_PORT = 51122


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


class BackgroundServer:
    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PROXY_PORT):
        self.host = host
        self.port = port
        self.server: Optional[uvicorn.Server] = None
        self.thread: Optional[threading.Thread] = None

    def start(self) -> str:
        if self.thread and self.thread.is_alive():
            return f"http://{self.host}:{self.port}/v1"

        # Check if another process or thread is already serving port 51122
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            if s.connect_ex((self.host, self.port)) == 0:
                # Already listening
                return f"http://{self.host}:{self.port}/v1"

        config = uvicorn.Config(
            app=app,
            host=self.host,
            port=self.port,
            log_level="error",
            access_log=False,
        )
        self.server = uvicorn.Server(config)

        def run():
            self.server.run()

        self.thread = threading.Thread(target=run, daemon=True, name="AntigravityProxy")
        self.thread.start()

        # Wait briefly for startup
        for _ in range(30):
            if self.server.started:
                break
            time.sleep(0.05)

        base_url = f"http://{self.host}:{self.port}/v1"
        logger.info("Antigravity in-process proxy started in background on %s", base_url)
        return base_url


_bg_server: Optional[BackgroundServer] = None


def ensure_proxy_running(port: int = DEFAULT_PROXY_PORT) -> str:
    global _bg_server
    if _bg_server is None:
        _bg_server = BackgroundServer(port=port)
    return _bg_server.start()


# Auto-start proxy on default port when module is imported
try:
    ensure_proxy_running()
except Exception as e:
    logger.debug("Failed to auto-start proxy on import: %s", e)
