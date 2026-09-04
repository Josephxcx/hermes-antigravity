"""Embedded in-process OpenAI-compatible proxy server for Antigravity using Starlette and Uvicorn."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time
from typing import AsyncGenerator, Optional

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from hermes_antigravity.auth.oauth import (
    ensure_valid_credentials,
    refresh_access_token,
)
from hermes_antigravity.client.client import (
    ENDPOINT_FALLBACKS,
    antigravity_headers,
    resolve_project_id,
)
from hermes_antigravity.models.models import FALLBACK_MODELS
from hermes_antigravity.stream.transformer import (
    aggregate_google_sse_to_openai_response,
    build_gemini_request,
    transform_google_sse_to_openai,
)

logger = logging.getLogger(__name__)
DEFAULT_PROXY_PORT = 51122

MAX_RETRIES_PER_ENDPOINT = 3
BASE_BACKOFF_SECS = 1.0

_shared_client: Optional[httpx.AsyncClient] = None


def get_proxy_client() -> httpx.AsyncClient:
    """Returns or creates the shared persistent httpx AsyncClient with connection pooling."""
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        limits = httpx.Limits(
            max_keepalive_connections=20,
            max_connections=50,
            keepalive_expiry=30.0,
        )
        _shared_client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=10.0),
            limits=limits,
        )
    return _shared_client


async def close_proxy_client() -> None:
    """Closes the shared persistent httpx AsyncClient."""
    global _shared_client
    if _shared_client is not None and not _shared_client.is_closed:
        await _shared_client.aclose()
        _shared_client = None


def is_retryable_status(status_code: int, error_text: str = "") -> bool:
    """Determines if a Google Cloud Code Assist error response is transient/retryable."""
    if status_code in (429, 500, 502, 503, 504):
        return True
    if status_code == 400:
        upper = error_text.upper()
        if "RESOURCE_EXHAUSTED" in upper or "QUOTA" in upper or "RATE_LIMIT" in upper:
            return True
    return False


async def handle_health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "healthy", "provider": "antigravity"})


async def handle_models(request: Request) -> JSONResponse:
    models_data = [
        {
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "antigravity",
        }
        for model_id in FALLBACK_MODELS
    ]
    return JSONResponse({"object": "list", "data": models_data})


async def _extract_lines(resp: httpx.Response) -> AsyncGenerator[str, None]:
    async for line in resp.aiter_lines():
        if line:
            yield line


async def handle_chat_completions(request: Request) -> JSONResponse | StreamingResponse:
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

    is_stream = bool(body.get("stream", False))
    model_name = body.get("model", "gemini-3.7-flash")
    client = get_proxy_client()

    # Handle Non-Streaming requests
    if not is_stream:
        last_status = 500
        last_err_text = ""
        refreshed_on_401 = False

        try:
            for endpoint in ENDPOINT_FALLBACKS:
                url = f"{endpoint}/v1internal:streamGenerateContent?alt=sse"
                attempt = 0
                while attempt < MAX_RETRIES_PER_ENDPOINT:
                    try:
                        async with client.stream("POST", url, headers=headers, json=envelope) as g_resp:
                            if g_resp.status_code == 200:
                                openai_response = await aggregate_google_sse_to_openai_response(
                                    _extract_lines(g_resp), model_name
                                )
                                return JSONResponse(openai_response)

                            last_status = g_resp.status_code
                            err_bytes = await g_resp.aread()
                            last_err_text = err_bytes.decode("utf-8", errors="replace")
                            logger.warning(
                                "Non-stream endpoint %s attempt %d failed (%d): %s",
                                endpoint,
                                attempt + 1,
                                last_status,
                                last_err_text,
                            )

                            # Handle in-flight 401 token refresh retry
                            if last_status == 401 and creds.refresh_token and not refreshed_on_401:
                                logger.info("Upstream returned 401 Unauthorized; attempting in-flight token refresh...")
                                try:
                                    creds = await refresh_access_token(creds)
                                    headers = antigravity_headers(creds.access_token)
                                    if body.get("model", "").lower().startswith("claude-"):
                                        headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"
                                    refreshed_on_401 = True
                                    continue
                                except Exception as refresh_err:
                                    logger.warning("In-flight token refresh failed: %s", refresh_err)

                            if not is_retryable_status(last_status, last_err_text):
                                break
                            if attempt < MAX_RETRIES_PER_ENDPOINT - 1:
                                await asyncio.sleep(BASE_BACKOFF_SECS * (2**attempt))
                    except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as net_err:
                        logger.warning(
                            "Network error in non-stream %s (attempt %d): %s",
                            endpoint,
                            attempt + 1,
                            net_err,
                        )
                        if attempt < MAX_RETRIES_PER_ENDPOINT - 1:
                            await asyncio.sleep(BASE_BACKOFF_SECS * (2**attempt))
                    attempt += 1

            return JSONResponse(
                {
                    "error": {
                        "message": f"Google Cloud Code Assist error ({last_status}): {last_err_text}",
                        "type": "antigravity_api_error",
                        "code": last_status,
                    }
                },
                status_code=last_status if last_status in (400, 401, 403, 404, 429) else 502,
            )
        except Exception as e:
            logger.error("Error during non-streaming chat completion: %s", e)
            return JSONResponse(
                {"error": {"message": str(e), "type": "internal_proxy_error"}},
                status_code=500,
            )

    # Handle Streaming requests
    async def sse_generator():
        success = False
        last_err_text = ""
        last_status = 500
        refreshed_on_401 = False
        current_headers = dict(headers)
        current_creds = creds

        try:
            for endpoint in ENDPOINT_FALLBACKS:
                if success:
                    break
                url = f"{endpoint}/v1internal:streamGenerateContent?alt=sse"
                attempt = 0

                while attempt < MAX_RETRIES_PER_ENDPOINT:
                    try:
                        async with client.stream("POST", url, headers=current_headers, json=envelope) as g_resp:
                            if g_resp.status_code == 200:
                                success = True
                                async for chunk in transform_google_sse_to_openai(_extract_lines(g_resp), model_name):
                                    yield chunk.encode("utf-8")
                                break

                            last_status = g_resp.status_code
                            err_bytes = await g_resp.aread()
                            last_err_text = err_bytes.decode("utf-8", errors="replace")
                            logger.warning(
                                "Endpoint %s attempt %d failed (%d): %s",
                                endpoint,
                                attempt + 1,
                                last_status,
                                last_err_text,
                            )

                            # Handle in-flight 401 token refresh retry
                            if last_status == 401 and current_creds.refresh_token and not refreshed_on_401:
                                logger.info("Upstream streaming returned 401; attempting in-flight token refresh...")
                                try:
                                    current_creds = await refresh_access_token(current_creds)
                                    current_headers = antigravity_headers(current_creds.access_token)
                                    if body.get("model", "").lower().startswith("claude-"):
                                        current_headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"
                                    refreshed_on_401 = True
                                    continue
                                except Exception as refresh_err:
                                    logger.warning("In-flight streaming token refresh failed: %s", refresh_err)

                            if not is_retryable_status(last_status, last_err_text):
                                break

                            if attempt < MAX_RETRIES_PER_ENDPOINT - 1:
                                backoff = BASE_BACKOFF_SECS * (2**attempt)
                                await asyncio.sleep(backoff)
                    except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as net_err:
                        logger.warning(
                            "Network error connecting to %s (attempt %d): %s",
                            endpoint,
                            attempt + 1,
                            net_err,
                        )
                        if attempt < MAX_RETRIES_PER_ENDPOINT - 1:
                            await asyncio.sleep(BASE_BACKOFF_SECS * (2**attempt))
                    attempt += 1

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


@contextlib.asynccontextmanager
async def lifespan(app_instance: Starlette):
    yield
    await close_proxy_client()


routes = [
    Route("/health", handle_health, methods=["GET"]),
    Route("/v1/models", handle_models, methods=["GET"]),
    Route("/v1/chat/completions", handle_chat_completions, methods=["POST"]),
]

app = Starlette(debug=False, routes=routes, lifespan=lifespan)


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
