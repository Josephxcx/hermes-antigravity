from __future__ import annotations

import json
import logging
import random
import time
from types import SimpleNamespace
from typing import Any, Dict, Iterator

import httpx

from hermes_antigravity.auth.credentials import load_credentials, save_credentials_to_file
from hermes_antigravity.auth.oauth import refresh_access_token
from hermes_antigravity.client.client import ENDPOINT_FALLBACKS, antigravity_headers, extract_project_id, default_project_id
from hermes_antigravity.stream.transformer import build_gemini_request, transform_google_sse_to_openai, aggregate_google_sse_to_openai_response

logger = logging.getLogger(__name__)

MAX_RETRIES_PER_ENDPOINT = 5
BASE_BACKOFF_SECS = 2.0
MAX_BACKOFF_SECS = 30.0

def _backoff_delay(attempt: int, status_code: int = 0) -> float:
    base = BASE_BACKOFF_SECS * (2 ** attempt)
    if status_code == 429:
        base = max(base, 5.0)
    capped = min(base, MAX_BACKOFF_SECS)
    jitter = capped * random.uniform(-0.25, 0.25)
    return max(0.5, capped + jitter)

def is_retryable_status(status_code: int, error_text: str = "") -> bool:
    if status_code in (429, 500, 502, 503, 504):
        return True
    if status_code == 400:
        upper = error_text.upper()
        if "RESOURCE_EXHAUSTED" in upper or "QUOTA" in upper or "RATE_LIMIT" in upper:
            return True
    return False

def _sync_list_cloud_ai_companion_projects(token: str) -> str | None:
    for endpoint in ENDPOINT_FALLBACKS:
        try:
            with httpx.Client(timeout=8.0) as client:
                res = client.post(
                    f"{endpoint}/v1internal:listCloudAICompanionProjects",
                    headers=antigravity_headers(token),
                    json={},
                )
                if res.status_code == 200:
                    extracted = extract_project_id(res.json())
                    if extracted:
                        return extracted
        except Exception as e:
            logger.debug("Failed sync listCloudAICompanionProjects on %s: %s", endpoint, e)
    return None

def _sync_resolve_project_id(token: str, seed: str = "antigravity-default") -> str:
    import os
    explicit = os.environ.get("ANTIGRAVITY_PROJECT_ID")
    if explicit and explicit.strip():
        return explicit.strip()

    discovered = _sync_list_cloud_ai_companion_projects(token)
    if discovered:
        return discovered

    return default_project_id(seed)


class AntigravityClient:
    """Minimal synchronous facade for Hermes Agent targeting Google Cloud Code Assist."""
    
    HERMES_SKIP_TRANSPORT_WRAP = True

    def __init__(self, **kwargs: Any):
        self.api_key = kwargs.get("api_key", "")
        self.base_url = kwargs.get("base_url", ENDPOINT_FALLBACKS[0])
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_chat_completion))
        self._http_client = httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0))
        
    def _create_chat_completion(self, **kwargs: Any) -> Any:
        try:
            from hermes_cli.auth import AuthError
        except ImportError:
            class AuthError(Exception):
                def __init__(self, msg, provider=None, code=None, relogin_required=False):
                    super().__init__(msg)
                    self.provider = provider
                    self.code = code
                    self.relogin_required = relogin_required
        # 1. Load credentials (to have refresh token access)
        creds = load_credentials()
        if not creds:
            raise AuthError("No Antigravity credentials found. Please run '/login antigravity'.", provider="antigravity", code="missing_credentials")
            
        is_stream = bool(kwargs.get("stream", False))
        
        # 2. Build Gemini envelope
        project_id = creds.project_id or _sync_resolve_project_id(creds.access_token, seed=creds.email or "antigravity-default")
        
        # 3. Dynamic model resolution
        from hermes_antigravity.client.catalog import resolve_runtime_model_sync
        from hermes_antigravity.models.models import get_runtime_model_id
        
        raw_model = kwargs.get("model", "gemini-3.1-pro")
        reasoning_effort = kwargs.get("reasoning_effort") or kwargs.get("reasoning", {}).get("effort")
        initial_runtime_model = get_runtime_model_id(raw_model, reasoning_effort)
        
        # Discover actual runtime model from Google API
        runtime_model = resolve_runtime_model_sync(creds.access_token, project_id, initial_runtime_model)
        
        # Override the kwargs model so transformer uses the resolved one
        kwargs["model"] = runtime_model
        
        _, envelope = build_gemini_request(kwargs, project_id)
        
        headers = antigravity_headers(creds.access_token)
        if kwargs.get("model", "").lower().startswith("claude-"):
            headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

        last_status = 500
        last_err_text = ""
        refreshed_on_401 = False

        # Attempt round-trips
        for endpoint in ENDPOINT_FALLBACKS:
            url = f"{endpoint}/v1internal:streamGenerateContent?alt=sse"
            attempt = 0
            
            while attempt < MAX_RETRIES_PER_ENDPOINT:
                try:
                    req = self._http_client.build_request("POST", url, headers=headers, json=envelope)
                    resp = self._http_client.send(req, stream=True)
                    
                    if resp.status_code == 200:
                        def _line_iterator() -> Iterator[str]:
                            for line in resp.iter_lines():
                                if line:
                                    yield line

                        if is_stream:
                            return self._stream_response(_line_iterator(), runtime_model, resp)
                        else:
                            return self._non_stream_response(_line_iterator(), runtime_model, resp)
                            
                    else:
                        resp.read()
                        last_status = resp.status_code
                        last_err_text = resp.text
                        resp.close()

                        # 401 Refresh Logic
                        if last_status == 401 and creds.refresh_token and not refreshed_on_401:
                            logger.info("Upstream returned 401 Unauthorized; refreshing token synchronously...")
                            try:
                                import asyncio
                                try:
                                    loop = asyncio.get_event_loop()
                                    if loop.is_running():
                                        raise RuntimeError("Event loop is running")
                                    creds = loop.run_until_complete(refresh_access_token(creds))
                                except RuntimeError:
                                    import threading
                                    def _run_refresh():
                                        new_loop = asyncio.new_event_loop()
                                        return new_loop.run_until_complete(refresh_access_token(creds))
                                    def _thread_target():
                                        try:
                                            res = _run_refresh()
                                            setattr(threading.current_thread(), "res", res)
                                        except Exception as e:
                                            setattr(threading.current_thread(), "err", e)
                                            
                                    t = threading.Thread(target=_thread_target)
                                    t.start()
                                    t.join()
                                    if hasattr(t, "err"):
                                        raise getattr(t, "err")
                                    creds = getattr(t, "res")

                                save_credentials_to_file(creds)
                                headers = antigravity_headers(creds.access_token)
                                if kwargs.get("model", "").lower().startswith("claude-"):
                                    headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"
                                refreshed_on_401 = True
                                continue # retry immediately
                            except Exception as refresh_err:
                                logger.error(f"Failed to refresh token: {refresh_err}")
                                raise AuthError("Antigravity token expired and refresh failed. Please run '/login antigravity'.", provider="antigravity", code="refresh_failed", relogin_required=True) from refresh_err
                        
                        if not is_retryable_status(last_status, last_err_text):
                            break
                        
                        if attempt < MAX_RETRIES_PER_ENDPOINT - 1:
                            time.sleep(_backoff_delay(attempt, last_status))

                except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as net_err:
                    if attempt < MAX_RETRIES_PER_ENDPOINT - 1:
                        time.sleep(_backoff_delay(attempt))
                
                attempt += 1

        # If we exhausted attempts or broke on non-retryable error
        if last_status in (401, 403):
            raise AuthError(f"Google Cloud Code Assist authentication error {last_status}: {last_err_text}", provider="antigravity", code="api_auth_error", relogin_required=True)
            
        raise RuntimeError(f"Antigravity API Error {last_status}: {last_err_text}")
        
    def _dict_to_ns(self, d: Any) -> Any:
        if isinstance(d, dict):
            return SimpleNamespace(**{k: self._dict_to_ns(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [self._dict_to_ns(i) for i in d]
        return d
        
    def _stream_response(self, line_iterator: Iterator[str], runtime_model: str, resp_obj: httpx.Response) -> Iterator[Any]:
        try:
            for chunk_str in transform_google_sse_to_openai(line_iterator, runtime_model):
                if chunk_str.startswith("data: ") and chunk_str.strip() != "data: [DONE]":
                    json_str = chunk_str[5:].strip()
                    try:
                        chunk_dict = json.loads(json_str)
                        yield self._dict_to_ns(chunk_dict)
                    except Exception:
                        pass
        finally:
            resp_obj.close()
            
    def _non_stream_response(self, line_iterator: Iterator[str], runtime_model: str, resp_obj: httpx.Response) -> Any:
        try:
            resp_dict = aggregate_google_sse_to_openai_response(line_iterator, runtime_model)
            return self._dict_to_ns(resp_dict)
        finally:
            resp_obj.close()
