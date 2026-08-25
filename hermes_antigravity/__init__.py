"""Hermes Agent plugin for Google Antigravity / Cloud Code Assist."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional, Tuple

from hermes_antigravity.auth.credentials import load_credentials
from hermes_antigravity.auth.oauth import (
    build_authorization_url,
    exchange_code_for_tokens,
    generate_pkce,
    parse_pasted_callback,
)
from hermes_antigravity.models.models import (
    FALLBACK_MODELS,
    PROVIDER_ID,
    PROVIDER_NAME,
)
from hermes_antigravity.proxy.server import (
    DEFAULT_PROXY_PORT,
    ensure_proxy_running,
)
from hermes_antigravity.usage.usage import format_quota_report

logger = logging.getLogger(__name__)

# Attempt to import Hermes base classes if running inside Hermes
try:
    from providers import register_provider
    from providers.base import ProviderProfile
except ImportError:
    # Standalone mode fallback for testing outside Hermes runtime
    class ProviderProfile:  # type: ignore
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    def register_provider(profile):  # type: ignore
        pass


class AntigravityProviderProfile(ProviderProfile):
    """Hermes ProviderProfile for Antigravity."""

    def build_api_kwargs_extras(
        self,
        *,
        model: Optional[str] = None,
        reasoning_config: Optional[Dict[str, Any]] = None,
        supports_reasoning: bool = False,
        **ctx: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        extra_body: Dict[str, Any] = {}
        if reasoning_config:
            extra_body["reasoning_effort"] = reasoning_config.get("effort", "medium")
        return extra_body, {}


# Default profile definition for Hermes
antigravity_profile = AntigravityProviderProfile(
    name="antigravity",
    display_name="Google Antigravity",
    description="Google Cloud Code Assist (Gemini 3.7 Flash, Claude Sonnet 4.6, GPT-OSS 120B)",
    signup_url="https://cloud.google.com/products/gemini/code-assist",
    auth_type="api_key",
    api_mode="chat_completions",
    base_url="http://127.0.0.1:51122/v1",
    fallback_models=FALLBACK_MODELS,
    default_aux_model="gemini-3.1-pro",
    supports_vision=True,
    supports_vision_tool_messages=True,
    supports_health_check=True,
    env_vars=("ANTIGRAVITY_TOKEN", "GOOGLE_ACCESS_TOKEN"),
)

register_provider(antigravity_profile)


async def command_antigravity_auth(*args, **kwargs) -> str:
    """CLI/Interactive command to authenticate with Google Antigravity."""
    verifier, challenge = generate_pkce()
    state = generate_pkce()[0][:16]
    auth_url = build_authorization_url(state, challenge)

    callback_url = kwargs.get("callback")
    if callback_url:
        try:
            code, parsed_state = parse_pasted_callback(callback_url, state)
            creds = await exchange_code_for_tokens(code, verifier)
            return f"✅ Antigravity authentication successful! Logged in as: {creds.email}"
        except Exception as e:
            return f"❌ Failed to process callback URL: {e}"

    from starlette.applications import Starlette
    from starlette.responses import HTMLResponse
    from starlette.requests import Request
    from starlette.routing import Route
    import uvicorn
    import html

    callback_data = {}
    auth_event = asyncio.Event()

    async def oauth_callback(request: Request):
        err = request.query_params.get("error")
        if err:
            callback_data["error"] = err
            auth_event.set()
            return HTMLResponse(f"<h1>Authentication failed</h1><p>{html.escape(err)}</p>")

        code = request.query_params.get("code")
        ret_state = request.query_params.get("state")
        
        if not code or not ret_state:
            callback_data["error"] = "Missing code or state"
            auth_event.set()
            return HTMLResponse("<h1>Authentication failed</h1><p>Missing code or state.</p>", status_code=400)

        if ret_state != state:
            callback_data["error"] = "State mismatch"
            auth_event.set()
            return HTMLResponse("<h1>Authentication failed</h1><p>State mismatch.</p>", status_code=400)

        callback_data["code"] = code
        auth_event.set()
        return HTMLResponse(
            "<h1>Antigravity Authentication Complete</h1>"
            "<p>You can close this window and return to Hermes.</p>"
            "<script>window.close()</script>"
        )

    app = Starlette(routes=[
        Route("/oauth-callback", oauth_callback)
    ])

    config = uvicorn.Config(app=app, host="127.0.0.1", port=51121, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    
    server_task = asyncio.create_task(server.serve())

    import webbrowser
    try:
        webbrowser.open(auth_url)
    except Exception:
        pass

    prompt = (
        "=== Antigravity Authentication ===\n"
        "Your browser should have opened automatically to log in with your Google account.\n"
        "If it didn't, please open this URL manually:\n\n"
        f"   {auth_url}\n\n"
        "Waiting for authentication callback on localhost:51121 (timeout 5 minutes)...\n"
    )
    print(prompt)

    try:
        await asyncio.wait_for(auth_event.wait(), timeout=300.0)
    except asyncio.TimeoutError:
        server.should_exit = True
        return "❌ Authentication timed out waiting for browser callback."

    server.should_exit = True
    await server_task

    if "error" in callback_data:
        return f"❌ Authentication failed: {callback_data["error"]}"

    code = callback_data.get("code")
    if code:
        try:
            creds = await exchange_code_for_tokens(code, verifier)
            return f"✅ Antigravity authentication successful! Logged in as: {creds.email}"
        except Exception as e:
            return f"❌ Failed to exchange code for tokens: {e}"

    return "❌ Authentication failed for unknown reason." 


async def command_antigravity_quota() -> str:
    """CLI command to show Antigravity quota and tier status."""
    creds = load_credentials()
    if not creds:
        return "No Antigravity credentials found. Please authenticate first using /antigravity.auth"

    project_id = creds.project_id or "antigravity-default"
    return await format_quota_report(creds.access_token, project_id, email=creds.email)


async def command_antigravity_doctor() -> str:
    """CLI command to diagnose Antigravity connection and token validity."""
    creds = load_credentials()
    if not creds:
        return "❌ Antigravity Status: Not authenticated. (Run /antigravity.auth)"

    status_lines = [
        "Antigravity Diagnostics",
        "=" * 25,
        f"Account: {creds.email or 'Authenticated'}",
        f"Token Status: {'Expired (will refresh automatically)' if creds.is_expired() else 'Active'}",
        f"Refresh Token: {'Present' if creds.refresh_token else 'Missing'}",
        f"Project ID: {creds.project_id or 'Auto-resolved'}",
    ]

    try:
        proxy_url = ensure_proxy_running()
        status_lines.append(f"In-process Proxy: Active ({proxy_url})")
    except Exception as e:
        status_lines.append(f"In-process Proxy: Error ({e})")

    return "\n".join(status_lines)
