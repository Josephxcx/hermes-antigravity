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


async def command_antigravity_auth() -> str:
    """CLI/Interactive command to authenticate with Google Antigravity."""
    verifier, challenge = generate_pkce()
    state = generate_pkce()[0][:16]
    auth_url = build_authorization_url(state, challenge)

    prompt = (
        "=== Antigravity Authentication ===\n"
        "1. Open the following URL in your browser:\n\n"
        f"   {auth_url}\n\n"
        "2. Log in with your Google account.\n"
        "3. If you are in a desktop environment, the browser will redirect to localhost:51121.\n"
        "   If you are on a remote/headless machine, copy the redirected URL and run:\n"
        "   /antigravity.auth --callback <URL>\n"
    )
    return prompt


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
