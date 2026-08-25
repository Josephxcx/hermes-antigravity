"""Google OAuth 2.0 Authorization Code + PKCE implementation for Antigravity."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import html
import logging
import os
import secrets
import time
import urllib.parse
from http import HTTPStatus
from typing import Any, Callable, Optional, Tuple

import httpx

from hermes_antigravity.auth.credentials import (
    AntigravityCredentials,
    load_credentials,
    save_credentials_to_file,
)

logger = logging.getLogger(__name__)

REDIRECT_URI = "http://localhost:51121/oauth-callback"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://www.googleapis.com/oauth2/v1/userinfo?alt=json"
OAUTH_CALLBACK_TIMEOUT_SECS = 300  # 5 minutes

SCOPES = [
    "https://www.googleapis.com/auth/aicode",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
]

# Google's Antigravity desktop OAuth client credentials
DEFAULT_CLIENT_ID_B64 = (
    "MTA3MTAwNjA2MDU5MS10bWhzc2luMmgyMWxjcmUyMzV2dG9sb2poNGc0MDNlc"
    "C5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbQ=="
)
DEFAULT_CLIENT_SECRET_B64 = "R09DU1BYLUs1OEZXUjQ4NkxkTEoxbUxCOHNYQzR6NnFEQWY="


def get_client_id() -> str:
    env_id = os.environ.get("ANTIGRAVITY_CLIENT_ID")
    if env_id:
        return env_id.strip()
    return base64.b64decode(DEFAULT_CLIENT_ID_B64).decode("utf-8")


def get_client_secret() -> str:
    env_sec = os.environ.get("ANTIGRAVITY_CLIENT_SECRET")
    if env_sec:
        return env_sec.strip()
    return base64.b64decode(DEFAULT_CLIENT_SECRET_B64).decode("utf-8")


def generate_pkce() -> Tuple[str, str]:
    """Generates (code_verifier, code_challenge) using S256."""
    verifier = secrets.token_urlsafe(32)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return verifier, challenge


def build_authorization_url(state: str, code_challenge: str) -> str:
    params = {
        "client_id": get_client_id(),
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "access_type": "offline",
        "prompt": "consent",
    }
    return f"{AUTH_URL}?{urllib.parse.urlencode(params)}"


def parse_pasted_callback(raw: str, expected_state: str) -> Tuple[str, str]:
    """Parses full URL or query string pasted by the user in headless/remote environments."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("No callback URL provided.")

    if "?" in text:
        qs = text.split("?", 1)[1]
    elif text.startswith("http://") or text.startswith("https://"):
        parsed = urllib.parse.urlparse(text)
        qs = parsed.query
    else:
        qs = text

    params = urllib.parse.parse_qs(qs)
    error = params.get("error", [None])[0]
    if error:
        raise ValueError(f"OAuth authorization failed: {error}")

    code = params.get("code", [None])[0]
    state = params.get("state", [None])[0]

    if not code or not state:
        raise ValueError("Missing 'code' or 'state' in callback parameters.")

    if state != expected_state:
        raise ValueError("OAuth state parameter mismatch.")

    return code, state


async def exchange_code_for_tokens(code: str, code_verifier: str) -> AntigravityCredentials:
    """Exchanges authorization code + verifier for tokens."""
    data = {
        "client_id": get_client_id(),
        "client_secret": get_client_secret(),
        "code": code,
        "code_verifier": code_verifier,
        "grant_type": "authorization_code",
        "redirect_uri": REDIRECT_URI,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(TOKEN_URL, data=data)
        if resp.status_code != 200:
            raise RuntimeError(f"Token exchange failed ({resp.status_code}): {resp.text}")
        payload = resp.json()

    access_token = payload["access_token"]
    refresh_token = payload.get("refresh_token", "")
    expires_in = int(payload.get("expires_in", 3600))
    expires_at = int((time.time() + expires_in) * 1000)

    # Query user email
    email: Optional[str] = None
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            user_resp = await client.get(
                USERINFO_URL, headers={"Authorization": f"Bearer {access_token}"}
            )
            if user_resp.status_code == 200:
                email = user_resp.json().get("email")
    except Exception as e:
        logger.debug("Failed to query userinfo email: %s", e)

    creds = AntigravityCredentials(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=expires_at,
        email=email,
    )
    save_credentials_to_file(creds)
    return creds


_refresh_lock = asyncio.Lock()


async def refresh_access_token(creds: AntigravityCredentials) -> AntigravityCredentials:
    """Refreshes an expired access token using the stored refresh_token."""
    async with _refresh_lock:
        clean_refresh = (creds.refresh_token or "").split("|")[0].strip()
        if not clean_refresh:
            raise ValueError(
                "No refresh token available for Antigravity OAuth. Please re-authenticate with /antigravity.auth."
            )

        data = {
            "client_id": get_client_id(),
            "client_secret": get_client_secret(),
            "refresh_token": clean_refresh,
            "grant_type": "refresh_token",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(TOKEN_URL, data=data)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Token refresh failed ({resp.status_code}): {resp.text}. "
                    "You may need to log in again with /antigravity.auth"
                )
            payload = resp.json()

        creds.access_token = payload["access_token"]
        if "refresh_token" in payload:
            creds.refresh_token = payload["refresh_token"]
        expires_in = int(payload.get("expires_in", 3600))
        creds.expires_at = int((time.time() + expires_in) * 1000)

        save_credentials_to_file(creds)
        return creds


async def ensure_valid_credentials() -> AntigravityCredentials:
    """Loads stored credentials and refreshes them if expired or near expiry."""
    creds = load_credentials()
    if not creds:
        raise ValueError(
            "No Antigravity credentials found. Please run `/antigravity.auth` to authenticate."
        )

    if creds.is_expired():
        if creds.refresh_token:
            try:
                creds = await refresh_access_token(creds)
            except Exception as primary_err:
                # Try fallback token from Pi store if primary Hermes token was invalid
                from hermes_antigravity.auth.credentials import PI_AUTH_PATH, load_credentials_from_file
                pi_creds = load_credentials_from_file(PI_AUTH_PATH)
                if pi_creds and pi_creds.refresh_token and pi_creds.refresh_token != creds.refresh_token:
                    logger.info("Attempting refresh from Pi credentials store fallback...")
                    creds = await refresh_access_token(pi_creds)
                else:
                    raise primary_err
        else:
            logger.warning("Antigravity token is expired and has no refresh token.")

    return creds
