"""Unit tests for Antigravity OAuth and credential handling."""

import base64
import hashlib
import json
import os
import time
import urllib.parse
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_antigravity.auth.credentials import (
    AntigravityCredentials,
    get_default_auth_save_path,
    get_hermes_auth_paths,
    load_credentials,
    load_credentials_from_file,
    save_credentials_to_file,
)
from hermes_antigravity.auth.oauth import (
    build_authorization_url,
    generate_pkce,
    parse_pasted_callback,
)


def test_pkce_generation():
    verifier, challenge = generate_pkce()
    assert len(verifier) >= 32
    # Verify challenge is base64url(sha256(verifier))
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).decode("ascii").rstrip("=")
    assert challenge == expected


def test_build_authorization_url():
    state = "test-state-12345"
    challenge = "test-challenge-abc"
    url = build_authorization_url(state, challenge)

    parsed = urllib.parse.urlparse(url)
    assert parsed.scheme == "https"
    assert parsed.netloc == "accounts.google.com"
    assert parsed.path == "/o/oauth2/v2/auth"

    params = urllib.parse.parse_qs(parsed.query)
    assert params["state"] == [state]
    assert params["code_challenge"] == [challenge]
    assert params["code_challenge_method"] == ["S256"]
    assert params["response_type"] == ["code"]
    assert "aicode" in params["scope"][0]


def test_parse_pasted_callback_valid():
    state = "state-987"
    raw_url = f"http://localhost:51121/oauth-callback?code=4/0Abc123xyz&state={state}"
    code, parsed_state = parse_pasted_callback(raw_url, state)
    assert code == "4/0Abc123xyz"
    assert parsed_state == state


def test_parse_pasted_callback_mismatched_state():
    with pytest.raises(ValueError, match="state parameter mismatch"):
        parse_pasted_callback("http://localhost:51121/oauth-callback?code=abc&state=wrong", "expected")


def test_parse_pasted_callback_error():
    with pytest.raises(ValueError, match="OAuth authorization failed: access_denied"):
        parse_pasted_callback("http://localhost:51121/oauth-callback?error=access_denied&state=s", "s")


def test_credential_expiry_logic():
    now_ms = int(time.time() * 1000)
    # Expired token
    creds_expired = AntigravityCredentials(
        access_token="tok1",
        refresh_token="ref1",
        expires_at=now_ms - 1000,
    )
    assert creds_expired.is_expired() is True

    # Token expiring in 2 minutes (buffer is 5 minutes -> should report expired for refresh)
    creds_near_expiry = AntigravityCredentials(
        access_token="tok2",
        refresh_token="ref2",
        expires_at=now_ms + 120_000,
    )
    assert creds_near_expiry.is_expired(buffer_seconds=300) is True

    # Token valid for 1 hour
    creds_valid = AntigravityCredentials(
        access_token="tok3",
        refresh_token="ref3",
        expires_at=now_ms + 3600_000,
    )
    assert creds_valid.is_expired() is False


def test_credential_save_and_load(tmp_path: Path):
    auth_file = tmp_path / "auth.json"
    creds = AntigravityCredentials(
        access_token="acc-token-123",
        refresh_token="ref-token-456",
        expires_at=1750000000000,
        email="dev@example.com",
        project_id="test-proj-789",
    )
    save_credentials_to_file(creds, auth_file)

    loaded = load_credentials_from_file(auth_file)
    assert loaded is not None
    assert loaded.access_token == "acc-token-123"
    assert loaded.refresh_token == "ref-token-456"
    assert loaded.expires_at == 1750000000000
    assert loaded.email == "dev@example.com"
    assert loaded.project_id == "test-proj-789"


def test_load_credentials_nested_providers(tmp_path: Path):
    auth_file = tmp_path / "auth.json"
    data = {
        "providers": {
            "antigravity": {
                "access_token": "nested-acc-token",
                "refresh_token": "nested-ref-token",
                "expires_at": 1800000000000,
                "email": "nested@example.com",
                "project_id": "nested-proj",
            }
        }
    }
    auth_file.write_text(json.dumps(data), encoding="utf-8")
    loaded = load_credentials_from_file(auth_file)
    assert loaded is not None
    assert loaded.access_token == "nested-acc-token"
    assert loaded.refresh_token == "nested-ref-token"
    assert loaded.email == "nested@example.com"
    assert loaded.project_id == "nested-proj"


def test_load_credentials_string_token(tmp_path: Path):
    auth_file = tmp_path / "auth.json"
    data = {"antigravity": "direct-string-token-abc"}
    auth_file.write_text(json.dumps(data), encoding="utf-8")
    loaded = load_credentials_from_file(auth_file)
    assert loaded is not None
    assert loaded.access_token == "direct-string-token-abc"
    assert loaded.refresh_token == ""


def test_hermes_home_multi_profile_resolution(tmp_path: Path, monkeypatch):
    profile_dir = tmp_path / "profiles" / "telegram"
    profile_dir.mkdir(parents=True)
    profile_auth = profile_dir / "auth.json"

    creds = AntigravityCredentials(
        access_token="telegram-profile-token",
        refresh_token="telegram-profile-refresh",
        expires_at=1800000000000,
        email="telegram@example.com",
    )
    save_credentials_to_file(creds, profile_auth)

    monkeypatch.setenv("HERMES_HOME", str(profile_dir))
    paths = get_hermes_auth_paths()
    assert paths[0] == profile_auth
    assert get_default_auth_save_path() == profile_auth

    loaded = load_credentials()
    assert loaded is not None
    assert loaded.access_token == "telegram-profile-token"
    assert loaded.email == "telegram@example.com"


def test_hermes_home_fallback_to_default(tmp_path: Path, monkeypatch):
    empty_profile_dir = tmp_path / "profiles" / "empty"
    empty_profile_dir.mkdir(parents=True)

    default_dir = tmp_path / "default_hermes"
    default_dir.mkdir(parents=True)
    default_auth = default_dir / "auth.json"

    creds = AntigravityCredentials(
        access_token="default-token",
        refresh_token="default-refresh",
        expires_at=1800000000000,
        email="default@example.com",
    )
    save_credentials_to_file(creds, default_auth)

    monkeypatch.setenv("HERMES_HOME", str(empty_profile_dir))
    with patch("hermes_antigravity.auth.credentials.DEFAULT_HERMES_AUTH_PATH", default_auth):
        loaded = load_credentials()
        assert loaded is not None
        assert loaded.access_token == "default-token"
        assert loaded.email == "default@example.com"
