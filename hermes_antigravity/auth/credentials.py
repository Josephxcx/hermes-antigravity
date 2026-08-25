"""Credential storage and loading for Antigravity OAuth."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

HERMES_AUTH_PATH = Path.home() / ".hermes" / "auth.json"
PI_AUTH_PATH = Path.home() / ".pi" / "agent" / "auth.json"


@dataclass
class AntigravityCredentials:
    access_token: str
    refresh_token: str
    expires_at: int  # Unix timestamp in milliseconds
    email: Optional[str] = None
    project_id: Optional[str] = None

    def is_expired(self, buffer_seconds: int = 300) -> bool:
        """Returns True if the token will expire within the buffer window (default 5 min)."""
        now_ms = int(time.time() * 1000)
        buffer_ms = buffer_seconds * 1000
        return now_ms >= (self.expires_at - buffer_ms)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AntigravityCredentials:
        tokens = data.get("tokens") if isinstance(data.get("tokens"), dict) else {}
        access_token = data.get("access_token") or data.get("access") or tokens.get("access_token") or tokens.get("access") or ""
        refresh_token = data.get("refresh_token") or data.get("refresh") or tokens.get("refresh_token") or tokens.get("refresh") or ""
        expires_at = data.get("expires_at") or data.get("expires") or tokens.get("expires_at") or data.get("last_refresh") or 0
        if expires_at and expires_at < 1e11:
            expires_at = int(expires_at * 1000)

        email = data.get("email") or tokens.get("email")
        project_id = data.get("project_id") or data.get("projectId") or tokens.get("project_id")
        return cls(
            access_token=str(access_token),
            refresh_token=str(refresh_token),
            expires_at=int(expires_at),
            email=email,
            project_id=project_id,
        )


def load_credentials_from_file(path: Path) -> Optional[AntigravityCredentials]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        entry = data.get("antigravity")
        if not entry and isinstance(data.get("providers"), dict):
            entry = data["providers"].get("antigravity")
        if isinstance(entry, dict):
            creds = AntigravityCredentials.from_dict(entry)
            if creds.access_token or creds.refresh_token:
                return creds
    except Exception as e:
        logger.warning("Failed to load credentials from %s: %s", path, e)
    return None


def save_credentials_to_file(creds: AntigravityCredentials, path: Path = HERMES_AUTH_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    existing["antigravity"] = {
        "type": "oauth",
        "access_token": creds.access_token,
        "refresh_token": creds.refresh_token,
        "expires_at": creds.expires_at,
        "email": creds.email,
        "project_id": creds.project_id,
    }

    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    tmp_path.replace(path)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def load_credentials() -> Optional[AntigravityCredentials]:
    """Loads credentials checking environment variables, Hermes store, and Pi store."""
    # 1. Check Hermes store (~/.hermes/auth.json)
    creds = load_credentials_from_file(HERMES_AUTH_PATH)
    if creds and creds.access_token:
        return creds

    # 2. Check Pi store (~/.pi/agent/auth.json)
    creds = load_credentials_from_file(PI_AUTH_PATH)
    if creds and creds.access_token:
        return creds

    # 3. Check environment variables
    env_token = os.environ.get("ANTIGRAVITY_TOKEN") or os.environ.get("GOOGLE_ACCESS_TOKEN")
    if env_token and not env_token.startswith("antigravity-local"):
        return AntigravityCredentials(
            access_token=env_token,
            refresh_token="",
            expires_at=int((time.time() + 3600) * 1000),
            email=os.environ.get("ANTIGRAVITY_EMAIL"),
            project_id=os.environ.get("ANTIGRAVITY_PROJECT_ID"),
        )

    return None
