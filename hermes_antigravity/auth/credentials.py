"""Credential storage and loading for Antigravity OAuth."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_HERMES_AUTH_PATH = Path.home() / ".hermes" / "auth.json"
HERMES_AUTH_PATH = DEFAULT_HERMES_AUTH_PATH  # backwards compatibility


def get_hermes_auth_paths() -> List[Path]:
    """Returns candidate auth.json paths in priority order (HERMES_HOME first, then default ~/.hermes)."""
    paths: List[Path] = []
    hermes_home = os.environ.get("HERMES_HOME")
    if hermes_home:
        candidate = Path(hermes_home).expanduser() / "auth.json"
        paths.append(candidate)
    if DEFAULT_HERMES_AUTH_PATH not in paths:
        paths.append(DEFAULT_HERMES_AUTH_PATH)
    return paths


def get_default_auth_save_path() -> Path:
    """Returns the preferred path to save auth credentials (HERMES_HOME if set, else ~/.hermes/auth.json)."""
    hermes_home = os.environ.get("HERMES_HOME")
    if hermes_home:
        return Path(hermes_home).expanduser() / "auth.json"
    return DEFAULT_HERMES_AUTH_PATH


@dataclass
class AntigravityCredentials:
    access_token: str
    refresh_token: str
    expires_at: int  # Unix timestamp in milliseconds
    email: Optional[str] = None
    project_id: Optional[str] = None

    def is_expired(self, buffer_seconds: int = 300) -> bool:
        """Returns True if the token will expire within the buffer window (default 5 min)."""
        if not self.expires_at:
            return True
        return time.time() * 1000 >= (self.expires_at - buffer_seconds * 1000)

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
    """Loads Antigravity credentials from a given JSON file (supports top-level and providers dict)."""
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
        elif isinstance(entry, str) and entry.strip():
            return AntigravityCredentials(
                access_token=entry.strip(),
                refresh_token="",
                expires_at=int((time.time() + 3600) * 1000),
            )
    except Exception as e:
        logger.warning("Failed to load credentials from %s: %s", path, e)
    return None


def save_credentials_to_file(creds: AntigravityCredentials, path: Optional[Path] = None) -> None:
    """Saves Antigravity credentials to JSON file."""
    target_path = path if path is not None else get_default_auth_save_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if target_path.exists():
        try:
            existing = json.loads(target_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    auth_entry = {
        "type": "oauth",
        "access_token": creds.access_token,
        "refresh_token": creds.refresh_token,
        "expires_at": creds.expires_at,
        "email": creds.email,
        "project_id": creds.project_id,
    }
    existing["antigravity"] = auth_entry
    if isinstance(existing.get("providers"), dict) and "antigravity" in existing["providers"]:
        existing["providers"]["antigravity"] = auth_entry

    tmp_path = target_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    tmp_path.replace(target_path)
    try:
        target_path.chmod(0o600)
    except OSError:
        pass


def load_credentials() -> Optional[AntigravityCredentials]:
    """Loads credentials checking Hermes multi-profile stores and environment variables."""
    # 1. Check Hermes multi-profile stores ($HERMES_HOME/auth.json then ~/.hermes/auth.json)
    for auth_path in get_hermes_auth_paths():
        creds = load_credentials_from_file(auth_path)
        if creds and (creds.access_token or creds.refresh_token):
            return creds

    # 2. Check environment variables
    env_token = os.environ.get("ANTIGRAVITY_TOKEN") or os.environ.get("GOOGLE_ACCESS_TOKEN")
    if env_token:
        return AntigravityCredentials(
            access_token=env_token,
            refresh_token="",
            expires_at=int((time.time() + 3600) * 1000),
            email=os.environ.get("ANTIGRAVITY_EMAIL"),
            project_id=os.environ.get("ANTIGRAVITY_PROJECT_ID"),
        )

    return None

def resolve_runtime_credentials() -> dict:
    """Pre-flight credential resolver for Hermes _OAUTH_RUNTIME_PROVIDERS."""
    creds = load_credentials()
    if not creds:
        from hermes_cli.auth import AuthError
        raise AuthError("No Antigravity credentials found. Run /login antigravity.", provider="antigravity", code="missing_credentials")
        
    if creds.is_expired():
        if not creds.refresh_token:
            from hermes_cli.auth import AuthError
            raise AuthError("Antigravity token expired and no refresh token available.", provider="antigravity", code="missing_refresh_token", relogin_required=True)
            
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Antigravity token expired pre-flight. Refreshing synchronously...")
        
        import asyncio
        import threading
        from hermes_antigravity.auth.oauth import refresh_access_token
        
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
            from hermes_cli.auth import AuthError
            raise AuthError("Failed to refresh Antigravity token.", provider="antigravity", code="refresh_failed", relogin_required=True) from getattr(t, "err")
            
        creds = getattr(t, "res")
        save_credentials_to_file(creds)
            
    return {
        "api_key": creds.access_token,
        "expires_at": creds.expires_at,
        "source": "oauth",
    }
