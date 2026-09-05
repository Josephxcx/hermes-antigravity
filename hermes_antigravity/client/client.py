"""Google Cloud Code Assist API client utilities and headers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://cloudcode-pa.googleapis.com"
ENDPOINT_FALLBACKS = [
    DEFAULT_ENDPOINT,
]

PROJECT_CACHE_TTL = 1800  # 30 minutes
_project_cache: Dict[str, Tuple[Optional[str], float]] = {}


def get_platform_name() -> str:
    system = platform.system().lower()
    if "darwin" in system:
        return "DARWIN"
    elif "windows" in system:
        return "WINDOWS"
    return "LINUX"


def get_default_user_agent() -> str:
    system = platform.system().lower()
    os_name = "darwin" if "darwin" in system else "windows" if "windows" in system else "linux"
    machine = platform.machine().lower()
    arch = "amd64" if machine in ("x86_64", "amd64") else "arm64" if "arm" in machine or "aarch64" in machine else machine
    return f"antigravity/1.15.8 {os_name}/{arch}"


def antigravity_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "User-Agent": os.environ.get("ANTIGRAVITY_USER_AGENT") or get_default_user_agent(),
        "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
        "Client-Metadata": json.dumps({
            "ideType": "ANTIGRAVITY",
            "platform": get_platform_name(),
            "pluginType": "GEMINI",
        }),
    }


def stable_project_id(seed: str) -> str:
    """Creates a deterministic UUID-shaped project id from a seed string."""
    raw = f"antigravity:{seed}".encode("utf-8")
    b = bytearray(hashlib.sha1(raw).digest()[:16])
    b[6] = (b[6] & 0x0F) | 0x50  # version 5
    b[8] = (b[8] & 0x3F) | 0x80  # variant RFC 4122
    hex_str = b.hex()
    return f"{hex_str[0:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:32]}"


def default_project_id(seed: str = "antigravity-default") -> str:
    env_proj = os.environ.get("ANTIGRAVITY_PROJECT_ID")
    if env_proj and env_proj.strip():
        return env_proj.strip()
    return stable_project_id(seed)


def extract_project_id(data: Any) -> Optional[str]:
    if isinstance(data, str) and data.strip():
        return data.strip()
    if not isinstance(data, dict):
        return None
    for key in (
        "antigravityProjectId",
        "projectId",
        "backendProjectId",
        "userDefinedCloudaicompanionProject",
        "cloudaicompanionProject",
        "project",
        "id",
    ):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, dict) and isinstance(val.get("id"), str):
            return str(val["id"]).strip()

    for key in ("projects", "projectIds", "cloudaicompanionProjects"):
        arr = data.get(key)
        if isinstance(arr, list):
            for item in arr:
                nested = extract_project_id(item)
                if nested:
                    return nested
    return None


async def list_cloud_ai_companion_projects(token: str) -> Optional[str]:
    """Probes endpoints for user-associated Google Cloud project ID."""
    for endpoint in ENDPOINT_FALLBACKS:
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                res = await client.post(
                    f"{endpoint}/v1internal:listCloudAICompanionProjects",
                    headers=antigravity_headers(token),
                    json={},
                )
                if res.status_code == 200:
                    extracted = extract_project_id(res.json())
                    if extracted:
                        return extracted
        except Exception as e:
            logger.debug("Failed listCloudAICompanionProjects on %s: %s", endpoint, e)
    return None


async def resolve_project_id(token: str, seed: str = "antigravity-default") -> str:
    """Resolves project ID via cache, discovery, or deterministic fallback."""
    explicit = os.environ.get("ANTIGRAVITY_PROJECT_ID")
    if explicit and explicit.strip():
        return explicit.strip()

    discovered = await list_cloud_ai_companion_projects(token)
    if discovered:
        return discovered

    return default_project_id(seed)
