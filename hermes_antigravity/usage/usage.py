"""Quota and account usage diagnostics for Google Cloud Code Assist."""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List, Optional

import httpx

from hermes_antigravity.client.client import ENDPOINT_FALLBACKS, antigravity_headers

logger = logging.getLogger(__name__)


def format_progress_bar(remaining_ratio: Optional[float], width: int = 20) -> str:
    if remaining_ratio is None:
        return f"[{'?' * width}]"
    ratio = max(0.0, min(1.0, remaining_ratio))
    filled = int(round(ratio * width))
    return f"[{'#' * filled}{'-' * (width - filled)}]"


def format_reset_time(reset_time_str: Optional[str]) -> str:
    if not reset_time_str:
        return "n/a"
    try:
        dt = datetime.datetime.fromisoformat(reset_time_str.replace("Z", "+00:00"))
        now = datetime.datetime.now(datetime.timezone.utc)
        diff = dt - now
        total_seconds = int(diff.total_seconds())
        if total_seconds <= 0:
            return "now"
        mins = (total_seconds % 3600) // 60
        hours = (total_seconds % 86400) // 3600
        days = total_seconds // 86400
        if days > 0:
            return f"{days}d {hours}h"
        if hours > 0:
            return f"{hours}h {mins}m"
        return f"{mins}m"
    except Exception:
        return reset_time_str


async def fetch_quota_summary(token: str, project_id: str) -> Dict[str, Any]:
    """Fetches quota details from Google Cloud Code Assist loadCodeAssist endpoint."""
    headers = antigravity_headers(token)
    headers["Accept"] = "application/json"

    for endpoint in ENDPOINT_FALLBACKS:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(
                    f"{endpoint}/v1internal:loadCodeAssist",
                    headers=headers,
                    json={"project": project_id},
                )
                if res.status_code == 200:
                    return res.json()
        except Exception as e:
            logger.debug("Failed loadCodeAssist on %s: %s", endpoint, e)

    return {}


async def format_quota_report(token: str, project_id: str, email: Optional[str] = None) -> str:
    """Formats a human-readable quota report."""
    data = await fetch_quota_summary(token, project_id)
    lines = [
        "Antigravity Quota Status",
        "=" * 30,
        f"Account: {email or 'Authenticated User'}",
        f"Project: {project_id}",
        "",
    ]

    tier = data.get("tier", {})
    tier_name = tier.get("name") or data.get("currentTier", {}).get("name") or "Standard"
    lines.append(f"Subscription Tier: {tier_name}")
    lines.append("")

    quotas = data.get("quotas") or data.get("quotaSummary", {}).get("groups", [])
    if isinstance(quotas, list) and quotas:
        lines.append("Model Quotas:")
        for q in quotas:
            if isinstance(q, dict):
                name = q.get("name") or q.get("modelName") or "General Quota"
                rem = q.get("remainingFraction") or q.get("remaining")
                bar = format_progress_bar(rem)
                pct = f"{round(rem * 100, 1)}%" if rem is not None else "Unknown"
                reset = format_reset_time(q.get("resetTime"))
                lines.append(f"  • {name:20} {bar} {pct:>6} (Resets: {reset})")
    else:
        lines.append("Quota details: Active (No specific bucket throttling detected)")

    return "\n".join(lines)
