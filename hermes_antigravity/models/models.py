"""Model routing, thinking configs, and runtime limits for Antigravity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

PROVIDER_ID = "antigravity"
PROVIDER_NAME = "Antigravity"

# Public model IDs → backend request model IDs by thinking effort
ANTIGRAVITY_ROUTING: Dict[str, Dict[str, Any]] = {
  "claude-opus-4-6": {
    "off": "claude-opus-4-6-thinking",
    "routing": {
      "minimal": "claude-opus-4-6-thinking",
      "low": "claude-opus-4-6-thinking",
      "medium": "claude-opus-4-6-thinking",
      "high": "claude-opus-4-6-thinking",
      "xhigh": "claude-opus-4-6-thinking",
    },
    "defaultRequestId": "claude-opus-4-6-thinking",
  },
  "claude-sonnet-4-6": {
    "off": "claude-sonnet-4-6",
    "routing": {
      "minimal": "claude-sonnet-4-6",
      "low": "claude-sonnet-4-6",
      "medium": "claude-sonnet-4-6",
      "high": "claude-sonnet-4-6",
      "xhigh": "claude-sonnet-4-6",
    },
    "defaultRequestId": "claude-sonnet-4-6",
  },
  "gemini-3.1-pro": {
    "off": "gemini-3.1-pro-low",
    "routing": {
      "minimal": "gemini-3.1-pro-low",
      "low": "gemini-3.1-pro-low",
      "medium": "gemini-3.1-pro-low",
      "high": "gemini-pro-agent",
      "xhigh": "gemini-pro-agent",
    },
    "defaultRequestId": "gemini-3.1-pro-low",
  },
  "gemini-3.7-flash": {
    "off": "gemini-3.7-flash-tiered",
    "routing": {
      "minimal": "gemini-3.7-flash-tiered",
      "low": "gemini-3.7-flash-tiered",
      "medium": "gemini-3.7-flash-tiered",
      "high": "gemini-3.7-flash-tiered",
      "xhigh": "gemini-3.7-flash-tiered",
    },
    "defaultRequestId": "gemini-3.7-flash-tiered",
  },
  "gemini-3.6-flash": {
    "off": "gemini-3.6-flash-low",
    "routing": {
      "minimal": "gemini-3.6-flash-low",
      "low": "gemini-3.6-flash-low",
      "medium": "gemini-3.6-flash-medium",
      "high": "gemini-3.6-flash-high",
      "xhigh": "gemini-3.6-flash-high",
    },
    "defaultRequestId": "gemini-3.6-flash-low",
  },
  "gemini-3.5-flash": {
    "off": "gemini-3.5-flash-extra-low",
    "routing": {
      "minimal": "gemini-3.5-flash-extra-low",
      "low": "gemini-3.5-flash-low",
      "medium": "gemini-3.5-flash-low",
      "high": "gemini-3-flash-agent",
      "xhigh": "gemini-3-flash-agent",
    },
    "defaultRequestId": "gemini-3.5-flash-extra-low",
  },
  "gpt-oss-120b": {
    "off": "gpt-oss-120b-medium",
    "routing": {
      "minimal": "gpt-oss-120b-medium",
      "low": "gpt-oss-120b-medium",
      "medium": "gpt-oss-120b-medium",
      "high": "gpt-oss-120b-medium",
    },
    "defaultRequestId": "gpt-oss-120b-medium",
  },
}

RUNTIME_MAX_OUTPUT_TOKENS: Dict[str, int] = {
  "gemini-3.7-flash": 65536,
  "gemini-3.7-flash-tiered": 65536,
  "gemini-3.6-flash": 65536,
  "gemini-3.6-flash-low": 65536,
  "gemini-3.6-flash-medium": 65536,
  "gemini-3.6-flash-high": 65536,
  "gemini-3.5-flash": 65536,
  "gemini-3.5-flash-extra-low": 65536,
  "gemini-3.5-flash-low": 65536,
  "gemini-3-flash-agent": 65536,
  "gemini-3.1-pro": 65535,
  "gemini-3.1-pro-low": 65535,
  "gemini-3.1-pro-high": 65535,
  "gemini-pro-agent": 65535,
  "claude-opus-4-6": 64000,
  "claude-opus-4-6-thinking": 64000,
  "claude-sonnet-4-6": 64000,
  "gpt-oss-120b": 32768,
  "gpt-oss-120b-medium": 32768,
}

FALLBACK_MODELS = (
  "gemini-3.7-flash",
  "claude-sonnet-4-6",
  "claude-opus-4-6",
  "gemini-3.1-pro",
  "gemini-3.6-flash",
  "gemini-3.5-flash",
  "gpt-oss-120b",
)


def get_runtime_model_id(model_id: str, thinking_effort: Optional[str] = None) -> str:
    """Maps a user-selected model id + thinking effort to Google Cloud Code Assist runtime ID."""
    clean_id = model_id.lower().replace("antigravity/", "").strip()
    config = ANTIGRAVITY_ROUTING.get(clean_id)
    if not config:
        return clean_id

    effort = (thinking_effort or "medium").lower()
    if effort == "off":
        return str(config.get("off") or config.get("defaultRequestId") or clean_id)

    routing = config.get("routing", {})
    if effort in routing:
        return str(routing[effort])

    return str(config.get("defaultRequestId") or clean_id)


def get_max_output_tokens(model_id: str, runtime_model: Optional[str] = None) -> int:
    if runtime_model and runtime_model in RUNTIME_MAX_OUTPUT_TOKENS:
        return RUNTIME_MAX_OUTPUT_TOKENS[runtime_model]
    clean_id = model_id.lower().replace("antigravity/", "").strip()
    if clean_id in RUNTIME_MAX_OUTPUT_TOKENS:
        return RUNTIME_MAX_OUTPUT_TOKENS[clean_id]
    if runtime_model:
        if runtime_model.startswith("claude-"):
            return 64000
        if runtime_model.startswith("gpt-oss-"):
            return 32768
        if runtime_model.startswith("gemini-3.1-pro") or runtime_model == "gemini-pro-agent":
            return 65535
        if runtime_model.startswith("gemini-"):
            return 65536
    return 8192
