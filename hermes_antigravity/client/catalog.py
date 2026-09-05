import json
import logging
import httpx
from typing import List, Optional

from hermes_antigravity.client.client import ENDPOINT_FALLBACKS, antigravity_headers, extract_project_id
from hermes_antigravity.models.models import FALLBACK_MODELS

logger = logging.getLogger(__name__)

def fetch_available_models_sync(token: str, project_id: str) -> List[str]:
    """Synchronously hits the Google fetchAvailableModels endpoint to discover supported models."""
    headers = antigravity_headers(token)
    payload = {"project": project_id}
    
    # Try all fallback endpoints
    for endpoint in ENDPOINT_FALLBACKS:
        try:
            with httpx.Client(timeout=8.0) as client:
                res = client.post(
                    f"{endpoint}/v1internal:fetchAvailableModels",
                    headers=headers,
                    json=payload,
                )
                if res.status_code == 200:
                    data = res.json()
                    models = []
                    # Process models map
                    if isinstance(data, dict) and isinstance(data.get("models"), dict):
                        for k, v in data["models"].items():
                            if k.startswith("gemini-") or k.startswith("claude-") or k.startswith("gpt-oss-"):
                                models.append(k)
                    
                    if models:
                        return list(set(models))
        except Exception as e:
            logger.debug(f"Failed to fetch models from {endpoint}: {e}")
            
    return list(FALLBACK_MODELS)

import re
import time

_MODEL_CACHE = {}
_MODEL_CACHE_TTL = 1800  # 30 mins

def is_usable_runtime_model_id(model_id: str) -> bool:
    low = model_id.lower()
    return (low.startswith("gemini-") or low.startswith("claude-") or low.startswith("gpt-oss-")) and " " not in low and not low.startswith("model_")

def build_model_match_regex(requested_id: str) -> re.Pattern:
    req = requested_id.lower()
    if req == "gemini-3.8-flash-low": return re.compile(r"gemini[- ]3\.8[- ]flash \(low\)", re.IGNORECASE)
    if req == "gemini-3.8-flash-medium": return re.compile(r"gemini[- ]3\.8[- ]flash \(medium\)", re.IGNORECASE)
    if req == "gemini-3.8-flash-high": return re.compile(r"gemini[- ]3\.8[- ]flash \(high\)", re.IGNORECASE)
    if req == "gemini-3.7-flash-low": return re.compile(r"gemini[- ]3\.7[- ]flash \(low\)", re.IGNORECASE)
    if req == "gemini-3.7-flash-medium": return re.compile(r"gemini[- ]3\.7[- ]flash \(medium\)", re.IGNORECASE)
    if req == "gemini-3.7-flash-high": return re.compile(r"gemini[- ]3\.7[- ]flash \(high\)", re.IGNORECASE)
    if req == "gemini-3.7-flash-tiered": return re.compile(r"gemini[- ]3\.7[- ]flash", re.IGNORECASE)
    if req == "gemini-3.6-flash-low": return re.compile(r"gemini[- ]3\.6[- ]flash \(low\)", re.IGNORECASE)
    if req == "gemini-3.6-flash-medium": return re.compile(r"gemini[- ]3\.6[- ]flash \(medium\)", re.IGNORECASE)
    if req == "gemini-3.6-flash-high": return re.compile(r"gemini[- ]3\.6[- ]flash \(high\)", re.IGNORECASE)
    if req == "gemini-3.5-flash-extra-low": return re.compile(r"gemini[- ]3\.5[- ]flash \(low\)", re.IGNORECASE)
    if req in ("gemini-3.5-flash-low", "gemini-3.5-flash-medium"): return re.compile(r"gemini[- ]3\.5[- ]flash \(medium\)", re.IGNORECASE)
    if req in ("gemini-3.5-flash-high", "gemini-3-flash-agent"): return re.compile(r"gemini[- ]3\.5[- ]flash \(high\)", re.IGNORECASE)
    if "claude-opus-4-6" in req: return re.compile(r"claude.*opus.*4\.6", re.IGNORECASE)
    if "claude-sonnet-4-6" in req: return re.compile(r"claude.*sonnet.*4\.6", re.IGNORECASE)
    if "gpt-oss-120b" in req: return re.compile(r"gpt.*oss.*120b", re.IGNORECASE)
    if req == "gemini-3.1-pro-low": return re.compile(r"gemini[- ]3\.1[- ]pro \(low\)", re.IGNORECASE)
    if req in ("gemini-3.1-pro-high", "gemini-pro-agent"): return re.compile(r"gemini[- ]3\.1[- ]pro \(high\)", re.IGNORECASE)
    
    escaped = re.escape(req).replace(r"\-", "[- ]")
    return re.compile(escaped, re.IGNORECASE)

def find_dynamic_model(value: Any, requested_id: str) -> Optional[str]:
    if not value: return None
    
    if isinstance(value, dict) and isinstance(value.get("models"), dict):
        models_map = value["models"]
        if is_usable_runtime_model_id(requested_id) and requested_id in models_map:
            return requested_id
            
        target_regex = build_model_match_regex(requested_id)
        for model_id, info in models_map.items():
            if not is_usable_runtime_model_id(model_id): continue
            if target_regex.search(model_id): return model_id
            if isinstance(info, dict):
                label = info.get("label") or info.get("displayName") or info.get("name")
                if isinstance(label, str) and target_regex.search(label):
                    return model_id
        return None
        
    return None

def resolve_runtime_model_sync(token: str, project_id: str, requested_runtime_model: str) -> str:
    """Dynamically resolves a requested model ID to the actual backend ID available."""
    cache_key = f"{token}::{project_id}::{requested_runtime_model}"
    now = time.time()
    if cache_key in _MODEL_CACHE:
        cached_model, expires_at = _MODEL_CACHE[cache_key]
        if expires_at > now:
            return cached_model

    payload = {"project": project_id}
    headers = antigravity_headers(token)
    
    for endpoint in ENDPOINT_FALLBACKS:
        try:
            with httpx.Client(timeout=8.0) as client:
                res = client.post(
                    f"{endpoint}/v1internal:fetchAvailableModels",
                    headers=headers,
                    json=payload,
                )
                if res.status_code == 200:
                    found = find_dynamic_model(res.json(), requested_runtime_model)
                    if found:
                        _MODEL_CACHE[cache_key] = (found, now + _MODEL_CACHE_TTL)
                        return found
        except Exception as e:
            logger.debug(f"Failed model resolution on {endpoint}: {e}")
            
    # Fallback to requested if discovery fails
    return requested_runtime_model

