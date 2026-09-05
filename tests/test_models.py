"""Unit tests for Antigravity model routing and token limits."""

from hermes_antigravity.models.models import (
    FALLBACK_MODELS,
    get_max_output_tokens,
    get_runtime_model_id,
)


def test_gemini_38_flash_routing():
    # Gemini 3.8 uses tiered routing (low/medium/high)
    assert get_runtime_model_id("gemini-3.8-flash", "high") == "gemini-3.8-flash-high"
    assert get_runtime_model_id("gemini-3.8-flash", "medium") == "gemini-3.8-flash-medium"
    assert get_runtime_model_id("gemini-3.8-flash", "low") == "gemini-3.8-flash-low"
    assert get_runtime_model_id("antigravity/gemini-3.8-flash") == "gemini-3.8-flash-medium"


def test_gemini_37_flash_routing():
    # Gemini 3.7 uses tiered routing
    assert get_runtime_model_id("gemini-3.7-flash", "high") == "gemini-3.7-flash-tiered"
    assert get_runtime_model_id("gemini-3.7-flash", "off") == "gemini-3.7-flash-low"
    assert get_runtime_model_id("antigravity/gemini-3.7-flash", "medium") == "gemini-3.7-flash-tiered"


def test_claude_sonnet_routing():
    assert get_runtime_model_id("claude-sonnet-4-6", "high") == "claude-sonnet-4-6"
    assert get_runtime_model_id("antigravity/claude-sonnet-4-6") == "claude-sonnet-4-6"


def test_claude_opus_routing():
    assert get_runtime_model_id("claude-opus-4-6", "high") == "claude-opus-4-6-thinking"
    assert get_runtime_model_id("claude-opus-4-6", "medium") == "claude-opus-4-6-thinking"


def test_gemini_31_pro_routing():
    assert get_runtime_model_id("gemini-3.1-pro", "low") == "gemini-3.1-pro-low"
    assert get_runtime_model_id("gemini-3.1-pro", "high") == "gemini-pro-agent"


def test_gpt_oss_routing():
    assert get_runtime_model_id("gpt-oss-120b", "medium") == "gpt-oss-120b-medium"


def test_max_output_tokens():
    assert get_max_output_tokens("gemini-3.8-flash") == 65536
    assert get_max_output_tokens("gemini-3.7-flash") == 65536
    assert get_max_output_tokens("claude-sonnet-4-6") == 64000
    assert get_max_output_tokens("gpt-oss-120b") == 32768
    assert get_max_output_tokens("gemini-3.1-pro") == 65535


def test_fallback_models_exist():
    assert "gemini-3.8-flash" in FALLBACK_MODELS
    assert "gemini-3.7-flash" in FALLBACK_MODELS
    assert "claude-sonnet-4-6" in FALLBACK_MODELS
    assert "gemini-3.1-pro" in FALLBACK_MODELS
