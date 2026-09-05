"""Unit tests for AntigravityProviderProfile and plugin.yaml manifest."""

import re
from pathlib import Path

from hermes_antigravity import AntigravityProviderProfile, antigravity_profile


def test_antigravity_profile_attributes():
    assert antigravity_profile.name == "antigravity"
    assert antigravity_profile.auth_type == "oauth"
    assert antigravity_profile.supports_vision is True
    assert antigravity_profile.default_aux_model == "gemini-3.8-flash"


def test_antigravity_profile_supported_reasoning_efforts():
    profile = AntigravityProviderProfile()
    efforts = profile.supported_reasoning_efforts()
    assert efforts == ("off", "minimal", "low", "medium", "high", "xhigh")

    # Also check via antigravity_profile
    assert antigravity_profile.supported_reasoning_efforts() == (
        "off",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
    )


def test_antigravity_profile_resolve_aux_model():
    profile = AntigravityProviderProfile()
    assert profile.resolve_aux_model(vision=True) == "gemini-3.7-flash"
    assert profile.resolve_aux_model(vision=False) == "gemini-3.5-flash"
    assert antigravity_profile.resolve_aux_model(vision=True) == "gemini-3.7-flash"
    assert antigravity_profile.resolve_aux_model(vision=False) == "gemini-3.5-flash"


def test_antigravity_profile_build_api_kwargs_extras():
    profile = AntigravityProviderProfile()
    extra_body, top_kwargs = profile.build_api_kwargs_extras(
        reasoning_config={"effort": "high"}
    )
    assert extra_body == {"reasoning_effort": "high"}
    assert top_kwargs == {}


def test_plugin_yaml_manifest():
    plugin_yaml_path = Path(__file__).resolve().parent.parent / "plugin.yaml"
    assert plugin_yaml_path.exists(), "plugin.yaml must exist at repo root"

    content = plugin_yaml_path.read_text(encoding="utf-8")
    parsed: dict[str, str] = {}
    for line in content.splitlines():
        line = line.strip()
        if ":" in line and not line.startswith("-") and not line.startswith("#"):
            k, v = line.split(":", 1)
            parsed[k.strip()] = v.strip()

    assert parsed.get("name") == "antigravity"
    assert parsed.get("kind") == "model-provider"
    assert parsed.get("version") == "0.5.0"
    assert "description" in parsed
    assert "antigravity.usage" in content
    assert "antigravity.quota" not in content
    assert "antigravity.doctor" in content
    assert "antigravity.auth" in content
