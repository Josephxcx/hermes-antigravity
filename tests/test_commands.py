"""Unit tests for Antigravity commands and usage utilities."""

import pytest
from unittest.mock import AsyncMock, patch

from hermes_antigravity import (
    command_antigravity_auth,
    command_antigravity_doctor,
    command_antigravity_usage,
    register,
)
from hermes_antigravity.auth.credentials import AntigravityCredentials
from hermes_antigravity.usage.usage import format_progress_bar, format_reset_time


def test_format_progress_bar():
    assert format_progress_bar(1.0, width=10) == "[##########]"
    assert format_progress_bar(0.0, width=10) == "[----------]"
    assert format_progress_bar(0.5, width=10) == "[#####-----]"
    assert format_progress_bar(None, width=10) == "[??????????]"


def test_format_reset_time():
    assert format_reset_time(None) == "n/a"
    assert format_reset_time("") == "n/a"


@pytest.mark.asyncio
async def test_command_antigravity_doctor_unauthenticated():
    with patch("hermes_antigravity.load_credentials", return_value=None):
        out = await command_antigravity_doctor()
        assert "Not authenticated" in out


@pytest.mark.asyncio
async def test_command_antigravity_doctor_authenticated():
    creds = AntigravityCredentials(
        access_token="tok123",
        refresh_token="ref123",
        expires_at=9999999999999,
        email="test@gmail.com",
        project_id="proj-456",
    )
    with patch("hermes_antigravity.load_credentials", return_value=creds):
        with patch("hermes_antigravity.ensure_proxy_running", return_value="http://127.0.0.1:51122/v1"):
            out = await command_antigravity_doctor()
            assert "test@gmail.com" in out
            assert "Active" in out
            assert "proj-456" in out


@pytest.mark.asyncio
async def test_command_antigravity_usage_unauthenticated():
    with patch("hermes_antigravity.load_credentials", return_value=None):
        out = await command_antigravity_usage()
        assert "No Antigravity credentials found" in out


def test_plugin_register_with_context():
    class MockContext:
        def __init__(self):
            self.commands = {}

        def register_command(self, name, handler, description=""):
            self.commands[name] = {"handler": handler, "description": description}

    ctx = MockContext()
    with patch("hermes_antigravity.register_provider") as mock_reg_provider:
        register(ctx)
        mock_reg_provider.assert_called_once()

    assert "antigravity.usage" in ctx.commands
    assert "antigravity.quota" not in ctx.commands
    assert "antigravity.doctor" in ctx.commands
    assert "antigravity.auth" in ctx.commands

    assert ctx.commands["antigravity.usage"]["handler"] == command_antigravity_usage
    assert ctx.commands["antigravity.doctor"]["handler"] == command_antigravity_doctor
    assert ctx.commands["antigravity.auth"]["handler"] == command_antigravity_auth


def test_plugin_register_without_context():
    with patch("hermes_antigravity.register_provider") as mock_reg_provider:
        register()
        mock_reg_provider.assert_called_once()

    with patch("hermes_antigravity.register_provider") as mock_reg_provider:
        register(object())
        mock_reg_provider.assert_called_once()

