"""Unit tests for Antigravity commands and usage utilities."""

import pytest
from unittest.mock import AsyncMock, patch

from hermes_antigravity import (
    command_antigravity_doctor,
    command_antigravity_quota,
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
async def test_command_antigravity_quota_unauthenticated():
    with patch("hermes_antigravity.load_credentials", return_value=None):
        out = await command_antigravity_quota()
        assert "No Antigravity credentials found" in out
