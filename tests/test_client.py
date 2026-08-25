"""Unit tests for Antigravity client utilities and project ID extraction."""

import pytest
from hermes_antigravity.client.client import (
    antigravity_headers,
    default_project_id,
    extract_project_id,
    get_default_user_agent,
    get_platform_name,
    stable_project_id,
)


def test_antigravity_headers():
    headers = antigravity_headers("test-token-xyz")
    assert headers["Authorization"] == "Bearer test-token-xyz"
    assert headers["Content-Type"] == "application/json"
    assert headers["Accept"] == "text/event-stream"
    assert "vscode_cloudshelleditor" in headers["X-Goog-Api-Client"]
    assert "ANTIGRAVITY" in headers["Client-Metadata"]


def test_stable_project_id():
    p1 = stable_project_id("user1@example.com")
    p2 = stable_project_id("user1@example.com")
    p3 = stable_project_id("user2@example.com")

    # Deterministic output for same seed
    assert p1 == p2
    assert p1 != p3
    assert len(p1.split("-")) == 5  # UUID shaped


def test_extract_project_id():
    # Direct field
    assert extract_project_id({"projectId": "proj-direct"}) == "proj-direct"
    assert extract_project_id({"antigravityProjectId": "proj-agy"}) == "proj-agy"

    # Nested object
    assert extract_project_id({"project": {"id": "proj-nested"}}) == "proj-nested"

    # List of projects
    assert extract_project_id({"projects": ["proj-in-list"]}) == "proj-in-list"
    assert extract_project_id({"projects": [{"id": "proj-in-dict-list"}]}) == "proj-in-dict-list"

    # None for empty/unrecognized
    assert extract_project_id({}) is None
    assert extract_project_id(None) is None


def test_platform_and_user_agent():
    platform_name = get_platform_name()
    assert platform_name in ("LINUX", "DARWIN", "WINDOWS")
    ua = get_default_user_agent()
    assert ua.startswith("antigravity/")
