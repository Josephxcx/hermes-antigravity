import json
import pytest
from unittest.mock import patch, MagicMock

from hermes_antigravity.auth.credentials import AntigravityCredentials
from hermes_antigravity.client.native import AntigravityClient, ENDPOINT_FALLBACKS
from hermes_antigravity.__init__ import AntigravityProviderProfile

@pytest.fixture
def mock_creds():
    return AntigravityCredentials("tok123", "ref123", expires_at=9999999999999, email="test@test.com", project_id="proj1")

def test_create_client_returns_native():
    profile = AntigravityProviderProfile(name="antigravity")
    client = profile.create_client(api_key="tok123", base_url="test")
    assert isinstance(client, AntigravityClient)
    assert getattr(client, "HERMES_SKIP_TRANSPORT_WRAP", False) is True
    assert hasattr(client.chat.completions, "create")
    assert client.api_key == "tok123"

def test_native_client_accepts_hermes_kwargs():
    # Hermes might pass random kwargs like timeout, default_headers, ssl_verify, etc.
    client = AntigravityClient(
        api_key="abc", 
        base_url="def", 
        timeout=10, 
        default_headers={"X-Test": "1"},
        ssl_verify=False,
    )
    assert client.api_key == "abc"
    assert client.base_url == "def"

def test_native_client_401_triggers_refresh_and_retry(mock_creds):
    client = AntigravityClient()
    
    # We will mock the httpx.Client to return a 401 on the first call, and 200 on the second
    mock_resp_401 = MagicMock()
    mock_resp_401.status_code = 401
    mock_resp_401.text = "Unauthorized"
    
    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    mock_resp_200.iter_lines.return_value = iter([
        'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}], "finishReason": "STOP"}}',
        "data: [DONE]"
    ])
    
    # Store state to assert call counts
    call_count = {"send": 0}
    
    def fake_send(*args, **kwargs):
        call_count["send"] += 1
        if call_count["send"] == 1:
            return mock_resp_401
        return mock_resp_200

    mock_http_client = MagicMock()
    mock_http_client.build_request.return_value = "req"
    mock_http_client.send.side_effect = fake_send
    
    client._http_client = mock_http_client
    
    async def mock_refresh(c):
        return AntigravityCredentials("new_tok", "new_ref", expires_at=9999999999999)
        
    with patch("hermes_antigravity.client.native.load_credentials", return_value=mock_creds):
        with patch("hermes_antigravity.client.catalog.resolve_runtime_model_sync", return_value="gemini-3.7-flash-tiered"):
            with patch("hermes_antigravity.client.native.refresh_access_token", side_effect=mock_refresh) as mock_ref:
                with patch("hermes_antigravity.client.native.save_credentials_to_file") as mock_save:
                    # Should not raise exception
                    resp = client.chat.completions.create(model="gemini-3.7-flash", messages=[{"role":"user", "content":"hi"}], stream=False)
                    
                    assert call_count["send"] == 2
                    mock_ref.assert_called_once()
                    mock_save.assert_called_once()
                    # Verify response was parsed
                    assert resp.choices[0].message.content == "Hello"


def test_native_client_401_refresh_failure_raises_auth_error(mock_creds):
    client = AntigravityClient()
    
    mock_resp_401 = MagicMock()
    mock_resp_401.status_code = 401
    mock_resp_401.text = "Unauthorized"
    
    mock_http_client = MagicMock()
    mock_http_client.build_request.return_value = "req"
    mock_http_client.send.return_value = mock_resp_401
    client._http_client = mock_http_client
    
    async def mock_refresh_fail(c):
        raise Exception("Refresh failed")
        
    with patch("hermes_antigravity.client.native.load_credentials", return_value=mock_creds):
        with patch("hermes_antigravity.client.catalog.resolve_runtime_model_sync", return_value="gemini-3.7-flash-tiered"):
            with patch("hermes_antigravity.client.native.refresh_access_token", side_effect=mock_refresh_fail):
                with pytest.raises(Exception) as exc_info:
                    client.chat.completions.create(model="gemini-3.7-flash", messages=[{"role":"user", "content":"hi"}], stream=False)
            
            try:
                from hermes_cli.auth import AuthError
            except ImportError:
                # The fallback AuthError inside the client module
                AuthError = type(exc_info.value)
                
            assert isinstance(exc_info.value, AuthError)
            assert exc_info.value.provider == "antigravity"
            assert exc_info.value.code == "refresh_failed"


def test_endpoint_fallback(mock_creds):
    client = AntigravityClient()
    
    mock_resp_503 = MagicMock()
    mock_resp_503.status_code = 503
    
    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    mock_resp_200.iter_lines.return_value = iter([
        'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}], "finishReason": "STOP"}}',
        "data: [DONE]"
    ])
    
    call_count = {"send": 0}
    def fake_send(*args, **kwargs):
        call_count["send"] += 1
        if call_count["send"] <= 5: # Fails MAX_RETRIES_PER_ENDPOINT times on first endpoint
            return mock_resp_503
        return mock_resp_200

    mock_http_client = MagicMock()
    mock_http_client.build_request.return_value = "req"
    mock_http_client.send.side_effect = fake_send
    client._http_client = mock_http_client
    
    with patch("hermes_antigravity.client.native.BASE_BACKOFF_SECS", 0.001):
        with patch("hermes_antigravity.client.native.ENDPOINT_FALLBACKS", ["http://end1", "http://end2"]):
            with patch("hermes_antigravity.client.native.load_credentials", return_value=mock_creds):
                with patch("hermes_antigravity.client.catalog.resolve_runtime_model_sync", return_value="gemini-3.7-flash-tiered"):
                    # No AuthError, should succeed on the second endpoint
                    resp = client.chat.completions.create(model="gemini-3.7-flash", messages=[{"role":"user", "content":"hi"}], stream=False)
                    assert call_count["send"] == 6 # 5 attempts on first, 1 attempt on second
                assert resp.choices[0].message.content == "Hello"


def test_fetch_models_fallback_catalog():
    from hermes_antigravity.__init__ import AntigravityProviderProfile
    from hermes_antigravity.models.models import FALLBACK_MODELS
    
    profile = AntigravityProviderProfile(name="antigravity")
    
    with patch("hermes_antigravity.__init__.fetch_available_models_sync", side_effect=Exception("Failed")):
        models = profile.fetch_models()
        assert set(models) == set(FALLBACK_MODELS)
        assert "gemini-3.7-flash" in models

