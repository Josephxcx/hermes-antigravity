"""Live authenticated smoke test for hermes-antigravity-pi-port including multi-turn tool calling."""

import asyncio
import json
import logging
import sys

import httpx

from hermes_antigravity.auth.credentials import load_credentials
from hermes_antigravity.proxy.server import ensure_proxy_running

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def main():
    logger.info("=== Starting Authenticated Live Smoke Test ===")

    # 1. Check credentials
    creds = load_credentials()
    if not creds or not creds.access_token:
        logger.error("❌ No Antigravity credentials found in ~/.hermes/auth.json or ~/.pi/agent/auth.json")
        sys.exit(1)

    logger.info("✅ Found credentials for account: %s (expires: %s)", creds.email, creds.expires_at)

    # 2. Start in-process proxy
    base_url = ensure_proxy_running()
    logger.info("✅ In-process proxy running at %s", base_url)

    # 3. Test simple completion with streaming content
    logger.info("\n--- Testing Model Completion (gemini-3.1-pro) ---")
    req_payload = {
        "model": "gemini-3.1-pro",
        "messages": [
            {"role": "user", "content": "Reply with exactly three words: Antigravity is live"}
        ],
        "temperature": 0.2,
        "max_tokens": 100,
        "stream": True,
    }

    full_content = ""
    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream("POST", f"{base_url}/chat/completions", json=req_payload) as resp:
            assert resp.status_code == 200
            async for line in resp.aiter_lines():
                if line.startswith("data:"):
                    chunk_str = line[5:].strip()
                    if chunk_str and chunk_str != "[DONE]":
                        try:
                            data = json.loads(chunk_str)
                            delta = data["choices"][0]["delta"]
                            if "content" in delta:
                                full_content += delta["content"]
                        except Exception:
                            pass

    logger.info("Received output: %r", full_content.strip())
    assert len(full_content.strip()) > 0
    logger.info("✅ Live model completion test PASSED")

    # 4. Test Multi-Turn Tool Calling with thoughtSignature roundtrip
    logger.info("\n--- Testing Multi-Turn Tool Calling (gemini-3.1-pro) ---")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city name"}
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    messages = [
        {"role": "user", "content": "What is the weather in Tokyo? Use get_weather tool."}
    ]

    # Turn 1: Model should call get_weather
    tool_calls_received = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream(
            "POST",
            f"{base_url}/chat/completions",
            json={"model": "gemini-3.1-pro", "messages": messages, "tools": tools, "stream": True},
        ) as resp:
            assert resp.status_code == 200
            async for line in resp.aiter_lines():
                if line.startswith("data:"):
                    chunk_str = line[5:].strip()
                    if chunk_str and chunk_str != "[DONE]":
                        try:
                            data = json.loads(chunk_str)
                            delta = data["choices"][0]["delta"]
                            if "tool_calls" in delta:
                                tool_calls_received.extend(delta["tool_calls"])
                        except Exception:
                            pass

    assert len(tool_calls_received) > 0, "Expected tool call in turn 1"
    tc = tool_calls_received[0]
    call_id = tc["id"]
    fn_name = tc["function"]["name"]
    fn_args = tc["function"]["arguments"]
    logger.info("Turn 1 Tool Call: id=%s, name=%s, args=%s", call_id, fn_name, fn_args)

    # Turn 2: Provide tool output back to model (this tests thoughtSignature preservation!)
    messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [tc],
    })
    messages.append({
        "role": "tool",
        "tool_call_id": call_id,
        "name": fn_name,
        "content": json.dumps({"temp": "21°C", "condition": "Sunny"}),
    })

    final_turn_output = ""
    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream(
            "POST",
            f"{base_url}/chat/completions",
            json={"model": "gemini-3.1-pro", "messages": messages, "tools": tools, "stream": True},
        ) as resp:
            assert resp.status_code == 200
            async for line in resp.aiter_lines():
                if line.startswith("data:"):
                    chunk_str = line[5:].strip()
                    if chunk_str and chunk_str != "[DONE]":
                        try:
                            data = json.loads(chunk_str)
                            delta = data["choices"][0]["delta"]
                            if "content" in delta:
                                final_turn_output += delta["content"]
                        except Exception:
                            pass

    logger.info("Turn 2 Response from model: %r", final_turn_output.strip())
    assert len(final_turn_output.strip()) > 0
    assert "21" in final_turn_output or "sunny" in final_turn_output.lower() or "tokyo" in final_turn_output.lower()
    logger.info("✅ Multi-turn tool calling with thoughtSignature verified successfully!")

    logger.info("\n🎉 All live authenticated smoke tests PASSED!")


if __name__ == "__main__":
    asyncio.run(main())
