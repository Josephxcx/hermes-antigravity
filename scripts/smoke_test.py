"""Live authenticated smoke test for hermes-antigravity-pi-port."""

import asyncio
import json
import logging
import sys

import httpx

from hermes_antigravity.auth.credentials import load_credentials
from hermes_antigravity.proxy.server import get_or_start_proxy

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
    base_url = await get_or_start_proxy()
    logger.info("✅ In-process proxy running at %s", base_url)

    # 3. Test simple completion with streaming reasoning + content
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
    reasoning_content = ""
    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream("POST", f"{base_url}/chat/completions", json=req_payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                logger.error("❌ Request failed with status %d: %s", resp.status_code, body.decode())
                sys.exit(1)

            async for line in resp.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                chunk_str = line[5:].strip()
                if not chunk_str or chunk_str == "[DONE]":
                    continue
                try:
                    data = json.loads(chunk_str)
                    delta = data["choices"][0]["delta"]
                    if "reasoning_content" in delta:
                        reasoning_content += delta["reasoning_content"]
                    if "content" in delta:
                        full_content += delta["content"]
                except Exception:
                    pass

    logger.info("Received reasoning (%d chars)", len(reasoning_content))
    logger.info("Received output: %r", full_content.strip())
    if "antigravity" in full_content.lower() or len(full_content.strip()) > 0:
        logger.info("✅ Live model completion test PASSED")
    else:
        logger.error("❌ Completion output was empty")
        sys.exit(1)

    # 4. Test tool calling
    logger.info("\n--- Testing Live Tool Calling (gemini-3.1-pro) ---")
    tool_req = {
        "model": "gemini-3.1-pro",
        "messages": [
            {"role": "user", "content": "What is the weather in Tokyo right now? Call get_weather tool."}
        ],
        "tools": [
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
        ],
        "temperature": 0.0,
        "stream": True,
    }

    tool_calls_received = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream("POST", f"{base_url}/chat/completions", json=tool_req) as resp:
            assert resp.status_code == 200
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                chunk_str = line[5:].strip()
                if not chunk_str or chunk_str == "[DONE]":
                    continue
                try:
                    data = json.loads(chunk_str)
                    delta = data["choices"][0]["delta"]
                    if "tool_calls" in delta:
                        tool_calls_received.extend(delta["tool_calls"])
                except Exception:
                    pass

    logger.info("Received tool calls: %s", tool_calls_received)
    assert len(tool_calls_received) > 0, "No tool calls received"
    fn_name = tool_calls_received[0]["function"]["name"]
    logger.info("Tool called: %s with args: %s", fn_name, tool_calls_received[0]["function"]["arguments"])
    assert fn_name == "get_weather"
    logger.info("✅ Live tool calling test PASSED")

    logger.info("\n🎉 All live authenticated smoke tests PASSED successfully!")


if __name__ == "__main__":
    asyncio.run(main())
