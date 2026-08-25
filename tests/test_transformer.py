"""Unit tests for OpenAI to Google Cloud Code Assist payload translation and SSE stream decoding."""

import json
import pytest

from hermes_antigravity.stream.transformer import (
    build_gemini_request,
    convert_openai_tools_to_gemini,
    transform_google_sse_to_openai,
)


def test_build_gemini_request_simple_message():
    openai_req = {
        "model": "gemini-3.7-flash",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, world!"},
        ],
        "temperature": 0.7,
        "max_tokens": 1000,
        "reasoning_effort": "high",
    }
    runtime_model, envelope = build_gemini_request(openai_req, "proj-123")

    assert runtime_model == "gemini-3.7-flash-tiered"
    assert envelope["project"] == "proj-123"
    assert envelope["model"] == "gemini-3.7-flash-tiered"

    req_body = envelope["request"]
    assert req_body["generationConfig"]["temperature"] == 0.7
    assert req_body["generationConfig"]["maxOutputTokens"] == 1000
    assert req_body["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "HIGH"

    # Verify system instruction contains user system prompt
    sys_texts = [p["text"] for p in req_body["systemInstruction"]["parts"]]
    assert any("helpful assistant" in t for t in sys_texts)

    # Verify user message
    assert len(req_body["contents"]) == 1
    assert req_body["contents"][0]["role"] == "user"
    assert req_body["contents"][0]["parts"][0]["text"] == "Hello, world!"


def test_build_gemini_request_tool_calling():
    openai_req = {
        "model": "claude-sonnet-4-6",
        "messages": [
            {"role": "user", "content": "What is the weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Tokyo"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "name": "get_weather",
                "content": '{"temp": "22C"}',
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Fetch weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ],
    }
    runtime_model, envelope = build_gemini_request(openai_req, "proj-abc")
    req_body = envelope["request"]

    # Verify tools converted
    assert "tools" in req_body
    decl = req_body["tools"][0]["functionDeclarations"][0]
    assert decl["name"] == "get_weather"

    # Verify conversation history roles
    contents = req_body["contents"]
    assert len(contents) == 3
    assert contents[0]["role"] == "user"
    assert contents[1]["role"] == "model"
    assert "functionCall" in contents[1]["parts"][0]
    assert contents[1]["parts"][0]["functionCall"]["name"] == "get_weather"

    assert contents[2]["role"] == "user"
    assert "functionResponse" in contents[2]["parts"][0]
    assert contents[2]["parts"][0]["functionResponse"]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_transform_google_sse_to_openai_stream():
    google_sse_lines = [
        'data: {"response": {"candidates": [{"content": {"parts": [{"thought": true, "text": "Thinking..."}]}}]}}',
        'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Hello "}]}}]}}',
        'data: {"response": {"candidates": [{"content": {"parts": [{"text": "there!"}]}, "finishReason": "STOP"}]}}',
        "data: [DONE]",
    ]

    async def fake_stream():
        for line in google_sse_lines:
            yield line

    output_chunks = []
    async for chunk_str in transform_google_sse_to_openai(fake_stream(), "gemini-3.7-flash"):
        output_chunks.append(chunk_str)

    assert len(output_chunks) >= 4
    assert output_chunks[-1] == "data: [DONE]\n\n"

    # Verify reasoning chunk
    c1 = json.loads(output_chunks[0].replace("data: ", "").strip())
    assert c1["choices"][0]["delta"]["reasoning_content"] == "Thinking..."

    # Verify text chunks
    c2 = json.loads(output_chunks[1].replace("data: ", "").strip())
    assert c2["choices"][0]["delta"]["content"] == "Hello "

    c3 = json.loads(output_chunks[2].replace("data: ", "").strip())
    assert c3["choices"][0]["delta"]["content"] == "there!"
