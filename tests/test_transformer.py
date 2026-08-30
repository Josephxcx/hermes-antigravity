"""Unit tests for OpenAI to Google Cloud Code Assist payload translation and SSE stream decoding."""

import json
import pytest

from hermes_antigravity.stream.transformer import (
    aggregate_google_sse_to_openai_response,
    build_gemini_request,
    convert_openai_tools_to_gemini,
    inline_and_sanitize_schema,
    record_thought_signature,
    retrieve_thought_signature,
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


def test_build_gemini_request_tool_calling_implicit_name_and_batching():
    openai_req = {
        "model": "gemini-3.7-flash",
        "messages": [
            {"role": "user", "content": "Fetch data"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "query_alpha", "arguments": "{}"},
                    },
                    {
                        "id": "call_b",
                        "type": "function",
                        "function": {"name": "query_beta", "arguments": "{}"},
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_a",
                "content": "res_a",
            },
            {
                "role": "tool",
                "tool_call_id": "call_b",
                "content": "res_b",
            },
        ],
    }
    runtime_model, envelope = build_gemini_request(openai_req, "proj-abc")
    contents = envelope["request"]["contents"]

    assert len(contents) == 3
    assert contents[1]["role"] == "model"
    assert len(contents[1]["parts"]) == 2

    assert contents[2]["role"] == "user"
    # Both tool responses should be grouped in the single user turn and have correct tool names
    assert len(contents[2]["parts"]) == 2
    assert contents[2]["parts"][0]["functionResponse"]["name"] == "query_alpha"
    assert contents[2]["parts"][1]["functionResponse"]["name"] == "query_beta"


def test_thought_signature_injection():
    record_thought_signature("test_sig_abc123", tool_id="call_999", fn_name="test_tool", args={"x": 1})

    openai_req = {
        "model": "gemini-3.1-pro",
        "messages": [
            {"role": "user", "content": "Run test tool"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_999",
                        "type": "function",
                        "function": {
                            "name": "test_tool",
                            "arguments": '{"x": 1}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_999",
                "name": "test_tool",
                "content": '{"result": "ok"}',
            },
        ],
    }
    runtime_model, envelope = build_gemini_request(openai_req, "proj-abc")
    model_turn = envelope["request"]["contents"][1]
    assert "thoughtSignature" in model_turn["parts"][0]
    assert model_turn["parts"][0]["thoughtSignature"] == "test_sig_abc123"


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

@pytest.mark.asyncio
async def test_transform_google_sse_with_usage_metadata():
    google_sse_lines = [
        'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Hello world"}]}}], "usageMetadata": {"promptTokenCount": 15, "candidatesTokenCount": 8, "totalTokenCount": 23}}}',
        'data: {"response": {"candidates": [{"content": {"parts": []}, "finishReason": "STOP"}]}}',
        "data: [DONE]",
    ]

    async def fake_stream():
        for line in google_sse_lines:
            yield line

    output_chunks = []
    async for chunk_str in transform_google_sse_to_openai(fake_stream(), "gemini-3.7-flash"):
        output_chunks.append(chunk_str)

    finish_chunk = json.loads(output_chunks[-2].replace("data: ", "").strip())
    assert "usage" in finish_chunk
    assert finish_chunk["usage"]["prompt_tokens"] == 15
    assert finish_chunk["usage"]["completion_tokens"] == 8
    assert finish_chunk["usage"]["total_tokens"] == 23


@pytest.mark.asyncio
async def test_aggregate_google_sse_to_openai_response():
    google_sse_lines = [
        'data: {"response": {"candidates": [{"content": {"parts": [{"thought": true, "text": "Plan: say hello."}]}}]}}',
        'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Hello!"}]}}], "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}}}',
        'data: {"response": {"candidates": [{"content": {"parts": []}, "finishReason": "STOP"}]}}',
        "data: [DONE]",
    ]

    async def fake_stream():
        for line in google_sse_lines:
            yield line

    response = await aggregate_google_sse_to_openai_response(fake_stream(), "gemini-3.7-flash")

    assert response["object"] == "chat.completion"
    assert response["model"] == "gemini-3.7-flash"
    assert response["usage"]["total_tokens"] == 15
    choice = response["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "Hello!"
    assert choice["message"]["reasoning_content"] == "Plan: say hello."



def test_inline_and_sanitize_schema_defs_and_refs():
    raw_schema = {
        "type": "object",
        "$defs": {
            "UserDetail": {
                "type": "object",
                "properties": {
                    "age": {"type": "integer"},
                    "city": {"type": "string"},
                },
                "required": ["age"],
            }
        },
        "properties": {
            "name": {"type": "string"},
            "details": {"$ref": "#/$defs/UserDetail"},
        },
        "required": ["name", "details"],
    }

    sanitized = inline_and_sanitize_schema(raw_schema)

    # Verify $defs removed
    assert "$defs" not in sanitized
    assert sanitized["type"] == "object"
    assert "name" in sanitized["properties"]
    assert "details" in sanitized["properties"]

    # Verify $ref inlined
    details_prop = sanitized["properties"]["details"]
    assert "$ref" not in details_prop
    assert details_prop["type"] == "object"
    assert "age" in details_prop["properties"]
    assert details_prop["properties"]["age"]["type"] == "integer"


def test_inline_and_sanitize_schema_anyof_nullable():
    raw_schema = {
        "type": "object",
        "properties": {
            "query": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "null"},
                ]
            }
        },
    }

    sanitized = inline_and_sanitize_schema(raw_schema)
    query_prop = sanitized["properties"]["query"]
    assert query_prop["type"] == "string"
    assert query_prop.get("nullable") is True


def test_inline_and_sanitize_schema_allof_merging():
    raw_schema = {
        "allOf": [
            {
                "type": "object",
                "properties": {"foo": {"type": "string"}},
                "required": ["foo"],
            },
            {
                "type": "object",
                "properties": {"bar": {"type": "integer"}},
                "required": ["bar"],
            },
        ]
    }

    sanitized = inline_and_sanitize_schema(raw_schema)
    assert sanitized["type"] == "object"
    assert "foo" in sanitized["properties"]
    assert "bar" in sanitized["properties"]
    assert "foo" in sanitized["required"]
    assert "bar" in sanitized["required"]


def test_inline_and_sanitize_schema_strips_unsupported_validation_keys():
    raw_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "questions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "description": "List of questions",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "minLength": 1, "maxLength": 500},
                        "choices": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 4,
                        },
                        "score": {"type": "number", "minimum": 0.0, "maximum": 100.0},
                    },
                    "required": ["question"],
                },
            }
        },
        "required": ["questions"],
    }

    sanitized = inline_and_sanitize_schema(raw_schema)

    assert sanitized["type"] == "object"
    assert "additionalProperties" not in sanitized
    assert "questions" in sanitized["properties"]

    q_prop = sanitized["properties"]["questions"]
    assert q_prop["type"] == "array"
    assert "minItems" not in q_prop
    assert "maxItems" not in q_prop
    assert q_prop["description"] == "List of questions"

    item_schema = q_prop["items"]
    assert item_schema["type"] == "object"
    assert "question" in item_schema["properties"]
    assert "choices" in item_schema["properties"]
    assert "score" in item_schema["properties"]

    # Verify nested properties do not contain unsupported keys
    assert "minLength" not in item_schema["properties"]["question"]
    assert "maxLength" not in item_schema["properties"]["question"]
    assert "minItems" not in item_schema["properties"]["choices"]
    assert "maxItems" not in item_schema["properties"]["choices"]
    assert "minimum" not in item_schema["properties"]["score"]
    assert "maximum" not in item_schema["properties"]["score"]


def test_inline_and_sanitize_schema_properties_mapping_integrity():
    raw_schema = {
        "type": "object",
        "properties": {
            "preset": {
                "type": "string",
                "description": "Layout preset",
            }
        },
        "required": ["preset"],
    }

    sanitized = inline_and_sanitize_schema(raw_schema)

    # The properties map must strictly only contain the property names ('preset'),
    # not internal schema keywords like 'type' or 'properties'.
    assert set(sanitized["properties"].keys()) == {"preset"}
    assert sanitized["properties"]["preset"]["type"] == "string"
    assert sanitized["properties"]["preset"]["description"] == "Layout preset"


