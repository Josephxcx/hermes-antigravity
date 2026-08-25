"""Bidirectional transformer between OpenAI Chat Completions API and Google Cloud Code Assist format."""

from __future__ import annotations

import collections
import json
import logging
import re
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from hermes_antigravity.models.models import (
    get_max_output_tokens,
    get_runtime_model_id,
)

logger = logging.getLogger(__name__)

ANTIGRAVITY_SYSTEM_INSTRUCTION = (
    "You are Antigravity, a powerful agentic AI coding assistant designed by Google DeepMind. "
    "You are pair programming with a user to solve coding tasks. Be concise, practical, and tool-aware."
)

ANTIGRAVITY_NO_PREAMBLE_INSTRUCTION = (
    'CRITICAL: NEVER output rule checks, formatting guidelines, constraint checklists (e.g. "No emdashes"), '
    "or your thinking/personality preambles in the final response. Output only the final response."
)

_tool_call_counter = 0

# Cache for cryptographic thoughtSignatures returned by Google for tool calls
_SIGNATURE_CACHE_SIZE = 2000
_signature_by_id: collections.OrderedDict[str, str] = collections.OrderedDict()
_signature_by_fn_name: collections.OrderedDict[str, str] = collections.OrderedDict()
_last_thought_signature: Optional[str] = None


def record_thought_signature(
    signature: str,
    tool_id: Optional[str] = None,
    fn_name: Optional[str] = None,
    args: Optional[Dict[str, Any]] = None,
) -> None:
    global _last_thought_signature
    if not signature:
        return
    _last_thought_signature = signature

    if tool_id:
        _signature_by_id[tool_id] = signature
        if len(_signature_by_id) > _SIGNATURE_CACHE_SIZE:
            _signature_by_id.popitem(last=False)

    if fn_name:
        _signature_by_fn_name[fn_name] = signature
        if len(_signature_by_fn_name) > _SIGNATURE_CACHE_SIZE:
            _signature_by_fn_name.popitem(last=False)


def retrieve_thought_signature(
    tool_id: Optional[str] = None,
    fn_name: Optional[str] = None,
    args: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    if tool_id and tool_id in _signature_by_id:
        return _signature_by_id[tool_id]

    if fn_name and fn_name in _signature_by_fn_name:
        return _signature_by_fn_name[fn_name]

    return _last_thought_signature


def sanitize_tool_call_id(tool_id: Optional[str], fallback_name: str = "tool") -> str:
    global _tool_call_counter
    if tool_id:
        cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_id)[:64]
        if cleaned:
            return cleaned
    _tool_call_counter += 1
    return f"{fallback_name}_{int(time.time())}_{_tool_call_counter}"


def needs_tool_call_id(model_id: str, runtime_model: str) -> bool:
    mid = model_id.lower()
    rm = runtime_model.lower()
    return (
        mid.startswith("claude-")
        or mid.startswith("gpt-oss-")
        or rm.startswith("claude-")
        or rm.startswith("gpt-oss-")
        or "gemini" in mid
        or "gemini" in rm
    )


def convert_openai_tools_to_gemini(
    tools: Optional[List[Dict[str, Any]]],
    use_legacy_parameters: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None

    declarations = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        name = func.get("name", "")
        if not name:
            continue
        desc = func.get("description", "")
        params = func.get("parameters", {"type": "object", "properties": {}})

        decl: Dict[str, Any] = {"name": name, "description": desc}
        if use_legacy_parameters:
            decl["parameters"] = params
        else:
            decl["parametersJsonSchema"] = params
        declarations.append(decl)

    if not declarations:
        return None
    return [{"functionDeclarations": declarations}]


def build_gemini_request(
    openai_req: Dict[str, Any],
    project_id: str,
) -> Tuple[str, Dict[str, Any]]:
    """Translates an OpenAI /v1/chat/completions payload to (runtime_model_id, envelope_dict)."""
    raw_model = openai_req.get("model", "gemini-3.1-pro")
    reasoning_effort = openai_req.get("reasoning_effort") or openai_req.get("reasoning", {}).get("effort")
    runtime_model = get_runtime_model_id(raw_model, reasoning_effort)

    contents: List[Dict[str, Any]] = []
    system_instructions: List[Dict[str, str]] = [
        {"text": ANTIGRAVITY_SYSTEM_INSTRUCTION},
        {"text": ANTIGRAVITY_NO_PREAMBLE_INSTRUCTION},
    ]

    messages = openai_req.get("messages", [])
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role in ("system", "developer"):
            if isinstance(content, str) and content.strip():
                system_instructions.append({"text": content.strip()})
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        system_instructions.append({"text": part.get("text", "")})
            continue

        if role == "user":
            parts: List[Dict[str, Any]] = []
            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            parts.append({"text": part.get("text", "")})
                        elif part.get("type") == "image_url":
                            img_url = part.get("image_url", {}).get("url", "")
                            if img_url.startswith("data:"):
                                match = re.match(r"^data:([^;]+);base64,(.+)$", img_url, re.DOTALL)
                                if match:
                                    parts.append({
                                        "inlineData": {
                                            "mimeType": match.group(1),
                                            "data": match.group(2).strip(),
                                        }
                                    })
            if parts:
                contents.append({"role": "user", "parts": parts})

        elif role == "assistant":
            parts = []
            # Note: For Gemini tool calling, skip separate thought part if functionCall is present
            # to keep thoughtSignature directly attached to functionCall as required by Google.
            tool_calls = msg.get("tool_calls", [])
            reasoning = msg.get("reasoning_content")
            if reasoning and not tool_calls:
                parts.append({"thought": True, "text": reasoning})

            if isinstance(content, str) and content.strip():
                parts.append({"text": content})

            for tc in tool_calls:
                func = tc.get("function", {})
                fn_name = func.get("name", "")
                args_raw = func.get("arguments", "{}")
                if isinstance(args_raw, str):
                    try:
                        args = json.loads(args_raw)
                    except Exception:
                        args = {"raw_args": args_raw}
                else:
                    args = args_raw or {}

                call_id = tc.get("id") or sanitize_tool_call_id(None, fn_name)
                fn_part: Dict[str, Any] = {
                    "functionCall": {
                        "name": fn_name,
                        "args": args,
                    }
                }
                if needs_tool_call_id(raw_model, runtime_model):
                    fn_part["functionCall"]["id"] = call_id

                # Attach cryptographic thoughtSignature if available
                sig = (
                    tc.get("thought_signature")
                    or tc.get("thoughtSignature")
                    or retrieve_thought_signature(call_id, fn_name, args)
                )
                if sig:
                    fn_part["thoughtSignature"] = sig

                parts.append(fn_part)

            if parts:
                contents.append({"role": "model", "parts": parts})

        elif role == "tool":
            tool_call_id = msg.get("tool_call_id")
            name = msg.get("name") or "tool"
            tool_resp_str = content if isinstance(content, str) else json.dumps(content or {})
            resp_part: Dict[str, Any] = {
                "functionResponse": {
                    "name": name,
                    "response": {"output": tool_resp_str},
                }
            }
            if needs_tool_call_id(raw_model, runtime_model):
                resp_part["functionResponse"]["id"] = sanitize_tool_call_id(tool_call_id, name)
            contents.append({"role": "user", "parts": [resp_part]})

    # Ensure conversation starts with user turn for Google API compatibility
    if contents and contents[0].get("role") == "model":
        contents.insert(0, {"role": "user", "parts": [{"text": "Hello"}]})
    elif not contents:
        contents = [{"role": "user", "parts": [{"text": "Hello"}]}]

    request_body: Dict[str, Any] = {
        "contents": contents,
        "systemInstruction": {
            "role": "user",
            "parts": system_instructions,
        },
    }

    # Generation config
    gen_config: Dict[str, Any] = {}
    if "temperature" in openai_req:
        gen_config["temperature"] = openai_req["temperature"]

    max_tokens = openai_req.get("max_tokens") or openai_req.get("max_completion_tokens")
    max_allowed = get_max_output_tokens(raw_model, runtime_model)
    if max_tokens is not None:
        gen_config["maxOutputTokens"] = min(int(max_tokens), max_allowed)
    else:
        gen_config["maxOutputTokens"] = max_allowed

    if runtime_model == "gemini-3.7-flash-tiered":
        effort = (reasoning_effort or "medium").lower()
        level = "HIGH" if effort in ("high", "xhigh") else "MEDIUM" if effort == "medium" else "LOW"
        gen_config["thinkingConfig"] = {"thinkingLevel": level}

    if gen_config:
        request_body["generationConfig"] = gen_config

    # Tools
    use_legacy = raw_model.lower().startswith("claude-") or raw_model.lower().startswith("gpt-oss-")
    tools = convert_openai_tools_to_gemini(openai_req.get("tools"), use_legacy_parameters=use_legacy)
    if tools:
        request_body["tools"] = tools

    envelope = {
        "project": project_id,
        "model": runtime_model,
        "request": request_body,
    }
    return runtime_model, envelope


async def transform_google_sse_to_openai(
    response_stream: AsyncGenerator[str, None],
    model_id: str,
) -> AsyncGenerator[str, None]:
    """Decodes Google Cloud Code Assist SSE lines and yields standard OpenAI SSE data chunks."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())
    tool_idx = 0
    current_thought_sig: Optional[str] = None
    active_tool_ids: Dict[int, str] = {}

    async for line in response_stream:
        line = line.strip()
        if not line:
            continue
        if not line.startswith("data:"):
            continue

        json_str = line[5:].strip()
        if not json_str or json_str == "[DONE]":
            continue

        try:
            chunk = json.loads(json_str)
        except Exception:
            continue

        if "error" in chunk:
            err = chunk["error"]
            err_msg = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(f"Google Cloud Code Assist API Error: {err_msg}")

        resp_data = chunk.get("response", chunk)
        candidates = resp_data.get("candidates", [])
        if not candidates:
            continue

        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        for part in parts:
            sig = part.get("thoughtSignature") or part.get("thought_signature")
            if sig:
                current_thought_sig = sig
                record_thought_signature(sig)

            # 1. Thinking / Reasoning text
            if part.get("thought") is True:
                thought_text = part.get("text", "")
                if thought_text:
                    openai_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "reasoning_content": thought_text,
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(openai_chunk)}\n\n"

            # 2. Standard content text
            elif "text" in part:
                text = part.get("text", "")
                if text:
                    openai_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": text,
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(openai_chunk)}\n\n"

            # 3. Tool call / Function call
            elif "functionCall" in part:
                fc = part["functionCall"]
                fn_name = fc.get("name", "")
                args = fc.get("args", {})
                
                # Maintain stable ID per tool call index in this turn
                if tool_idx not in active_tool_ids:
                    active_tool_ids[tool_idx] = fc.get("id") or sanitize_tool_call_id(None, fn_name)
                call_id = active_tool_ids[tool_idx]

                args_str = json.dumps(args) if isinstance(args, dict) else str(args)

                # Record the signature with this specific call_id and function
                if current_thought_sig:
                    record_thought_signature(current_thought_sig, tool_id=call_id, fn_name=fn_name, args=args if isinstance(args, dict) else None)

                openai_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": tool_idx,
                                        "id": call_id,
                                        "type": "function",
                                        "function": {
                                            "name": fn_name,
                                            "arguments": args_str,
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
                tool_idx += 1
                yield f"data: {json.dumps(openai_chunk)}\n\n"

        # Check finish reason
        finish_reason = candidate.get("finishReason")
        if finish_reason:
            fr_mapped = "stop"
            if finish_reason == "MAX_TOKENS":
                fr_mapped = "length"
            elif finish_reason in ("TOOL_USE", "FUNCTION_CALL"):
                fr_mapped = "tool_calls"

            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": fr_mapped,
                    }
                ],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"

    yield "data: [DONE]\n\n"
