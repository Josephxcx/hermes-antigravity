"""Bidirectional transformer between OpenAI Chat Completions API and Google Cloud Code Assist format."""

from __future__ import annotations

import collections
import copy
import json
import logging
import re
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple

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


def inline_and_sanitize_schema(schema: Any) -> Dict[str, Any]:
    """Recursively dereferences $defs/definitions/$ref and cleans JSON Schemas

    to strictly comply with Google Cloud Code Assist Gemini OpenAPI schema validator.
    """
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    schema_copy = copy.deepcopy(schema)

    # 1. Collect all local definitions
    defs: Dict[str, Any] = {}
    for def_key in ("$defs", "definitions"):
        if def_key in schema_copy and isinstance(schema_copy[def_key], dict):
            defs.update(schema_copy[def_key])

    def resolve_ref(ref_str: str) -> Optional[Dict[str, Any]]:
        # e.g., #/$defs/MyType or #/definitions/MyType
        parts = [p for p in ref_str.split("/") if p and p != "#"]
        if not parts:
            return None
        target_name = parts[-1]
        if target_name in defs:
            return defs[target_name]
        return None

    def clean_node(node: Any, seen_refs: Set[str]) -> Any:
        if isinstance(node, list):
            return [clean_node(item, seen_refs) for item in node]

        if not isinstance(node, dict):
            return node

        # Handle $ref
        if "$ref" in node and isinstance(node["$ref"], str):
            ref_target = node["$ref"]
            if ref_target in seen_refs:
                return {"type": "object", "description": f"Recursive ref to {ref_target}"}

            resolved = resolve_ref(ref_target)
            if resolved is not None:
                new_seen = seen_refs | {ref_target}
                merged = copy.deepcopy(resolved)
                for k, v in node.items():
                    if k != "$ref":
                        merged[k] = v
                return clean_node(merged, new_seen)
            else:
                return {"type": "string"}

        # Handle allOf (merge sub-schemas)
        if "allOf" in node and isinstance(node["allOf"], list):
            merged_all: Dict[str, Any] = {}
            for sub in node["allOf"]:
                cleaned_sub = clean_node(sub, seen_refs)
                if isinstance(cleaned_sub, dict):
                    if "properties" in cleaned_sub:
                        merged_all.setdefault("properties", {}).update(cleaned_sub["properties"])
                    if "required" in cleaned_sub and isinstance(cleaned_sub["required"], list):
                        merged_all.setdefault("required", []).extend(cleaned_sub["required"])
                    for k, v in cleaned_sub.items():
                        if k not in ("properties", "required"):
                            merged_all[k] = v
            for k, v in node.items():
                if k != "allOf":
                    merged_all[k] = v
            return clean_node(merged_all, seen_refs)

        # Handle anyOf / oneOf
        if ("anyOf" in node and isinstance(node["anyOf"], list)) or ("oneOf" in node and isinstance(node["oneOf"], list)):
            variants = node.get("anyOf") or node.get("oneOf") or []
            has_null = any(isinstance(v, dict) and v.get("type") == "null" for v in variants)
            non_null_raw = [v for v in variants if isinstance(v, dict) and v.get("type") != "null"]
            cleaned_variants = [clean_node(v, seen_refs) for v in non_null_raw]

            if len(cleaned_variants) >= 1:
                primary = copy.deepcopy(cleaned_variants[0])
                if has_null:
                    primary["nullable"] = True
                for k, v in node.items():
                    if k not in ("anyOf", "oneOf"):
                        primary[k] = clean_node(v, seen_refs)
                return clean_node(primary, seen_refs)
            else:
                return {"type": "string", "nullable": True}

        cleaned: Dict[str, Any] = {}

        # Copy and clean attributes
        for key, val in node.items():
            # Skip disallowed Gemini OpenAPI meta keys
            if key in ("$defs", "definitions", "$schema", "$id", "title", "additionalProperties", "default"):
                continue
            cleaned[key] = clean_node(val, seen_refs)

        # Handle array of types like type: ["string", "null"]
        if "type" in cleaned and isinstance(cleaned["type"], list):
            types = [t for t in cleaned["type"] if t != "null"]
            if "null" in cleaned["type"]:
                cleaned["nullable"] = True
            cleaned["type"] = str(types[0]) if types else "string"

        # Ensure type is present and valid
        if "type" not in cleaned:
            if "properties" in cleaned:
                cleaned["type"] = "object"
            elif "items" in cleaned:
                cleaned["type"] = "array"
            elif "enum" in cleaned:
                cleaned["type"] = "string"
            else:
                cleaned["type"] = "object"
        elif isinstance(cleaned.get("type"), str):
            cleaned["type"] = cleaned["type"].lower()
            if cleaned["type"] not in ("string", "number", "integer", "boolean", "array", "object"):
                cleaned["type"] = "string"

        # If type is array, items MUST be present and must be a dict
        if cleaned.get("type") == "array":
            if "items" not in cleaned or not isinstance(cleaned["items"], dict):
                cleaned["items"] = {"type": "string"}

        # If type is object, properties should be present and every value must be an object
        if cleaned.get("type") == "object":
            if "properties" not in cleaned or not isinstance(cleaned["properties"], dict):
                cleaned["properties"] = {}
            else:
                clean_props: Dict[str, Any] = {}
                for prop_k, prop_v in cleaned["properties"].items():
                    if isinstance(prop_v, dict):
                        clean_props[prop_k] = prop_v
                    elif isinstance(prop_v, str):
                        clean_props[prop_k] = {"type": prop_v.lower() if prop_v.lower() in ("string", "number", "integer", "boolean", "array", "object") else "string"}
                    else:
                        clean_props[prop_k] = {"type": "string"}
                cleaned["properties"] = clean_props

        # Clean required list
        if "required" in cleaned and isinstance(cleaned["required"], list):
            valid_required = []
            props = cleaned.get("properties", {})
            for r in cleaned["required"]:
                if isinstance(r, str) and (r in props or not props):
                    if r not in valid_required:
                        valid_required.append(r)
            cleaned["required"] = valid_required
            if not valid_required:
                cleaned.pop("required", None)

        return cleaned

    result = clean_node(schema_copy, set())
    if not isinstance(result, dict):
        return {"type": "object", "properties": {}}
    return result


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
        raw_params = func.get("parameters", {"type": "object", "properties": {}})
        sanitized_params = inline_and_sanitize_schema(raw_params)

        decl: Dict[str, Any] = {"name": name, "description": desc}
        if use_legacy_parameters:
            decl["parameters"] = sanitized_params
        else:
            decl["parametersJsonSchema"] = sanitized_params
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

    tool_call_id_to_name: Dict[str, str] = {}

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
                if call_id and fn_name:
                    tool_call_id_to_name[call_id] = fn_name
                    sanitized_id = sanitize_tool_call_id(call_id, fn_name)
                    tool_call_id_to_name[sanitized_id] = fn_name

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
            name = (
                msg.get("name")
                or (tool_call_id_to_name.get(tool_call_id) if tool_call_id else None)
                or (tool_call_id_to_name.get(sanitize_tool_call_id(tool_call_id, "tool")) if tool_call_id else None)
                or "tool"
            )
            tool_resp_str = content if isinstance(content, str) else json.dumps(content or {})
            resp_part: Dict[str, Any] = {
                "functionResponse": {
                    "name": name,
                    "response": {"output": tool_resp_str},
                }
            }
            if needs_tool_call_id(raw_model, runtime_model):
                resp_part["functionResponse"]["id"] = sanitize_tool_call_id(tool_call_id, name)

            if contents and contents[-1].get("role") == "user" and any("functionResponse" in p for p in contents[-1].get("parts", [])):
                contents[-1]["parts"].append(resp_part)
            else:
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

    finish_chunk_sent = False
    usage_dict: Optional[Dict[str, int]] = None

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
        usage_meta = resp_data.get("usageMetadata") or chunk.get("usageMetadata")
        if usage_meta and isinstance(usage_meta, dict):
            p_tokens = usage_meta.get("promptTokenCount", 0)
            c_tokens = usage_meta.get("candidatesTokenCount", 0)
            t_tokens = usage_meta.get("totalTokenCount", p_tokens + c_tokens)
            usage_dict = {
                "prompt_tokens": p_tokens,
                "completion_tokens": c_tokens,
                "total_tokens": t_tokens,
            }

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

            final_chunk: Dict[str, Any] = {
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
            if usage_dict:
                final_chunk["usage"] = usage_dict
            finish_chunk_sent = True
            yield f"data: {json.dumps(final_chunk)}\n\n"

    if not finish_chunk_sent:
        fallback_finish: Dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        if usage_dict:
            fallback_finish["usage"] = usage_dict
        yield f"data: {json.dumps(fallback_finish)}\n\n"

    yield "data: [DONE]\n\n"


async def aggregate_google_sse_to_openai_response(
    response_stream: AsyncGenerator[str, None],
    model_id: str,
) -> Dict[str, Any]:
    """Aggregates Google SSE events into a single non-streaming OpenAI /v1/chat/completions response."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())

    full_content = ""
    full_reasoning = ""
    tool_calls: List[Dict[str, Any]] = []
    finish_reason = "stop"
    usage_dict: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    async for chunk_str in transform_google_sse_to_openai(response_stream, model_id):
        if not chunk_str.startswith("data:"):
            continue
        payload_str = chunk_str[5:].strip()
        if not payload_str or payload_str == "[DONE]":
            continue

        try:
            chunk = json.loads(payload_str)
        except Exception:
            continue

        if "usage" in chunk and isinstance(chunk["usage"], dict):
            usage_dict = chunk["usage"]

        choices = chunk.get("choices", [])
        if not choices:
            continue

        choice = choices[0]
        delta = choice.get("delta", {})

        if "content" in delta and delta["content"]:
            full_content += delta["content"]

        if "reasoning_content" in delta and delta["reasoning_content"]:
            full_reasoning += delta["reasoning_content"]

        if "tool_calls" in delta and delta["tool_calls"]:
            for tc in delta["tool_calls"]:
                tool_calls.append(tc)

        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

    message: Dict[str, Any] = {
        "role": "assistant",
        "content": full_content if full_content or not tool_calls else None,
    }
    if full_reasoning:
        message["reasoning_content"] = full_reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage_dict,
    }
