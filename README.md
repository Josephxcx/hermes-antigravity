# Hermes Antigravity Provider Plugin (`hermes-antigravity-pi-port`)

A native Hermes Agent model-provider plugin that enables access to Google Antigravity / Cloud Code Assist models (Gemini 3.7 Flash, Claude Sonnet 4.6, Claude Opus 4.6, Gemini 3.1 Pro, GPT-OSS 120B).

> **⚠️ Disclaimer & Terms of Service Warning:**
> This plugin uses Google's internal Cloud Code Assist APIs and OAuth flows. It is an **unofficial community project** and is neither endorsed nor supported by Google. Use at your own risk in accordance with Google Cloud Terms of Service.

---

## Origin & Credits

This project is a Python port of [`pi-antigravity`](https://github.com/Rahularya01/pi-antigravity) by **Rahul Arya**, originally designed for the Pi Coding Agent harness.
- Original author: Rahul Arya
- License: MIT License (see `LICENSE`)

---

## Features

- **Google OAuth 2.0 PKCE:** Loopback callback authorization on `http://localhost:51121/oauth-callback`.
- **Credential Storage:** Stores credentials securely in Hermes's credential store (`~/.hermes/auth.json`).
- **Embedded In-Process Proxy:** Transparently bridges Hermes Agent's OpenAI `chat_completions` transport to Google Cloud Code Assist JSON envelopes and SSE streaming.
- **Dynamic Model & Thinking Routing:**
  - `gemini-3.7-flash` (with `thinkingConfig` mapping)
  - `claude-sonnet-4-6` (Thinking)
  - `claude-opus-4-6` (Thinking)
  - `gemini-3.1-pro`
  - `gpt-oss-120b`
- **Tool Calling:** Translates OpenAI function call declarations and tool invocation messages to Google Cloud Code Assist format and back.
- **Quota & Diagnostics:** Commands for `/antigravity.quota` and `/antigravity.doctor`.

---

## Architecture

```
Hermes Agent (chat_completions)
  │
  ▼
Hermes ProviderProfile (hermes_antigravity)
  │
  ▼ [Local Ephemeral HTTP Bridge (127.0.0.1)]
Embedded Asyncio Proxy
  ├── Request Transformer (OpenAI tools/messages -> Cloud Code Assist format)
  ├── Google Cloud Code Assist Client (Headers, Project ID, Auth Bearer)
  └── SSE Stream Decoder (Extracts thinking/reasoning_content & tool_calls)
```

---

## Installation & Setup

### 1. Automated Install (Recommended)

Run the included installer script:

```bash
git clone https://github.com/Josephxcx/hermes-antigravity-pi-port.git
cd hermes-antigravity-pi-port
./scripts/install.sh
```

This script:
- Locates your Hermes Python virtual environment (`~/.hermes/hermes-agent/venv`).
- Installs the plugin in editable mode with the `hermes_agent.plugins` entry point.
- Creates directory symlinks under `~/.hermes/plugins/model-providers/antigravity` and `~/.hermes/plugins/antigravity`.
- Runs a diagnostic check.

### 2. Manual Installation

If you prefer manual setup:

```bash
# In Hermes's virtual environment:
~/.hermes/hermes-agent/venv/bin/pip install -e .

# Create persistent model-providers directory symlink:
mkdir -p ~/.hermes/plugins/model-providers
ln -sfn "$(pwd)" ~/.hermes/plugins/model-providers/antigravity
```

---

## Hermes Update Resilience

This plugin is designed to persist across Hermes updates (`hermes update` / git pulls):
1. **Directory Symlinks**: The link in `~/.hermes/plugins/model-providers/` resides outside the `~/.hermes/hermes-agent` git repository and will not be overwritten by upstream updates.
2. **Schema Sanitization**: Tool schemas are strictly sanitized against the Google Cloud Code Assist OpenAPI subset (`ALLOWED_GEMINI_SCHEMA_KEYS`). Any new JSON Schema validation keywords (e.g. `minItems`, `maxItems`, `pattern`) added by future Hermes tool updates are automatically and safely stripped.
3. **If Hermes re-creates its venv**: Simply re-run `./scripts/install.sh` to reinstall the pip entry point into the new venv.

---

## Usage

### Run with Hermes CLI

```bash
# One-shot command
hermes -z "Explain quantum entanglement in 2 sentences" --provider antigravity -m gemini-3.7-flash

# Interactive chat
hermes --provider antigravity -m claude-sonnet-4-6
```

### Supported Models

- `gemini-3.7-flash` (supports reasoning/thinking effort)
- `claude-sonnet-4-6`
- `claude-opus-4-6`
- `gemini-3.1-pro`
- `gemini-3.6-flash`
- `gemini-3.5-flash`
- `gpt-oss-120b`

---

## Development & Testing

Run the test suite using `uv`:

```bash
uv run --extra dev pytest
```

Run the live authenticated smoke test:

```bash
uv run python scripts/smoke_test.py
```

