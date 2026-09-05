# hermes-antigravity

> Google Antigravity (Cloud Code Assist) provider plugin for [Hermes Agent](https://github.com/hermesagent/hermes) — access Gemini 3.8 Flash, Claude Opus 4.6, Claude Sonnet 4.6, Gemini 3.1 Pro, and more via Google OAuth.

> [!CAUTION]
> This is an **unofficial community plugin** not endorsed by Google. It connects via Google's Cloud Code Assist APIs. Use at your own risk in accordance with Google Cloud terms and policies.

---

## ⚡ Quick Install

Run the installer inside your Hermes setup:

```bash
git clone https://github.com/Josephxcx/hermes-antigravity.git
cd hermes-antigravity
bash scripts/install.sh
```

### 1. Authenticate with Google

Run the auth command in Hermes:

```bash
hermes /antigravity.auth
```

Your default browser will open to complete the Google OAuth PKCE flow on `http://localhost:51121/oauth-callback`. Tokens are saved securely to `~/.hermes/auth.json`.

### 2. Test It

```bash
hermes -z "Hello from Antigravity!" --provider antigravity -m gemini-3.8-flash
```

---

## 🤖 Supported Models

| Model | ID | Reasoning / Thinking |
|---|---|---|
| **Gemini 3.8 Flash** | `gemini-3.8-flash` | Low / Medium / High thinking tiers |
| **Gemini 3.7 Flash** | `gemini-3.7-flash` | Low / Medium / High thinking tiers |
| **Gemini 3.6 Flash** | `gemini-3.6-flash` | Low / Medium / High thinking tiers |
| **Gemini 3.5 Flash** | `gemini-3.5-flash` | Standard / Extra-low / Low / High |
| **Gemini 3.1 Pro** | `gemini-3.1-pro` | Low / High reasoning |
| **Claude Opus 4.6** | `claude-opus-4-6` | Extended thinking |
| **Claude Sonnet 4.6** | `claude-sonnet-4-6` | Thinking |
| **GPT-OSS 120B** | `gpt-oss-120b` | Medium reasoning |

---

## 🛠️ Usage

### CLI Commands

```bash
# One-shot query with Gemini 3.8 Flash
hermes -z "Explain WebSockets in 2 sentences" --provider antigravity -m gemini-3.8-flash

# Use Claude Opus 4.6 with thinking
hermes -z "Review this algorithm for race conditions" --provider antigravity -m claude-opus-4-6

# Interactive agent session
hermes --provider antigravity -m gemini-3.1-pro
```

### Set as Default Provider

In `~/.hermes/config.yaml`:

```yaml
model:
  provider: antigravity
  default: gemini-3.8-flash
```

---

## 🔌 Plugin Commands

Inside Hermes chat / CLI:

| Command | Description |
|---|---|
| `/antigravity.auth` | Interactive Google OAuth sign-in |
| `/antigravity.doctor` | Diagnose connection, token validity, and proxy health |
| `/antigravity.usage` | View quota usage and tier details |

---

## 🏗️ Architecture

```
Hermes Agent (OpenAI chat_completions format)
  │
  ▼
Hermes ProviderProfile (antigravity)
  │
  ▼ [Local Loopback Bridge: http://127.0.0.1:51122/v1]
Embedded Starlette/Uvicorn Async Proxy
  ├── Request Transformer (OpenAI tools/messages ➔ Google Cloud Code Assist format)
  ├── Google Cloud Code Assist Client (Auto-refreshed OAuth Bearer, Project ID Discovery)
  └── SSE Stream Decoder (Extracts thought signatures, thinking, & tool_calls)
```

- **Clean Isolation:** Uses an in-process loopback proxy on `127.0.0.1:51122`. No monkey-patching of Hermes internal dictionaries.
- **Robust Tool Calling:** Full bidirectional conversion between OpenAI function declarations and Google functionCall / functionResponse envelopes.
- **Resilient Re-auth:** Automatic in-flight token refresh when tokens expire.

---

## 🔄 After Hermes Updates

If Hermes updates and recreates its Python virtual environment, simply re-run:

```bash
cd hermes-antigravity
bash scripts/install.sh
```

---

## 🧪 Development & Testing

Run the full pytest suite with `uv`:

```bash
uv run --extra dev pytest
```

---

## 📜 Credits & License

Port of [pi-antigravity](https://github.com/Rahularya01/pi-antigravity) by **Rahul Arya**, originally designed for Pi Coding Agent.

Distributed under the **MIT License**. See `LICENSE` for details.
