# Rust API

A simple REST API built with Rust using [Axum](https://github.com/tokio-rs/axum).

## Prerequisites

- Rust & Cargo (install via [rustup](https://rustup.rs/))
- OpenAI API key (`OPENAI_API_KEY`) for `/agent/chat`

## Getting Started

```bash
# Build
cargo build

# Run
export OPENAI_API_KEY=your_api_key_here
# Optional overrides:
# export OPENAI_MODEL=gpt-4o-mini
# export OPENAI_BASE_URL=https://api.openai.com/v1
cargo run

# Run in release mode
cargo run --release
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Health check |
| POST | /echo | Echo JSON body back |
| POST | /agent/chat | Chat completion via OpenAI |

### Examples

```bash
# Health check
curl http://localhost:3000/health

# Echo
curl -X POST http://localhost:3000/echo \
  -H "Content-Type: application/json" \
  -d '{"message": "hello"}'

# Agent chat
curl -X POST http://localhost:3000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      { "role": "system", "content": "You are concise." },
      { "role": "user", "content": "Say hello in one sentence." }
    ]
  }'
```

## Tech Stack

- **axum** - Web framework
- **tokio** - Async runtime
- **serde** - Serialization
- **tracing** - Logging
- **async-openai** - OpenAI API client
