# Rust API

A simple REST API built with Rust using [Axum](https://github.com/tokio-rs/axum).

## Prerequisites

- Rust & Cargo (install via [rustup](https://rustup.rs/))

## Getting Started

```bash
# Build
cargo build

# Run
cargo run

# Run in release mode
cargo run --release
```

## Endpoints

| Method | Path     | Description          |
|--------|----------|----------------------|
| GET    | /health  | Health check         |
| POST   | /echo    | Echo JSON body back  |

### Examples

```bash
# Health check
curl http://localhost:3000/health

# Echo
curl -X POST http://localhost:3000/echo \
  -H "Content-Type: application/json" \
  -d '{"message": "hello"}'
```

## Tech Stack

- **axum** — Web framework
- **tokio** — Async runtime
- **serde** — Serialization
- **tracing** — Logging
