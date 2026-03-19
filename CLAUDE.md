# Scriba

Intent-aware transcription tool. See `docs/superpowers/specs/2026-03-19-scriba-design.md` for full design.

## Dev Setup
uv pip install -e ".[dev]"
pytest

## Architecture
- `src/scriba/contracts.py` — all data models
- `src/scriba/router/` — constraint-based backend selection
- `src/scriba/backends/` — MLX Whisper, WhisperX, OpenAI adapters
- `src/scriba/mcp/server.py` — FastMCP server
- `src/scriba/cli.py` — Click CLI
