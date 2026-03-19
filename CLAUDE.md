# Scriba

Intent-aware transcription tool. See `docs/superpowers/specs/2026-03-19-scriba-design.md` for full design.

## Dev Setup

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest
```

Requires Python 3.11+ and ffmpeg.

## Architecture
- `src/scriba/contracts.py` — all data models (Segment, TranscriptionConfig, Estimate, RoutingDecision, TranscriptionResult)
- `src/scriba/router/` — constraint-based backend selection (normalize → filter → estimate → rank → select)
- `src/scriba/backends/` — MLX Whisper, WhisperX, OpenAI adapters (Protocol-based)
- `src/scriba/media/` — ffprobe/ffmpeg audio extraction and subtitle burning
- `src/scriba/secrets/` — keychain → env var → .env secret resolution chain
- `src/scriba/output/` — format rendering (json/text/srt/vtt/md)
- `src/scriba/formatting.py` — shared timestamp formatting and SRT/VTT generation
- `src/scriba/mcp/server.py` — FastMCP server (transcribe, estimate, backends tools)
- `src/scriba/cli.py` — Click CLI (transcribe, estimate, backends, configure commands)
- `plugin/` — Claude Code plugin manifest, MCP config, and transcription skill
