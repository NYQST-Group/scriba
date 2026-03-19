# NYQST-Scriba

Intent-aware transcription tool with local and cloud backends behind a constraint-based router.

## Quick Start

```bash
# Clone and set up
git clone https://github.com/NYQST-Group/scriba.git
cd scriba
uv venv .venv && source .venv/bin/activate

# Install with all backends (MLX Whisper, WhisperX, OpenAI)
uv pip install -e ".[all]"

# Or install only what you need
uv pip install -e ".[mlx]"       # Apple Silicon local transcription
uv pip install -e ".[openai]"    # Cloud transcription via OpenAI

# Configure API keys (optional — local backends work without keys)
scriba configure

# Transcribe
scriba transcribe recording.m4a -q fast -t raw -f text
scriba transcribe meeting.mp4 -d -s 4 -q high -f md

# Check available backends
scriba backends
```

## Backends

| Backend | Type | Diarization | Cost |
|---------|------|-------------|------|
| MLX Whisper | Local (Apple Silicon) | No | Free |
| WhisperX + pyannote | Local | Yes | Free |
| OpenAI STT | Cloud | Yes (select models) | Per-minute |

## Intent-Based Routing

Instead of choosing backends manually, describe what you need:

- **"quick transcript"** → fast local transcription
- **"meeting notes, 4 speakers"** → high-quality diarized output
- **"transcribe in 30s, budget 20 cents"** → router picks optimal backend

The router selects the best backend, model, and settings based on your constraints (quality, budget, timeout, diarization). Trade-offs are explained when constraints conflict.

## CLI Commands

```bash
scriba transcribe <file>   # Transcribe audio/video
scriba estimate <file>     # Preview cost/time without transcribing
scriba backends            # List available backends
scriba configure           # Interactive API key setup
```

Key options for `transcribe`:
- `-q fast|balanced|high` — quality tier
- `-t raw|timestamped|diarized|enriched` — output depth
- `-f json|text|srt|vtt|md` — output format
- `-d` / `--diarize` — identify speakers
- `--subtitle` — burn subtitles onto video
- `--dry-run` — show routing decision without transcribing
- `--enrich` — summarize via OpenAI

## MCP Server (Claude Code Integration)

Scriba runs as an MCP server for use with Claude Code:

```bash
# Start directly
scriba-mcp

# Or add to your Claude Code .mcp.json
```

```json
{
  "mcpServers": {
    "scriba": {
      "command": "uv",
      "args": ["run", "scriba-mcp"]
    }
  }
}
```

The MCP server exposes `transcribe`, `estimate`, and `backends` tools.

## Development

```bash
# Set up dev environment
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Run tests
pytest

# Lint and type check
ruff check src/ tests/
pyright src/scriba/
```

Requires Python 3.11+ and ffmpeg (`brew install ffmpeg` on macOS).

See [CLAUDE.md](CLAUDE.md) for architecture overview and `docs/superpowers/specs/` for the full design spec.

## License

MIT
