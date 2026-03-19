# NYQST-Scriba

Intent-aware transcription tool with local and cloud backends.

## Quick Start

```bash
# Install with all backends
uv pip install -e ".[all]"

# Configure API keys
scriba configure

# Transcribe
scriba transcribe recording.m4a -q fast -t raw -f text
scriba transcribe meeting.mp4 -d -s 4 -q high -f md

# Use as MCP server (for Claude Code)
scriba-mcp
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

## Claude Code Plugin

Add to your `.mcp.json`:
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

## License

MIT
