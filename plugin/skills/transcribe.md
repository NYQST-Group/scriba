---
name: transcribe
description: Transcribe audio and video files using local or cloud backends. Use when the user asks to transcribe, caption, subtitle, or get text from audio/video files.
---

# Transcription Skill

You have access to the Scriba transcription MCP server with these tools:

## Tools

- **transcribe**: Transcribe audio/video files
- **estimate**: Preview cost/time before transcribing
- **backends**: Check which backends are available

## Intent Mapping

When the user describes what they want, map to these parameters:

| User says | Parameters |
|-----------|-----------|
| "quick transcript" / "just the text" | quality=fast, tier=raw, format=text |
| "transcribe this" (default) | quality=balanced, tier=timestamped |
| "meeting notes with speakers" | quality=high, diarize=true, tier=diarized |
| "high quality, N speakers" | quality=high, diarize=true, speakers=N |
| "subtitle this video" | diarize=true, tier=diarized, subtitle_video=true |
| mentions budget/cost | set budget_cents accordingly |
| mentions time pressure | set timeout_seconds accordingly |

## Workflow

1. If the file is large (>10 min) or user mentions cost/time, call **estimate** first
2. Present the routing decision — especially trade-offs if any
3. Call **transcribe** with the right parameters
4. If input was video, offer to add subtitles

## Defaults

- Always prefer local/free backends unless the user indicates willingness to pay
- Default to `quality=balanced`, `tier=timestamped`, `format=json`
- If user asks for speakers/diarization, set `diarize=true`
