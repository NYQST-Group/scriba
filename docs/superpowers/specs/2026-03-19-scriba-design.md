# NYQST-Scriba: Intent-Aware Transcription Tool

**Date:** 2026-03-19
**Repo:** `NYQST-Group/scriba`
**Status:** Design approved

## Overview

Scriba is an intent-aware transcription tool that unifies three proven backends (MLX Whisper, WhisperX+pyannote, OpenAI STT) behind a constraint-based router. Users describe what they need — fast, high-quality, diarized, within budget — and the router selects the best backend, model, and settings.

Three interfaces: CLI, MCP server, and Claude Code plugin. Local-first by default, cloud when constraints demand it.

## Architecture

```
User Intent / Explicit Params
         │
    ┌────▼────┐
    │  Router  │  ← cost model, timing estimates, file analysis
    └────┬────┘
         │ selects
    ┌────┼──────────────┐
    ▼    ▼              ▼
  MLX   WhisperX    OpenAI STT
Whisper  +pyannote   (cloud)
    │    │              │
    └────┼──────────────┘
         ▼
   Output Pipeline
   (format, subtitle, SRT/VTT/JSON/MD/plain)
```

### Core Components

1. **Media Ingester** — accepts video/audio in any format, uses ffmpeg to probe metadata and extract/normalize audio to 16kHz mono WAV.
2. **Router** — takes constraints (quality, budget, timeout, diarize, speakers) and selects backend + model. Returns trade-off explanations when constraints conflict.
3. **Backend Adapters** — uniform async interface over MLX Whisper, WhisperX+pyannote, OpenAI STT.
4. **Output Pipeline** — renders results into requested tier and format.
5. **Subtitle Burner** — generates SRT/VTT and optionally burns subtitles onto source video via ffmpeg.
6. **Secrets Provider** — macOS Keychain with abstraction layer for future enterprise backends (Vault, AWS SM, GCP SM).
7. **Cost Estimator** — estimates cloud cost from audio duration/model, tracks spend.

## Media Ingester

**Input formats:** mov, mp4, mkv, webm, m4a, wav, mp3, flac, ogg, aac — anything ffmpeg can decode.

**Pipeline:**

1. **Probe** — `ffprobe` extracts duration, codec, channels, sample rate, file size. Metadata feeds the router's time/cost estimates.
2. **Extract & normalize** — strips audio to 16kHz mono WAV. For files already in spec, skip re-encoding.
3. **Preserve video reference** — when input is video and subtitle burning is requested, the original video path is retained.

**Size handling:**

- For OpenAI cloud backend: auto-compress to 24kbps mono MP3 if extracted audio exceeds 25MB.
- For local backends: no size limit beyond available memory.

## Router & Constraint Model

### Input Modes

**Explicit params (traditional):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `quality` | `fast \| balanced \| high` | `balanced` | Transcription quality tier |
| `budget_cents` | `int \| null` | `null` (no limit) | Maximum spend in cents |
| `timeout_seconds` | `int \| null` | `null` (no limit) | Maximum wall-clock time |
| `diarize` | `bool` | `false` | Identify speakers |
| `speakers` | `int \| null` | `null` | Hint for expected speaker count |
| `output_tier` | `raw \| timestamped \| diarized \| enriched` | `timestamped` | Content depth |
| `language` | `str \| null` | `null` (auto-detect) | Language hint (ISO 639-1, e.g. `en`, `es`) |
| `output_format` | `json \| text \| srt \| vtt \| md` | `json` | Serialization format |

**Intent string (natural language):**

When used as a Claude Code plugin, Claude interprets the user's natural language and translates to explicit params before calling the MCP tool. The `intent` field on the MCP `transcribe` tool is for logging/provenance only — it records what the user originally said but does not drive routing. The router always operates on explicit params.

For CLI use, a simple keyword parser maps common phrases to params:

- "quick transcript of my voice note" → `{quality: fast, diarize: false, output_tier: raw}`
- "high quality, 4 speakers, no rush" → `{quality: high, diarize: true, speakers: 4}`
- "transcribe in 30 seconds, willing to spend 20 cents" → `{timeout_seconds: 30, budget_cents: 20}`

**Parameter interaction rules:**

- `output_tier: diarized` or `output_tier: enriched` implies `diarize: true`. If the user explicitly sets `diarize: false` with `output_tier: diarized`, the router treats this as a conflict and returns an explanation.
- `output_tier: raw` with `diarize: true` is valid — diarization is performed but output is plain text only (speakers not rendered). Useful when the caller will process the structured result themselves.

### Decision Process

1. **Normalize** — resolve parameter interactions. `output_tier: diarized/enriched` implies `diarize: true`. Detect and flag contradictions.

2. **Filter** — remove backends that cannot satisfy hard constraints:
   - `diarize: true` eliminates MLX Whisper (no diarization capability)
   - `diarize: true` with OpenAI eliminates `whisper-1` (does not support `diarized_json`; only `gpt-4o-mini-transcribe` and `gpt-4o-transcribe` support it)
   - `budget_cents: 0` eliminates all cloud options
   - `quality: fast` with `diarize: true` → no valid candidate (diarization is inherently slower). Return trade-off explanation.
   - Backend not installed → eliminated

3. **Estimate** — for each remaining (backend, model) pair:
   - Estimated wall-clock time (local: duration × model multiplier; cloud: upload + API processing)
   - Cost in cents (local: 0; cloud: duration × rate)

4. **Rank** — score candidates against soft constraints. Prefer local when constraints are met.

5. **Select or explain** — if a candidate satisfies all constraints, select it. If constraints conflict, return a `RoutingDecision` containing the best feasible option, what was traded off, and alternative options for Claude to present.

**RoutingDecision consumption:**

- **MCP server:** the `RoutingDecision` is included in the response metadata. When `trade_offs` is non-empty, the response includes a `routing_notes` field with human-readable explanations. The server always proceeds with the best feasible option — it does not block for confirmation. Claude can inspect the trade-offs and present them to the user.
- **CLI:** when `trade_offs` is non-empty, the trade-off explanations are printed to stderr before transcription proceeds. The `--dry-run` flag (alias for the `estimate` command) shows the full routing decision without transcribing.
- **Both:** the `RoutingDecision` is attached to `TranscriptionResult.routing` so downstream consumers always know what was selected and why.

### Cost Model

| Backend / Model | Cost per Minute | Time Multiplier (approx) |
|-----------------|----------------|--------------------------|
| MLX tiny | $0.00 | 0.05× realtime |
| MLX base | $0.00 | 0.08× realtime |
| MLX small | $0.00 | 0.12× realtime |
| MLX medium | $0.00 | 0.20× realtime |
| MLX large-v3 | $0.00 | 0.30× realtime |
| WhisperX large-v3 + diarize | $0.00 | 0.50× realtime |
| OpenAI whisper-1 | $0.006 | 0.15× + upload |
| OpenAI gpt-4o-mini-transcribe | $0.003 | 0.15× + upload |
| OpenAI gpt-4o-transcribe | $0.006 | 0.15× + upload |

Time multipliers are initial estimates for Apple Silicon M-series Pro/Max chips. Base M-series may be 2-3× slower for larger models. WhisperX runs via PyTorch (not MLX-optimized), so the 0.50× estimate includes significant pyannote overhead.

**Self-calibration:** after each transcription, the tool records `(backend, model, audio_duration, wall_clock_time)` to `~/.config/scriba/calibration.json`. A rolling average of the last 10 runs per (backend, model) pair updates the time multipliers. Stale entries (>30 days) are discarded. This is a simple mechanism — no concurrent write protection needed because local backends run single-threaded (see Concurrency Model below).

### Output Tier × Quality Matrix

| | fast | balanced | high |
|---|---|---|---|
| **raw** | MLX tiny, text only | MLX medium, text | MLX/cloud large-v3, text |
| **timestamped** | MLX small, segments | MLX medium, segments | large-v3, segments |
| **diarized** | unavailable | WhisperX medium | WhisperX/cloud large-v3 |
| **enriched** | unavailable | WhisperX medium + summary | cloud large-v3 + summary |

"Fast diarized" does not exist — diarization is inherently slower. The router communicates this rather than silently degrading.

## Backend Adapters

### Common Interface

```python
class BackendAdapter(Protocol):
    async def transcribe(self, audio_path: Path, config: TranscriptionConfig) -> TranscriptionResult
    def estimate(self, duration_seconds: float, config: TranscriptionConfig) -> Estimate
    def is_available(self) -> bool
```

### Data Models

```python
@dataclass
class TranscriptionConfig:
    """Resolved configuration passed to a backend adapter."""
    model: str                       # e.g. "large-v3", "gpt-4o-transcribe"
    language: str | None = None      # ISO 639-1 or None for auto-detect
    diarize: bool = False
    speakers: int | None = None      # hint for diarization
    output_tier: str = "timestamped" # raw | timestamped | diarized | enriched

@dataclass
class Segment:
    start: float          # seconds
    end: float            # seconds
    text: str
    speaker: str | None   # e.g. "SPEAKER_00"

@dataclass
class TranscriptionResult:
    text: str
    segments: list[Segment]
    duration_seconds: float
    model_used: str
    backend: str          # "mlx_whisper" | "whisperx" | "openai_stt"
    cost_cents: float     # 0.0 for local
    diarized: bool
    enrichment_available: bool = False  # True when tier=enriched but enrichment deferred to caller
    routing: RoutingDecision | None = None  # included when trade-offs were made

@dataclass
class Estimate:
    backend: str
    model: str
    time_seconds: float
    cost_cents: float
    available: bool
    recommended: bool              # router's top pick for given constraints
    reason_unavailable: str | None

@dataclass
class RoutingDecision:
    """Returned by the router engine."""
    selected: Estimate             # chosen (backend, model) pair
    alternatives: list[Estimate]   # other viable options
    trade_offs: list[str] | None   # human-readable explanations if constraints conflicted
```

**Overlapping speech:** diarization can produce overlapping time ranges where two speakers talk simultaneously. The `Segment` model represents these as separate segments with overlapping `start`/`end` ranges and different `speaker` values. Consumers should not assume segments are non-overlapping. The SRT/VTT formatter merges overlaps into `[SPEAKER_00 & SPEAKER_01]: text` lines.

### MLX Whisper Adapter

- Models: `tiny`, `base`, `small`, `medium`, `large-v3`
- Apple Silicon native via MLX framework
- No diarization capability
- Models download on first use to `~/.cache/scriba/models`

### WhisperX + pyannote Adapter

- Same Whisper model sizes + pyannote speaker diarization pipeline
- Requires HuggingFace token for pyannote gated models
- Supports speaker count hints
- Heavier dependency footprint (PyTorch)

### OpenAI STT Adapter

- Models: `whisper-1`, `gpt-4o-mini-transcribe`, `gpt-4o-transcribe`
- Diarization via `diarized_json` response format. Only `gpt-4o-mini-transcribe` and `gpt-4o-transcribe` support this; `whisper-1` does not. The router enforces this constraint.
- Auto-compression for files exceeding 25MB upload limit
- Retry with exponential backoff (3 attempts, jittered wait)
- Language hint passed through as `language` parameter

### Backend Availability

Probed at startup and cached. The router never selects an unavailable backend. Missing optional dependencies (mlx-whisper, whisperx, openai) cause graceful degradation — those backends silently drop from the candidate pool.

## Output Pipeline

### Formats

- **json** — canonical structured output with full metadata
- **text** — plain text rendering (raw: just words; diarized: `SPEAKER_00: text`)
- **srt** — SubRip subtitle format with timestamps
- **vtt** — WebVTT subtitle format with timestamps
- **md** — Markdown with speaker headers and timestamps

SRT/VTT require timestamped tier or above.

### Subtitle Burner

When input was video and subtitles are requested:

1. Generate SRT from segments
2. Burn into video via ffmpeg:
   - **Soft subtitles (default):** embedded as a subtitle stream. Uses `-c:s mov_text` for MP4/MOV, `-c:s srt` for MKV, `-c:s webvtt` for WebM. Container format detected from input extension.
   - **Hard burn-in:** filter overlay via `-vf subtitles=`. Works with any container.
3. Output path: `{input_stem}.subtitled.{ext}`

### Enrichment (Tier 4)

- **As Claude plugin (MCP):** when the MCP server receives `output_tier: enriched`, it performs transcription at the `diarized` level and returns the result with `enrichment_available: true` in metadata. Claude handles the actual summarization from the returned transcript — no extra API call from the server. This avoids double-billing and lets Claude use its own context.
- **As standalone CLI:** `--enrich` flag calls OpenAI for summary, action items, decisions (structured output matching the `openai_insights.py` pattern from speaker-transcript-system). Requires OpenAI API key; if unavailable, falls back to `diarized` tier with a warning.

## Error Handling

All errors are wrapped in a `ScribaError` hierarchy with actionable recovery messages:

```python
class ScribaError(Exception): ...
class DependencyMissing(ScribaError): ...    # ffmpeg not found, backend not installed
class BackendError(ScribaError): ...         # transcription failed mid-run
class AudioError(ScribaError): ...           # corrupt file, zero-length, unsupported format
class SecretsError(ScribaError): ...         # missing API key, auth failure
class RoutingError(ScribaError): ...         # no backends available, unresolvable constraints
class BudgetExceeded(ScribaError): ...       # estimated cost exceeds budget_cents
```

**Failure strategies:**

| Scenario | Behavior |
|----------|----------|
| ffmpeg/ffprobe not installed | `DependencyMissing` with install instructions (`brew install ffmpeg`) |
| All backends unavailable | `RoutingError` listing what's missing and how to install each |
| Backend fails mid-run (OOM, network) | `BackendError` with backend name, original error, and suggestion (e.g. "try a smaller model or cloud backend") |
| Corrupt/zero-length audio | `AudioError` after ffprobe detects invalid stream |
| OpenAI rate limit / auth error | Retry (exponential backoff) then `BackendError` with specific API error |
| HF token valid but pyannote license not accepted | `SecretsError` with link to HuggingFace model page |
| Model download fails / partial | Retry once. On second failure, `BackendError` with suggestion to clear cache and retry. Integrity verified via file size check post-download. |

The MCP server maps these to structured error responses. The CLI prints human-readable messages with recovery steps.

## Concurrency Model

- **Local backends (MLX, WhisperX):** single request at a time. Both consume GPU memory aggressively — concurrent inference risks OOM. The MCP server queues incoming requests and processes them sequentially for local backends.
- **Cloud backend (OpenAI):** concurrent requests allowed (up to 3 parallel, configurable). OpenAI handles server-side concurrency.
- **Mixed:** a cloud request can run concurrently with a queued local request.

This is enforced via an `asyncio.Semaphore` per backend category. Defaults are hardcoded (`local=1`, `cloud=3`) but overridable in config (see Configuration section).

## Long Audio Handling

- **Local backends:** no explicit chunking — MLX Whisper and WhisperX handle long audio internally via their own sliding window / VAD segmentation. Memory usage scales with duration; for recordings >2 hours on base M-series (8GB unified memory), the router should prefer smaller models or cloud.
- **OpenAI cloud:** the API handles chunking internally when `chunking_strategy: auto` is set (for supported models). For `whisper-1`, the 25MB upload limit is the effective ceiling — the auto-compression step produces a file of adequate length at 24kbps (≈55 minutes per 10MB).
- **Duration-aware routing:** the router factors audio duration into its estimates. For very long recordings (>60 min), it biases toward cloud when budget allows, because local processing time becomes significant.

## Secrets Provider

### Interface

```python
class SecretsProvider(Protocol):
    async def get(self, key: str) -> str | None
    async def set(self, key: str, value: str) -> None
    async def delete(self, key: str) -> None
```

### Resolution Order

Keychain → environment variable → `.env` file → prompt user interactively

### Implementations

1. **KeychainProvider** — via `keyring` package. Supports macOS Keychain, Windows Credential Locker, Linux SecretService. Service name: `scriba`.
2. **EnvProvider** — reads `SCRIBA_OPENAI_API_KEY`, `SCRIBA_HF_TOKEN`. For CI, Docker, non-macOS.
3. **Future enterprise:** `VaultProvider`, `AWSSecretsManagerProvider`, `GCPSecretManagerProvider` — new file + config toggle, zero changes to consumers.

### Managed Secrets

| Key | Env Var Fallback | Required For |
|-----|-----------------|--------------|
| `scriba/openai-api-key` | `SCRIBA_OPENAI_API_KEY` | OpenAI STT backend |
| `scriba/hf-token` | `SCRIBA_HF_TOKEN` | pyannote diarization models |

### Setup Command

`scriba configure` walks through interactive setup: detects platform, stores keys in keychain, validates them (test API call, HF model access check), reports which backends are now available.

## MCP Server

Built with FastMCP.

### Tools

**`transcribe`** — main transcription tool

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file_path` | `str` | yes | — | Path to audio/video file |
| `quality` | `str` | no | `balanced` | `fast`, `balanced`, `high` |
| `budget_cents` | `int` | no | `null` | Max spend |
| `timeout_seconds` | `int` | no | `null` | Max wall-clock time |
| `diarize` | `bool` | no | `false` | Identify speakers |
| `speakers` | `int` | no | `null` | Expected speaker count hint |
| `language` | `str` | no | `null` | ISO 639-1 language hint (auto-detect if null) |
| `output_tier` | `str` | no | `timestamped` | `raw`, `timestamped`, `diarized`, `enriched` |
| `output_format` | `str` | no | `json` | `json`, `text`, `srt`, `vtt`, `md` |
| `subtitle_video` | `bool` | no | `false` | Burn subtitles onto source video |
| `intent` | `str` | no | `null` | Natural language alternative to explicit params |

Returns: transcript in requested format + metadata (backend used, cost, duration, file paths).

**`estimate`** — preview cost/time without transcribing. Accepts the same constraint params as `transcribe` so the router can produce accurate recommendations.

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `file_path` | `str` | yes | — |
| `quality` | `str` | no | `balanced` |
| `budget_cents` | `int` | no | `null` |
| `timeout_seconds` | `int` | no | `null` |
| `diarize` | `bool` | no | `false` |
| `speakers` | `int` | no | `null` |
| `output_tier` | `str` | no | `timestamped` |

Returns: `RoutingDecision` with `selected` (recommended), `alternatives`, and `trade_offs` if constraints conflict.

**`backends`** — list available backends and status

Returns: `[{name, available, models[], reason_unavailable}]`.

### Transport

- Default: `stdio` (for Claude Code plugin use)
- Also supports: `sse`, `streamable-http` (for standalone/web use)
- Configurable via `SCRIBA_MCP_TRANSPORT`, `SCRIBA_MCP_HOST`, `SCRIBA_MCP_PORT`

## Integration Model

The CLI and MCP server are separate entry points that share the same core library:

```
scriba (CLI)  ──┐
                ├──→ Router → Backend Adapters → Output Pipeline
scriba-mcp    ──┘
```

- **CLI (`scriba`):** parses args via Click, calls the router directly, writes output to stdout/file.
- **MCP server (`scriba-mcp`):** exposes tools via FastMCP, calls the same router, returns structured responses over the transport.
- **Claude Code plugin:** starts `scriba-mcp` as a subprocess via stdio transport. The plugin's `.mcp.json` specifies `uv run scriba-mcp` as the command.

Both entry points can coexist — the CLI is for direct use, the MCP server is for Claude integration. They share config, secrets, and calibration data.

## CLI Interface

### Subcommands

```
scriba transcribe <file>   Transcribe audio/video file
scriba estimate <file>     Preview cost/time per backend
scriba backends            List available backends and status
scriba configure           Interactive setup (secrets, validation)
```

### `scriba transcribe`

```
Usage: scriba transcribe [OPTIONS] FILE

Options:
  -q, --quality [fast|balanced|high]   Quality tier (default: balanced)
  -t, --tier [raw|timestamped|diarized|enriched]  Output tier (default: timestamped)
  -f, --format [json|text|srt|vtt|md]  Output format (default: json)
  -d, --diarize                        Enable speaker diarization
  -s, --speakers INT                   Expected speaker count hint
  -l, --language TEXT                   ISO 639-1 language code
  --budget INT                         Max spend in cents
  --timeout INT                        Max wall-clock seconds
  --subtitle                           Burn subtitles onto video (video input only)
  --subtitle-mode [soft|hard]          Subtitle embedding mode (default: soft)
  --enrich                             Summarize via OpenAI (enriched tier)
  -o, --output PATH                    Output file path (default: stdout for text/json/md,
                                       <stem>.srt/.vtt for subtitle formats)
  --dry-run                            Show routing decision without transcribing
  --intent TEXT                         Natural language intent (alternative to flags)
```

**Output behavior:**
- By default, writes to stdout (json, text, md formats) or to `<stem>.<ext>` (srt, vtt).
- `--output` overrides the destination.
- Trade-off explanations from the router print to stderr, never polluting stdout.
- Subtitled video always writes to `<stem>.subtitled.<ext>`.

**Examples:**
```bash
# Quick voice note transcript
scriba transcribe recording.m4a -q fast -t raw -f text

# Diarized meeting, markdown output
scriba transcribe meeting.mp4 -d -s 4 -q high -t diarized -f md -o meeting.md

# Subtitled video
scriba transcribe talk.mov -d --subtitle -t diarized

# Check what it would cost
scriba estimate long-interview.wav -d -q high

# Natural language intent
scriba transcribe memo.m4a --intent "quick transcript, I just need the text"
```

### `scriba estimate`

Prints a table of backend options with estimated time, cost, and recommendation marker. Accepts the same constraint flags as `transcribe`.

### `scriba configure`

Interactive walkthrough:
1. Detect platform (macOS → keychain, Linux → SecretService, other → env fallback)
2. Prompt for OpenAI API key → validate with a test API call → store in keychain
3. Prompt for HuggingFace token → validate pyannote model access → store in keychain
4. Probe installed backends → report what's available
5. Idempotent — safe to re-run to update keys or check status

Exit codes: 0 = success, 1 = error, 2 = partial setup (some backends unavailable).

## Claude Code Plugin

### Structure

```
plugin/
  plugin.json              # Manifest
  skills/
    transcribe.md          # Intent interpretation skill
  .mcp.json                # MCP server config
```

### Skill (`transcribe.md`)

Instructs Claude:

- How to interpret user requests ("quick transcript" → fast/raw, "meeting notes with speakers" → high/diarized)
- When to call `estimate` first (large files, budget constraints)
- How to present trade-offs when constraints conflict
- To default to local/free when no constraints suggest otherwise
- To offer subtitle burning when input is video

## Project Structure

```
NYQST-Group/scriba/
├── pyproject.toml
├── README.md
├── CLAUDE.md
├── .env.example
├── .github/workflows/ci.yml
├── src/scriba/
│   ├── __init__.py
│   ├── cli.py                     # Click-based CLI
│   ├── config.py                  # Settings, defaults, config.toml loader
│   ├── contracts.py               # TranscriptionResult, Segment, Estimate
│   ├── media/
│   │   ├── __init__.py
│   │   ├── ingest.py              # ffprobe + ffmpeg extract/normalize
│   │   └── subtitle.py            # SRT generation + video burn-in
│   ├── router/
│   │   ├── __init__.py
│   │   ├── constraints.py         # Constraint model + validation
│   │   ├── cost_model.py          # Pricing tables + time estimates
│   │   └── engine.py              # Routing logic + trade-off explanation
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py                # Protocol definition
│   │   ├── mlx_whisper.py         # MLX adapter
│   │   ├── whisperx.py            # WhisperX + pyannote adapter
│   │   └── openai_stt.py          # OpenAI API adapter
│   ├── output/
│   │   ├── __init__.py
│   │   └── formatter.py           # Tier + format rendering
│   ├── secrets/
│   │   ├── __init__.py
│   │   ├── provider.py            # Abstract interface
│   │   ├── keychain.py            # macOS Keychain via keyring
│   │   └── env.py                 # Env var / dotenv fallback
│   └── mcp/
│       ├── __init__.py
│       └── server.py              # FastMCP server
├── plugin/
│   ├── plugin.json
│   ├── skills/
│   │   └── transcribe.md
│   └── .mcp.json
└── tests/
    ├── conftest.py                # Fixtures + mock backends
    ├── fixtures/                   # Short test audio/video files
    ├── test_router.py
    ├── test_cost_model.py
    ├── test_ingest.py
    ├── test_formatter.py
    ├── test_subtitle.py
    ├── test_secrets.py
    ├── test_backends/
    │   ├── test_mlx.py
    │   ├── test_whisperx.py
    │   └── test_openai.py
    └── test_mcp/
        └── test_server.py
```

## Dependencies

```toml
[project]
name = "nyqst-scriba"
requires-python = ">=3.11"
dependencies = [
    "anyio>=4.0",
    "click>=8.0",
    "keyring>=25.0",
    "mcp>=1.15",
]

[project.optional-dependencies]
mlx = ["mlx-whisper>=0.4"]
whisperx = ["whisperx>=3.7", "pyannote.audio>=3.4"]
openai = ["openai>=1.50"]
all = ["nyqst-scriba[mlx,whisperx,openai]"]
dev = ["pytest>=8", "pytest-asyncio>=0.23", "pytest-mock>=3", "ruff>=0.4", "pyright>=1.1"]

[project.scripts]
scriba = "scriba.cli:main"
scriba-mcp = "scriba.mcp.server:main"
```

## Configuration

Default location: `~/.config/scriba/config.toml`

```toml
[defaults]
quality = "balanced"
output_tier = "timestamped"
output_format = "json"
diarize = false

[backends]
prefer_local = true

[concurrency]
max_local = 1       # semaphore for local backends (MLX, WhisperX)
max_cloud = 3       # semaphore for cloud backends (OpenAI)

[calibration]
path = "~/.config/scriba/calibration.json"  # timing data for self-calibration
max_samples = 10    # rolling average window per (backend, model) pair
stale_days = 30     # discard entries older than this

[openai]
model = "gpt-4o-mini-transcribe"
max_budget_cents_per_job = 50

[mlx]
default_model = "large-v3"
cache_dir = "~/.cache/scriba/models"

[whisperx]
default_model = "large-v3"
```

## Testing Strategy

TDD approach. Transcription logic is proven from existing projects; tests focus on integration, routing, and packaging layers.

### Unit Tests (mocked backends)

- **Router:** constraint → backend selection. Edge cases: conflicting constraints, all backends unavailable, budget boundary conditions.
- **Cost model:** duration × model → correct cost. Budget → correct model ceiling.
- **Formatter:** TranscriptionResult → correct JSON/text/SRT/VTT/MD output.
- **Subtitle:** segments → valid SRT timing. Edge cases: overlapping segments, empty text.
- **Secrets:** keychain mock, env fallback, resolution order.
- **Media ingest:** ffprobe output parsing, format detection.

### Integration Tests (real inference)

- 2-5 second WAV/video test fixtures committed to repo.
- MLX Whisper: transcribe fixture, verify output shape. Skipped in CI if no Apple Silicon.
- OpenAI: transcribe via API. Skipped unless `SCRIBA_OPENAI_API_KEY` set.
- ffmpeg: extract audio from video fixture, verify output format.
- Subtitle: generate SRT + overlay on test video, verify output file.

### MCP Server Tests

- Tool registration: all tools listed with correct schemas.
- Parameter validation: missing required params, invalid enum values.
- End-to-end: mock backend → call transcribe → verify response structure.

### CI (GitHub Actions)

- **Ubuntu runner:** unit tests + linting (ruff) + type checking (pyright).
- **macOS runner:** MLX integration tests (if Apple Silicon available).
- **OpenAI integration:** manual trigger or tagged releases only (costs money).

## Distribution

### Phase 1: Local MCP Server

- Clone repo, `uv pip install -e ".[all,dev]"`, run `scriba-mcp`.
- Point Claude Code at it via `.mcp.json`.

### Phase 2: Claude Code Plugin

- Package as published Claude plugin.
- One-click install from plugin registry.
- Auto-starts MCP server on plugin load.
