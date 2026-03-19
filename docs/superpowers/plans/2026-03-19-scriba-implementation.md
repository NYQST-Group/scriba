# NYQST-Scriba Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an intent-aware transcription tool with CLI, MCP server, and Claude Code plugin interfaces, unifying MLX Whisper, WhisperX+pyannote, and OpenAI STT behind a constraint-based router.

**Architecture:** Layered design — contracts and errors at the bottom, config and secrets above them, media ingestion and backend adapters in the middle, router orchestrating backends, output formatting on top, with CLI and MCP server as thin entry points sharing the core library.

**Tech Stack:** Python 3.11+, Click (CLI), FastMCP (MCP server), keyring (secrets), ffmpeg (media), mlx-whisper / whisperx / openai (backends), pytest (testing), ruff + pyright (linting/types)

**Spec:** `docs/superpowers/specs/2026-03-19-scriba-design.md`

---

## Review Fixes (apply during implementation)

The following corrections from plan review MUST be applied when implementing the tasks below. Each fix references the task it modifies.

### RF-1: Extract shared utilities (DRY) — affects Tasks 6, 11, 12, 13, 14

Create `src/scriba/formatting.py` with shared timestamp formatting and SRT/VTT generation:
- Move `_fmt_srt()`, `_fmt_vtt()`, `_fmt_ts_text()` here as public functions `fmt_srt_ts()`, `fmt_vtt_ts()`, `fmt_text_ts()`
- Move `generate_srt()` and `generate_vtt()` from `media/subtitle.py` here
- `output/formatter.py` calls `generate_srt()` / `generate_vtt()` from `formatting.py` instead of reimplementing
- `media/subtitle.py` imports from `formatting.py` for SRT/VTT generation

### RF-2: Single source of truth for backend metadata — affects Tasks 6, 7, 8, 13, 14

Add to each backend adapter class:
```python
class MlxWhisperBackend:
    name = "mlx_whisper"
    models = ["tiny", "base", "small", "medium", "large-v3"]
    supports_diarize = False
    diarize_models: set[str] = set()
```
Create a shared function in `backends/__init__.py`:
```python
def backend_to_info(adapter: BackendAdapter) -> BackendInfo:
    return BackendInfo(
        name=adapter.name, available=adapter.is_available(),
        models=adapter.models, supports_diarize=adapter.supports_diarize,
        diarize_models=adapter.diarize_models,
    )
```
Remove duplicated `_backend_to_info` from `mcp/server.py` and `_to_backend_infos` from `cli.py`. Both call `backend_to_info()` instead.

### RF-3: `ConstraintConflict` inherits `ScribaError` — affects Task 7

Change `class ConstraintConflict(Exception)` to `class ConstraintConflict(ScribaError)` in `router/constraints.py`. Import `ScribaError` from `scriba.errors`.

### RF-4: Add retry logic to OpenAI adapter — affects Task 10

Use `tenacity` in `openai_stt.py`:
```python
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

@retry(wait=wait_exponential_jitter(initial=1, max=20), stop=stop_after_attempt(3))
async def _call_api(client, **kwargs):
    return await client.audio.transcriptions.create(**kwargs)
```
Call `_call_api` instead of `client.audio.transcriptions.create` directly.

### RF-5: Guard `discover_backends()` imports — affects Task 8

Wrap each import in `discover_backends()` with `try/except ImportError`:
```python
def discover_backends() -> list[BackendAdapter]:
    backends = []
    from scriba.backends.mlx_whisper import MlxWhisperBackend
    backends.append(MlxWhisperBackend())
    try:
        from scriba.backends.whisperx import WhisperXBackend
        backends.append(WhisperXBackend())
    except ImportError:
        pass
    try:
        from scriba.backends.openai_stt import OpenAISTTBackend
        backends.append(OpenAISTTBackend())
    except ImportError:
        pass
    return backends
```

### RF-6: Add missing tests — affects Tasks 5, 11, 13

- **Task 5**: Add `test_compress_for_upload` test to `test_ingest.py`
- **Task 11**: Add `test_format_json_enriched` test verifying `enrichment_available` in output
- **Task 13**: Add mock-backend e2e tests for `transcribe` and `estimate` MCP tools

### RF-7: Implement overlapping segment merging — affects Task 12

In `generate_srt()` and `generate_vtt()`, detect overlapping segments with different speakers and merge into `[SPEAKER_00 & SPEAKER_01]: text` lines.

### RF-8: Add `--enrich` CLI flag — affects Task 14

Add `--enrich` flag to `scriba transcribe` that, when set, calls OpenAI for summarization after transcription (using the same pattern as `openai_insights.py` from speaker-transcript-system). Falls back to `diarized` tier with a warning if no OpenAI key available.

---

## File Structure

```
src/scriba/
  __init__.py              - Package version and top-level imports
  contracts.py             - All dataclasses: Segment, TranscriptionConfig, TranscriptionResult, Estimate, RoutingDecision
  errors.py                - ScribaError hierarchy (6 exception classes + ConstraintConflict)
  config.py                - Settings dataclass + TOML loader from ~/.config/scriba/config.toml
  formatting.py            - Shared timestamp formatting, SRT/VTT generation (RF-1)
  secrets/
    __init__.py            - Re-export resolve_secret()
    provider.py            - SecretsProvider protocol + resolve_secret() chain
    keychain.py            - KeychainProvider via keyring
    env.py                 - EnvProvider (env vars + .env file)
  media/
    __init__.py            - Re-export probe_media, extract_audio
    ingest.py              - ffprobe metadata extraction + ffmpeg audio normalization
    subtitle.py            - SRT/VTT generation from segments + ffmpeg subtitle burn-in
  router/
    __init__.py            - Re-export route()
    constraints.py         - Constraints dataclass + normalize/validate logic
    cost_model.py          - Pricing tables, time estimation, calibration read/write
    engine.py              - Router: filter → estimate → rank → select/explain
  backends/
    __init__.py            - Registry: discover_backends() returns available adapters
    base.py                - BackendAdapter Protocol definition
    mlx_whisper.py         - MLX Whisper adapter
    whisperx.py            - WhisperX + pyannote adapter
    openai_stt.py          - OpenAI STT adapter
  output/
    __init__.py            - Re-export format_result()
    formatter.py           - Tier filtering + format rendering (JSON/text/SRT/VTT/MD)
  mcp/
    __init__.py
    server.py              - FastMCP server with transcribe, estimate, backends tools
  cli.py                   - Click CLI: transcribe, estimate, backends, configure subcommands
plugin/
  plugin.json              - Claude Code plugin manifest
  skills/
    transcribe.md          - Skill teaching Claude intent → params mapping
  .mcp.json                - MCP server config for plugin
tests/
  conftest.py              - Shared fixtures: sample segments, results, mock backends
  fixtures/                - Short test audio/video files (WAV + MP4, 2-5 seconds)
  test_contracts.py        - Dataclass construction, defaults, serialization
  test_errors.py           - Error hierarchy, messages, recovery hints
  test_config.py           - TOML loading, defaults, missing file handling
  test_secrets.py          - Resolution chain, keychain mock, env fallback
  test_ingest.py           - ffprobe parsing, audio extraction, format detection
  test_subtitle.py         - SRT/VTT generation, overlap merging, video burn-in
  test_constraints.py      - Normalization, validation, conflict detection
  test_cost_model.py       - Price calculation, time estimation, calibration
  test_router.py           - End-to-end routing decisions, edge cases
  test_formatter.py        - All 5 output formats × 4 tiers
  test_backends/
    test_mlx.py            - MLX adapter (mocked mlx_whisper)
    test_whisperx.py       - WhisperX adapter (mocked whisperx)
    test_openai.py         - OpenAI adapter (mocked openai client)
  test_mcp/
    test_server.py         - Tool registration, param validation, e2e with mocked backends
  test_cli.py              - Click test runner for all subcommands
```

---

### Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/scriba/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `.gitignore`
- Create: `CLAUDE.md`
- Create: `.env.example`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "nyqst-scriba"
version = "0.1.0"
description = "Intent-aware transcription tool with local and cloud backends"
requires-python = ">=3.11"
readme = "README.md"
dependencies = [
    "anyio>=4.0",
    "click>=8.0",
    "keyring>=25.0",
    "mcp>=1.15",
]

[project.optional-dependencies]
mlx = ["mlx-whisper>=0.4"]
whisperx = ["whisperx>=3.7", "pyannote.audio>=3.4"]
openai = ["openai>=1.50", "tenacity>=8.2"]
all = ["nyqst-scriba[mlx,whisperx,openai]"]
dev = ["pytest>=8", "pytest-asyncio>=0.23", "pytest-mock>=3", "ruff>=0.4", "pyright>=1.1"]

[project.scripts]
scriba = "scriba.cli:main"
scriba-mcp = "scriba.mcp.server:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
target-version = "py311"
line-length = 120

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Create package init, test init, conftest, .gitignore, CLAUDE.md, .env.example**

`src/scriba/__init__.py`:
```python
"""NYQST-Scriba: Intent-aware transcription tool."""
__version__ = "0.1.0"
```

`tests/__init__.py`: empty file.

`tests/conftest.py`:
```python
"""Shared test fixtures for Scriba."""
```

`.gitignore`:
```
__pycache__/
*.pyc
.venv/
*.egg-info/
dist/
build/
.pytest_cache/
.ruff_cache/
.env
```

`CLAUDE.md`:
```markdown
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
```

`.env.example`:
```
# Optional: set these instead of using `scriba configure` + keychain
# SCRIBA_OPENAI_API_KEY=sk-...
# SCRIBA_HF_TOKEN=hf_...
```

- [ ] **Step 3: Verify the package installs**

Run: `cd /Users/markforster/temp && uv pip install -e ".[dev]" 2>&1 | tail -5`
Expected: successful install with no errors.

- [ ] **Step 4: Verify pytest runs (0 tests)**

Run: `cd /Users/markforster/temp && python -m pytest --co -q`
Expected: `no tests ran`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/ tests/ .gitignore CLAUDE.md .env.example
git commit -m "feat: project scaffold with pyproject.toml and dev setup"
```

---

### Task 2: Contracts and Error Hierarchy

**Files:**
- Create: `src/scriba/contracts.py`
- Create: `src/scriba/errors.py`
- Create: `tests/test_contracts.py`
- Create: `tests/test_errors.py`

- [ ] **Step 1: Write failing tests for contracts**

`tests/test_contracts.py`:
```python
from scriba.contracts import (
    Segment,
    TranscriptionConfig,
    TranscriptionResult,
    Estimate,
    RoutingDecision,
)


def test_segment_defaults():
    s = Segment(start=0.0, end=1.5, text="hello")
    assert s.speaker is None


def test_segment_with_speaker():
    s = Segment(start=0.0, end=1.5, text="hello", speaker="SPEAKER_00")
    assert s.speaker == "SPEAKER_00"


def test_transcription_config_defaults():
    c = TranscriptionConfig(model="large-v3")
    assert c.language is None
    assert c.diarize is False
    assert c.speakers is None
    assert c.output_tier == "timestamped"


def test_transcription_result_defaults():
    r = TranscriptionResult(
        text="hello",
        segments=[],
        duration_seconds=1.0,
        model_used="large-v3",
        backend="mlx_whisper",
        cost_cents=0.0,
        diarized=False,
    )
    assert r.enrichment_available is False
    assert r.routing is None


def test_estimate_fields():
    e = Estimate(
        backend="mlx_whisper",
        model="large-v3",
        time_seconds=5.0,
        cost_cents=0.0,
        available=True,
        recommended=True,
        reason_unavailable=None,
    )
    assert e.recommended is True


def test_routing_decision():
    selected = Estimate(
        backend="mlx_whisper",
        model="large-v3",
        time_seconds=5.0,
        cost_cents=0.0,
        available=True,
        recommended=True,
        reason_unavailable=None,
    )
    rd = RoutingDecision(selected=selected, alternatives=[], trade_offs=None)
    assert rd.trade_offs is None
    assert rd.alternatives == []


def test_routing_decision_with_trade_offs():
    selected = Estimate(
        backend="openai_stt",
        model="gpt-4o-transcribe",
        time_seconds=10.0,
        cost_cents=12.0,
        available=True,
        recommended=True,
        reason_unavailable=None,
    )
    alt = Estimate(
        backend="whisperx",
        model="large-v3",
        time_seconds=60.0,
        cost_cents=0.0,
        available=True,
        recommended=False,
        reason_unavailable=None,
    )
    rd = RoutingDecision(
        selected=selected,
        alternatives=[alt],
        trade_offs=["Cloud selected to meet 30s timeout; local would take ~60s"],
    )
    assert len(rd.trade_offs) == 1
    assert len(rd.alternatives) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_contracts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scriba.contracts'`

- [ ] **Step 3: Implement contracts.py**

`src/scriba/contracts.py`:
```python
"""Data models for Scriba transcription pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass
class TranscriptionConfig:
    """Resolved configuration passed to a backend adapter."""
    model: str
    language: str | None = None
    diarize: bool = False
    speakers: int | None = None
    output_tier: str = "timestamped"


@dataclass
class Estimate:
    backend: str
    model: str
    time_seconds: float
    cost_cents: float
    available: bool
    recommended: bool
    reason_unavailable: str | None = None


@dataclass
class RoutingDecision:
    """Returned by the router engine."""
    selected: Estimate
    alternatives: list[Estimate] = field(default_factory=list)
    trade_offs: list[str] | None = None


@dataclass
class TranscriptionResult:
    text: str
    segments: list[Segment]
    duration_seconds: float
    model_used: str
    backend: str
    cost_cents: float
    diarized: bool
    enrichment_available: bool = False
    routing: RoutingDecision | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_contracts.py -v`
Expected: all 7 tests PASS

- [ ] **Step 5: Write failing tests for errors**

`tests/test_errors.py`:
```python
from scriba.errors import (
    ScribaError,
    DependencyMissing,
    BackendError,
    AudioError,
    SecretsError,
    RoutingError,
    BudgetExceeded,
)


def test_hierarchy():
    """All custom errors inherit from ScribaError."""
    for cls in [DependencyMissing, BackendError, AudioError, SecretsError, RoutingError, BudgetExceeded]:
        err = cls("test")
        assert isinstance(err, ScribaError)
        assert isinstance(err, Exception)


def test_dependency_missing_with_hint():
    err = DependencyMissing("ffmpeg", hint="brew install ffmpeg")
    assert "ffmpeg" in str(err)
    assert err.hint == "brew install ffmpeg"


def test_backend_error_wraps_cause():
    cause = RuntimeError("OOM")
    err = BackendError("mlx_whisper", cause=cause, suggestion="try a smaller model")
    assert err.backend == "mlx_whisper"
    assert err.cause is cause
    assert err.suggestion == "try a smaller model"


def test_routing_error_lists_missing():
    err = RoutingError(missing=["mlx-whisper", "whisperx", "openai"])
    assert len(err.missing) == 3


def test_budget_exceeded():
    err = BudgetExceeded(estimated_cents=25.0, budget_cents=10)
    assert err.estimated_cents == 25.0
    assert err.budget_cents == 10
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `python -m pytest tests/test_errors.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 7: Implement errors.py**

`src/scriba/errors.py`:
```python
"""Scriba error hierarchy with actionable recovery messages."""
from __future__ import annotations


class ScribaError(Exception):
    """Base error for all Scriba operations."""


class DependencyMissing(ScribaError):
    def __init__(self, name: str, *, hint: str | None = None):
        self.name = name
        self.hint = hint
        msg = f"Missing dependency: {name}"
        if hint:
            msg += f" — {hint}"
        super().__init__(msg)


class BackendError(ScribaError):
    def __init__(self, backend: str, *, cause: Exception | None = None, suggestion: str | None = None):
        self.backend = backend
        self.cause = cause
        self.suggestion = suggestion
        msg = f"Backend '{backend}' failed"
        if cause:
            msg += f": {cause}"
        if suggestion:
            msg += f" — {suggestion}"
        super().__init__(msg)


class AudioError(ScribaError):
    pass


class SecretsError(ScribaError):
    pass


class RoutingError(ScribaError):
    def __init__(self, msg: str = "No backends available", *, missing: list[str] | None = None):
        self.missing = missing or []
        if self.missing:
            msg += f". Missing: {', '.join(self.missing)}"
        super().__init__(msg)


class BudgetExceeded(ScribaError):
    def __init__(self, *, estimated_cents: float, budget_cents: int):
        self.estimated_cents = estimated_cents
        self.budget_cents = budget_cents
        super().__init__(
            f"Estimated cost {estimated_cents:.1f}c exceeds budget {budget_cents}c"
        )
```

- [ ] **Step 8: Run all tests**

Run: `python -m pytest tests/test_contracts.py tests/test_errors.py -v`
Expected: all 12 tests PASS

- [ ] **Step 9: Commit**

```bash
git add src/scriba/contracts.py src/scriba/errors.py tests/test_contracts.py tests/test_errors.py
git commit -m "feat: add contracts (data models) and error hierarchy"
```

---

### Task 3: Configuration Loader

**Files:**
- Create: `src/scriba/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

`tests/test_config.py`:
```python
import textwrap
from pathlib import Path

from scriba.config import ScribaConfig, load_config


def test_default_config():
    cfg = ScribaConfig()
    assert cfg.quality == "balanced"
    assert cfg.output_tier == "timestamped"
    assert cfg.output_format == "json"
    assert cfg.diarize is False
    assert cfg.prefer_local is True
    assert cfg.max_local_concurrency == 1
    assert cfg.max_cloud_concurrency == 3
    assert cfg.openai_model == "gpt-4o-mini-transcribe"
    assert cfg.mlx_model == "large-v3"
    assert cfg.whisperx_model == "large-v3"
    assert cfg.max_budget_cents_per_job == 50


def test_load_config_missing_file(tmp_path: Path):
    cfg = load_config(tmp_path / "nonexistent.toml")
    assert cfg == ScribaConfig()


def test_load_config_from_toml(tmp_path: Path):
    toml_file = tmp_path / "config.toml"
    toml_file.write_text(textwrap.dedent("""\
        [defaults]
        quality = "high"
        diarize = true

        [backends]
        prefer_local = false

        [concurrency]
        max_cloud = 5

        [openai]
        model = "gpt-4o-transcribe"
        max_budget_cents_per_job = 100

        [mlx]
        default_model = "medium"
    """))
    cfg = load_config(toml_file)
    assert cfg.quality == "high"
    assert cfg.diarize is True
    assert cfg.prefer_local is False
    assert cfg.max_cloud_concurrency == 5
    assert cfg.openai_model == "gpt-4o-transcribe"
    assert cfg.max_budget_cents_per_job == 100
    assert cfg.mlx_model == "medium"
    # Unchanged defaults
    assert cfg.output_format == "json"
    assert cfg.whisperx_model == "large-v3"


def test_load_config_partial_toml(tmp_path: Path):
    """A TOML with only some sections should merge with defaults."""
    toml_file = tmp_path / "config.toml"
    toml_file.write_text("[defaults]\nquality = \"fast\"\n")
    cfg = load_config(toml_file)
    assert cfg.quality == "fast"
    assert cfg.output_tier == "timestamped"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement config.py**

`src/scriba/config.py`:
```python
"""Configuration loader for Scriba."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "scriba" / "config.toml"


@dataclass
class ScribaConfig:
    # [defaults]
    quality: str = "balanced"
    output_tier: str = "timestamped"
    output_format: str = "json"
    diarize: bool = False
    # [backends]
    prefer_local: bool = True
    # [concurrency]
    max_local_concurrency: int = 1
    max_cloud_concurrency: int = 3
    # [calibration]
    calibration_path: str = "~/.config/scriba/calibration.json"
    calibration_max_samples: int = 10
    calibration_stale_days: int = 30
    # [openai]
    openai_model: str = "gpt-4o-mini-transcribe"
    max_budget_cents_per_job: int = 50
    # [mlx]
    mlx_model: str = "large-v3"
    mlx_cache_dir: str = "~/.cache/scriba/models"
    # [whisperx]
    whisperx_model: str = "large-v3"


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> ScribaConfig:
    """Load config from TOML file, falling back to defaults for missing keys."""
    if not path.exists():
        return ScribaConfig()
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    defaults = raw.get("defaults", {})
    backends = raw.get("backends", {})
    concurrency = raw.get("concurrency", {})
    calibration = raw.get("calibration", {})
    openai = raw.get("openai", {})
    mlx = raw.get("mlx", {})
    whisperx = raw.get("whisperx", {})
    return ScribaConfig(
        quality=defaults.get("quality", "balanced"),
        output_tier=defaults.get("output_tier", "timestamped"),
        output_format=defaults.get("output_format", "json"),
        diarize=defaults.get("diarize", False),
        prefer_local=backends.get("prefer_local", True),
        max_local_concurrency=concurrency.get("max_local", 1),
        max_cloud_concurrency=concurrency.get("max_cloud", 3),
        calibration_path=calibration.get("path", "~/.config/scriba/calibration.json"),
        calibration_max_samples=calibration.get("max_samples", 10),
        calibration_stale_days=calibration.get("stale_days", 30),
        openai_model=openai.get("model", "gpt-4o-mini-transcribe"),
        max_budget_cents_per_job=openai.get("max_budget_cents_per_job", 50),
        mlx_model=mlx.get("default_model", "large-v3"),
        mlx_cache_dir=mlx.get("cache_dir", "~/.cache/scriba/models"),
        whisperx_model=whisperx.get("default_model", "large-v3"),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_config.py -v`
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/scriba/config.py tests/test_config.py
git commit -m "feat: add TOML configuration loader with defaults"
```

---

### Task 4: Secrets Provider

**Files:**
- Create: `src/scriba/secrets/__init__.py`
- Create: `src/scriba/secrets/provider.py`
- Create: `src/scriba/secrets/keychain.py`
- Create: `src/scriba/secrets/env.py`
- Create: `tests/test_secrets.py`

- [ ] **Step 1: Write failing tests**

`tests/test_secrets.py`:
```python
import os
from unittest.mock import AsyncMock, patch

import pytest

from scriba.secrets.provider import SecretsChain
from scriba.secrets.env import EnvProvider
from scriba.secrets.keychain import KeychainProvider


@pytest.mark.asyncio
async def test_env_provider_reads_env_var():
    prov = EnvProvider(prefix="SCRIBA_")
    with patch.dict(os.environ, {"SCRIBA_OPENAI_API_KEY": "sk-test"}):
        val = await prov.get("openai-api-key")
    assert val == "sk-test"


@pytest.mark.asyncio
async def test_env_provider_returns_none_if_missing():
    prov = EnvProvider(prefix="SCRIBA_")
    with patch.dict(os.environ, {}, clear=True):
        val = await prov.get("openai-api-key")
    assert val is None


@pytest.mark.asyncio
async def test_keychain_provider_delegates_to_keyring():
    with patch("scriba.secrets.keychain.keyring") as mock_kr:
        mock_kr.get_password.return_value = "sk-from-keychain"
        prov = KeychainProvider(service="scriba")
        val = await prov.get("openai-api-key")
    assert val == "sk-from-keychain"
    mock_kr.get_password.assert_called_once_with("scriba", "openai-api-key")


@pytest.mark.asyncio
async def test_keychain_provider_returns_none():
    with patch("scriba.secrets.keychain.keyring") as mock_kr:
        mock_kr.get_password.return_value = None
        prov = KeychainProvider(service="scriba")
        val = await prov.get("openai-api-key")
    assert val is None


@pytest.mark.asyncio
async def test_chain_tries_in_order():
    """Chain tries keychain first, then env."""
    kc = AsyncMock()
    kc.get.return_value = None  # keychain misses
    env = AsyncMock()
    env.get.return_value = "sk-from-env"

    chain = SecretsChain([kc, env])
    val = await chain.get("openai-api-key")

    assert val == "sk-from-env"
    kc.get.assert_called_once_with("openai-api-key")
    env.get.assert_called_once_with("openai-api-key")


@pytest.mark.asyncio
async def test_chain_stops_on_first_hit():
    kc = AsyncMock()
    kc.get.return_value = "sk-from-keychain"
    env = AsyncMock()

    chain = SecretsChain([kc, env])
    val = await chain.get("openai-api-key")

    assert val == "sk-from-keychain"
    env.get.assert_not_called()


@pytest.mark.asyncio
async def test_chain_returns_none_if_all_miss():
    kc = AsyncMock()
    kc.get.return_value = None
    env = AsyncMock()
    env.get.return_value = None

    chain = SecretsChain([kc, env])
    val = await chain.get("openai-api-key")
    assert val is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_secrets.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement secrets modules**

`src/scriba/secrets/__init__.py`:
```python
"""Secrets management with keychain + env fallback."""
from scriba.secrets.provider import SecretsChain
from scriba.secrets.keychain import KeychainProvider
from scriba.secrets.env import EnvProvider

__all__ = ["SecretsChain", "KeychainProvider", "EnvProvider"]
```

`src/scriba/secrets/provider.py`:
```python
"""Secrets provider protocol and chain."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SecretsProvider(Protocol):
    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str) -> None: ...
    async def delete(self, key: str) -> None: ...


class SecretsChain:
    """Try providers in order, return first non-None result."""

    def __init__(self, providers: list[SecretsProvider]):
        self._providers = providers

    async def get(self, key: str) -> str | None:
        for provider in self._providers:
            value = await provider.get(key)
            if value is not None:
                return value
        return None

    async def set(self, key: str, value: str) -> None:
        if self._providers:
            await self._providers[0].set(key, value)

    async def delete(self, key: str) -> None:
        if self._providers:
            await self._providers[0].delete(key)
```

`src/scriba/secrets/keychain.py`:
```python
"""macOS Keychain (and cross-platform) secrets via keyring."""
from __future__ import annotations

import keyring


class KeychainProvider:
    def __init__(self, service: str = "scriba"):
        self._service = service

    async def get(self, key: str) -> str | None:
        return keyring.get_password(self._service, key)

    async def set(self, key: str, value: str) -> None:
        keyring.set_password(self._service, key, value)

    async def delete(self, key: str) -> None:
        try:
            keyring.delete_password(self._service, key)
        except keyring.errors.PasswordDeleteError:
            pass
```

`src/scriba/secrets/env.py`:
```python
"""Environment variable secrets provider."""
from __future__ import annotations

import os


class EnvProvider:
    def __init__(self, prefix: str = "SCRIBA_"):
        self._prefix = prefix

    def _env_key(self, key: str) -> str:
        return self._prefix + key.upper().replace("-", "_")

    async def get(self, key: str) -> str | None:
        return os.environ.get(self._env_key(key))

    async def set(self, key: str, value: str) -> None:
        os.environ[self._env_key(key)] = value

    async def delete(self, key: str) -> None:
        os.environ.pop(self._env_key(key), None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_secrets.py -v`
Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/scriba/secrets/ tests/test_secrets.py
git commit -m "feat: add secrets provider chain (keychain + env fallback)"
```

---

### Task 5: Media Ingester

**Files:**
- Create: `src/scriba/media/__init__.py`
- Create: `src/scriba/media/ingest.py`
- Create: `tests/test_ingest.py`
- Create: `tests/fixtures/` (with test audio generated via ffmpeg)

- [ ] **Step 1: Generate test fixtures**

```bash
# Generate a 2-second silent WAV for testing
ffmpeg -y -f lavfi -i "sine=frequency=440:duration=2" -ar 16000 -ac 1 tests/fixtures/test_tone.wav
# Generate a 2-second silent MP4 with audio for testing
ffmpeg -y -f lavfi -i "color=c=black:s=320x240:d=2" -f lavfi -i "sine=frequency=440:duration=2" -c:v libx264 -c:a aac -shortest tests/fixtures/test_video.mp4
```

- [ ] **Step 2: Write failing tests**

`tests/test_ingest.py`:
```python
from pathlib import Path

import pytest

from scriba.media.ingest import probe_media, extract_audio, MediaInfo
from scriba.errors import AudioError, DependencyMissing

FIXTURES = Path(__file__).parent / "fixtures"


def test_probe_wav():
    info = probe_media(FIXTURES / "test_tone.wav")
    assert info.duration_seconds == pytest.approx(2.0, abs=0.5)
    assert info.has_audio is True
    assert info.has_video is False
    assert info.file_size_bytes > 0


def test_probe_video():
    info = probe_media(FIXTURES / "test_video.mp4")
    assert info.has_audio is True
    assert info.has_video is True
    assert info.duration_seconds == pytest.approx(2.0, abs=0.5)


def test_probe_nonexistent():
    with pytest.raises(AudioError, match="not found"):
        probe_media(Path("/nonexistent/file.wav"))


def test_extract_audio_from_wav(tmp_path: Path):
    out = extract_audio(FIXTURES / "test_tone.wav", tmp_path / "out.wav")
    assert out.exists()
    assert out.stat().st_size > 0
    # Verify it's 16kHz mono
    info = probe_media(out)
    assert info.sample_rate == 16000
    assert info.channels == 1


def test_extract_audio_from_video(tmp_path: Path):
    out = extract_audio(FIXTURES / "test_video.mp4", tmp_path / "out.wav")
    assert out.exists()
    info = probe_media(out)
    assert info.has_audio is True
    assert info.has_video is False
    assert info.sample_rate == 16000
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_ingest.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement media ingester**

`src/scriba/media/__init__.py`:
```python
from scriba.media.ingest import probe_media, extract_audio, MediaInfo

__all__ = ["probe_media", "extract_audio", "MediaInfo"]
```

`src/scriba/media/ingest.py`:
```python
"""Media probing and audio extraction via ffmpeg/ffprobe."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from scriba.errors import AudioError, DependencyMissing


def _require_ffmpeg() -> None:
    if not shutil.which("ffprobe") or not shutil.which("ffmpeg"):
        raise DependencyMissing("ffmpeg", hint="brew install ffmpeg")


@dataclass
class MediaInfo:
    duration_seconds: float
    has_audio: bool
    has_video: bool
    file_size_bytes: int
    sample_rate: int | None = None
    channels: int | None = None
    codec: str | None = None


def probe_media(path: Path) -> MediaInfo:
    """Extract metadata from a media file using ffprobe."""
    _require_ffmpeg()
    if not path.exists():
        raise AudioError(f"File not found: {path}")
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AudioError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    has_video = any(s.get("codec_type") == "video" for s in streams)
    duration = float(fmt.get("duration", 0))

    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    sample_rate = int(audio_stream["sample_rate"]) if audio_stream and "sample_rate" in audio_stream else None
    channels = int(audio_stream["channels"]) if audio_stream and "channels" in audio_stream else None
    codec = audio_stream.get("codec_name") if audio_stream else None

    return MediaInfo(
        duration_seconds=duration,
        has_audio=has_audio,
        has_video=has_video,
        file_size_bytes=path.stat().st_size,
        sample_rate=sample_rate,
        channels=channels,
        codec=codec,
    )


def extract_audio(input_path: Path, output_path: Path) -> Path:
    """Extract and normalize audio to 16kHz mono WAV."""
    _require_ffmpeg()
    if not input_path.exists():
        raise AudioError(f"File not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AudioError(f"Audio extraction failed: {result.stderr.strip()}")
    return output_path


def compress_for_upload(input_path: Path, output_path: Path, *, bitrate: str = "24k") -> Path:
    """Compress audio to low-bitrate MP3 for cloud upload."""
    _require_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", bitrate,
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AudioError(f"Compression failed: {result.stderr.strip()}")
    return output_path
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_ingest.py -v`
Expected: all 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/scriba/media/ tests/test_ingest.py tests/fixtures/
git commit -m "feat: add media ingester (ffprobe + ffmpeg audio extraction)"
```

---

### Task 6: Cost Model

**Files:**
- Create: `src/scriba/router/__init__.py`
- Create: `src/scriba/router/cost_model.py`
- Create: `tests/test_cost_model.py`

- [ ] **Step 1: Write failing tests**

`tests/test_cost_model.py`:
```python
import json
from pathlib import Path

import pytest

from scriba.router.cost_model import (
    estimate_cost_cents,
    estimate_time_seconds,
    PRICING,
    TIME_MULTIPLIERS,
    load_calibration,
    save_calibration_entry,
)


def test_pricing_table_has_all_models():
    assert "openai_stt:whisper-1" in PRICING
    assert "openai_stt:gpt-4o-mini-transcribe" in PRICING
    assert "openai_stt:gpt-4o-transcribe" in PRICING


def test_local_cost_is_zero():
    assert estimate_cost_cents("mlx_whisper", "large-v3", duration_minutes=60) == 0.0
    assert estimate_cost_cents("whisperx", "large-v3", duration_minutes=60) == 0.0


def test_openai_cost():
    cost = estimate_cost_cents("openai_stt", "whisper-1", duration_minutes=10)
    assert cost == pytest.approx(6.0)  # $0.006/min * 10min = 6 cents


def test_openai_mini_cost():
    cost = estimate_cost_cents("openai_stt", "gpt-4o-mini-transcribe", duration_minutes=10)
    assert cost == pytest.approx(3.0)  # $0.003/min * 10min


def test_time_estimate_mlx_tiny():
    t = estimate_time_seconds("mlx_whisper", "tiny", duration_seconds=600)
    assert t == pytest.approx(30.0)  # 0.05 * 600


def test_time_estimate_whisperx_diarize():
    t = estimate_time_seconds("whisperx", "large-v3", duration_seconds=600)
    assert t == pytest.approx(300.0)  # 0.50 * 600


def test_calibration_round_trip(tmp_path: Path):
    cal_path = tmp_path / "calibration.json"
    save_calibration_entry(cal_path, "mlx_whisper", "large-v3", audio_duration=60.0, wall_clock=20.0)
    save_calibration_entry(cal_path, "mlx_whisper", "large-v3", audio_duration=120.0, wall_clock=35.0)

    cal = load_calibration(cal_path)
    key = "mlx_whisper:large-v3"
    assert key in cal
    assert len(cal[key]) == 2


def test_calibration_missing_file(tmp_path: Path):
    cal = load_calibration(tmp_path / "nonexistent.json")
    assert cal == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cost_model.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement cost_model.py**

`src/scriba/router/__init__.py`:
```python
"""Constraint-based transcription router."""
```

`src/scriba/router/cost_model.py`:
```python
"""Pricing tables and time estimation for transcription backends."""
from __future__ import annotations

import json
import time
from pathlib import Path

# Cost per minute in cents
PRICING: dict[str, float] = {
    "mlx_whisper:tiny": 0.0,
    "mlx_whisper:base": 0.0,
    "mlx_whisper:small": 0.0,
    "mlx_whisper:medium": 0.0,
    "mlx_whisper:large-v3": 0.0,
    "whisperx:tiny": 0.0,
    "whisperx:base": 0.0,
    "whisperx:small": 0.0,
    "whisperx:medium": 0.0,
    "whisperx:large-v3": 0.0,
    "openai_stt:whisper-1": 0.6,
    "openai_stt:gpt-4o-mini-transcribe": 0.3,
    "openai_stt:gpt-4o-transcribe": 0.6,
}

# Time as fraction of audio realtime (e.g. 0.3 means 1 min audio takes 18s)
TIME_MULTIPLIERS: dict[str, float] = {
    "mlx_whisper:tiny": 0.05,
    "mlx_whisper:base": 0.08,
    "mlx_whisper:small": 0.12,
    "mlx_whisper:medium": 0.20,
    "mlx_whisper:large-v3": 0.30,
    "whisperx:tiny": 0.15,
    "whisperx:base": 0.20,
    "whisperx:small": 0.25,
    "whisperx:medium": 0.35,
    "whisperx:large-v3": 0.50,
    "openai_stt:whisper-1": 0.15,
    "openai_stt:gpt-4o-mini-transcribe": 0.15,
    "openai_stt:gpt-4o-transcribe": 0.15,
}


def estimate_cost_cents(backend: str, model: str, *, duration_minutes: float) -> float:
    key = f"{backend}:{model}"
    rate = PRICING.get(key, 0.0)
    return rate * duration_minutes


def estimate_time_seconds(backend: str, model: str, *, duration_seconds: float) -> float:
    key = f"{backend}:{model}"
    multiplier = TIME_MULTIPLIERS.get(key, 1.0)
    return multiplier * duration_seconds


def load_calibration(path: Path) -> dict[str, list[dict]]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_calibration_entry(
    path: Path,
    backend: str,
    model: str,
    *,
    audio_duration: float,
    wall_clock: float,
    max_samples: int = 10,
    stale_days: int = 30,
) -> None:
    data = load_calibration(path)
    key = f"{backend}:{model}"
    entries = data.get(key, [])

    # Discard stale entries
    cutoff = time.time() - (stale_days * 86400)
    entries = [e for e in entries if e.get("ts", 0) > cutoff]

    entries.append({
        "audio_duration": audio_duration,
        "wall_clock": wall_clock,
        "ts": time.time(),
    })

    # Keep only the last max_samples
    entries = entries[-max_samples:]
    data[key] = entries

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cost_model.py -v`
Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/scriba/router/ tests/test_cost_model.py
git commit -m "feat: add cost model with pricing tables and calibration"
```

---

### Task 7: Constraints and Router Engine

**Files:**
- Create: `src/scriba/router/constraints.py`
- Create: `src/scriba/router/engine.py`
- Create: `tests/test_constraints.py`
- Create: `tests/test_router.py`

- [ ] **Step 1: Write failing tests for constraints**

`tests/test_constraints.py`:
```python
import pytest

from scriba.router.constraints import Constraints, normalize, validate, ConstraintConflict


def test_defaults():
    c = Constraints()
    assert c.quality == "balanced"
    assert c.diarize is False
    assert c.budget_cents is None


def test_normalize_diarized_tier_implies_diarize():
    c = Constraints(output_tier="diarized", diarize=False)
    n = normalize(c)
    assert n.diarize is True


def test_normalize_enriched_tier_implies_diarize():
    c = Constraints(output_tier="enriched")
    n = normalize(c)
    assert n.diarize is True


def test_validate_diarize_false_with_diarized_tier():
    """Explicit diarize=False + output_tier=diarized is a conflict."""
    c = Constraints(output_tier="diarized", diarize=False, _diarize_explicit=True)
    with pytest.raises(ConstraintConflict, match="diarize"):
        validate(c)


def test_validate_fast_diarize_conflict():
    c = Constraints(quality="fast", diarize=True)
    with pytest.raises(ConstraintConflict, match="fast.*diariz"):
        validate(c)


def test_validate_ok():
    c = Constraints(quality="high", diarize=True)
    validate(c)  # should not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_constraints.py -v`
Expected: FAIL

- [ ] **Step 3: Implement constraints.py**

`src/scriba/router/constraints.py`:
```python
"""Constraint model for routing decisions."""
from __future__ import annotations

from dataclasses import dataclass


class ConstraintConflict(Exception):
    """Raised when constraints are contradictory."""


@dataclass
class Constraints:
    quality: str = "balanced"
    budget_cents: int | None = None
    timeout_seconds: int | None = None
    diarize: bool = False
    speakers: int | None = None
    output_tier: str = "timestamped"
    language: str | None = None
    # Internal: tracks whether diarize was explicitly set by the user
    _diarize_explicit: bool = False


def normalize(c: Constraints) -> Constraints:
    """Resolve parameter interactions."""
    diarize = c.diarize
    if c.output_tier in ("diarized", "enriched") and not c.diarize:
        if c._diarize_explicit:
            # Will be caught by validate()
            return c
        diarize = True
    return Constraints(
        quality=c.quality,
        budget_cents=c.budget_cents,
        timeout_seconds=c.timeout_seconds,
        diarize=diarize,
        speakers=c.speakers,
        output_tier=c.output_tier,
        language=c.language,
        _diarize_explicit=c._diarize_explicit,
    )


def validate(c: Constraints) -> None:
    """Raise ConstraintConflict if constraints are contradictory."""
    if c._diarize_explicit and not c.diarize and c.output_tier in ("diarized", "enriched"):
        raise ConstraintConflict(
            f"Cannot set diarize=false with output_tier={c.output_tier}"
        )
    if c.quality == "fast" and c.diarize:
        raise ConstraintConflict(
            "fast quality with diarization is not available — diarization is inherently slower"
        )
```

- [ ] **Step 4: Run constraint tests**

Run: `python -m pytest tests/test_constraints.py -v`
Expected: all 6 tests PASS

- [ ] **Step 5: Write failing tests for router engine**

`tests/test_router.py`:
```python
import pytest

from scriba.contracts import Estimate, RoutingDecision
from scriba.router.constraints import Constraints
from scriba.router.engine import route, BackendInfo


def _mlx_backend(available: bool = True) -> BackendInfo:
    return BackendInfo(
        name="mlx_whisper",
        available=available,
        models=["tiny", "base", "small", "medium", "large-v3"],
        supports_diarize=False,
    )


def _whisperx_backend(available: bool = True) -> BackendInfo:
    return BackendInfo(
        name="whisperx",
        available=available,
        models=["tiny", "base", "small", "medium", "large-v3"],
        supports_diarize=True,
    )


def _openai_backend(available: bool = True) -> BackendInfo:
    return BackendInfo(
        name="openai_stt",
        available=available,
        models=["whisper-1", "gpt-4o-mini-transcribe", "gpt-4o-transcribe"],
        supports_diarize=True,
        diarize_models={"gpt-4o-mini-transcribe", "gpt-4o-transcribe"},
    )


def test_default_selects_mlx_balanced():
    constraints = Constraints()
    decision = route(constraints, duration_seconds=60, backends=[_mlx_backend(), _whisperx_backend()])
    assert decision.selected.backend == "mlx_whisper"
    assert decision.selected.model == "medium"  # balanced → medium


def test_diarize_selects_whisperx():
    constraints = Constraints(diarize=True, quality="balanced")
    decision = route(constraints, duration_seconds=60, backends=[_mlx_backend(), _whisperx_backend()])
    assert decision.selected.backend == "whisperx"


def test_diarize_excludes_mlx():
    constraints = Constraints(diarize=True)
    decision = route(constraints, duration_seconds=60, backends=[_mlx_backend()])
    # Only MLX available, no diarization possible
    assert decision.trade_offs is not None
    assert len(decision.trade_offs) > 0


def test_budget_zero_excludes_cloud():
    constraints = Constraints(budget_cents=0)
    decision = route(
        constraints, duration_seconds=60,
        backends=[_mlx_backend(), _openai_backend()],
    )
    assert decision.selected.backend == "mlx_whisper"


def test_high_quality_selects_large_model():
    constraints = Constraints(quality="high")
    decision = route(constraints, duration_seconds=60, backends=[_mlx_backend()])
    assert decision.selected.model == "large-v3"


def test_fast_selects_tiny():
    constraints = Constraints(quality="fast")
    decision = route(constraints, duration_seconds=60, backends=[_mlx_backend()])
    assert decision.selected.model == "tiny"


def test_timeout_prefers_faster_option():
    constraints = Constraints(timeout_seconds=5)
    decision = route(constraints, duration_seconds=60, backends=[_mlx_backend()])
    # With 60s audio and 5s timeout, should prefer tiny (0.05 * 60 = 3s)
    assert decision.selected.model == "tiny"


def test_openai_diarize_excludes_whisper1():
    constraints = Constraints(diarize=True, quality="balanced")
    decision = route(constraints, duration_seconds=60, backends=[_openai_backend()])
    assert decision.selected.model != "whisper-1"


def test_no_backends_available():
    from scriba.errors import RoutingError
    constraints = Constraints()
    with pytest.raises(RoutingError):
        route(constraints, duration_seconds=60, backends=[])


def test_all_backends_unavailable():
    from scriba.errors import RoutingError
    constraints = Constraints()
    with pytest.raises(RoutingError):
        route(constraints, duration_seconds=60, backends=[_mlx_backend(available=False)])
```

- [ ] **Step 6: Run router tests to verify they fail**

Run: `python -m pytest tests/test_router.py -v`
Expected: FAIL

- [ ] **Step 7: Implement router engine**

`src/scriba/router/engine.py`:
```python
"""Constraint-based backend router."""
from __future__ import annotations

from dataclasses import dataclass, field

from scriba.contracts import Estimate, RoutingDecision
from scriba.errors import RoutingError
from scriba.router.constraints import Constraints, normalize, validate, ConstraintConflict
from scriba.router.cost_model import estimate_cost_cents, estimate_time_seconds

# Quality → model mapping per backend
QUALITY_MODEL_MAP: dict[str, dict[str, str]] = {
    "mlx_whisper": {"fast": "tiny", "balanced": "medium", "high": "large-v3"},
    "whisperx": {"fast": "small", "balanced": "medium", "high": "large-v3"},
    "openai_stt": {"fast": "gpt-4o-mini-transcribe", "balanced": "gpt-4o-mini-transcribe", "high": "gpt-4o-transcribe"},
}


@dataclass
class BackendInfo:
    name: str
    available: bool
    models: list[str]
    supports_diarize: bool = False
    diarize_models: set[str] = field(default_factory=set)


def route(
    constraints: Constraints,
    *,
    duration_seconds: float,
    backends: list[BackendInfo],
    prefer_local: bool = True,
) -> RoutingDecision:
    """Select the best backend and model for the given constraints."""
    # Normalize
    try:
        constraints = normalize(constraints)
        validate(constraints)
    except ConstraintConflict as e:
        # Return best feasible with trade-off
        constraints = normalize(Constraints(
            quality=constraints.quality,
            budget_cents=constraints.budget_cents,
            timeout_seconds=constraints.timeout_seconds,
            diarize=False,
            output_tier="timestamped",
            language=constraints.language,
        ))
        fallback = _find_best(constraints, duration_seconds=duration_seconds, backends=backends, prefer_local=prefer_local)
        if fallback is None:
            raise RoutingError(missing=[b.name for b in backends if not b.available])
        return RoutingDecision(selected=fallback, alternatives=[], trade_offs=[str(e)])

    candidates = _build_candidates(constraints, duration_seconds=duration_seconds, backends=backends)

    if not candidates:
        available_names = [b.name for b in backends if b.available]
        if not available_names:
            raise RoutingError(missing=[b.name for b in backends])
        # All filtered out — return best with trade-offs
        relaxed = Constraints(quality=constraints.quality, language=constraints.language)
        fallback_candidates = _build_candidates(relaxed, duration_seconds=duration_seconds, backends=backends)
        if not fallback_candidates:
            raise RoutingError(missing=[b.name for b in backends if not b.available])
        best = _rank(fallback_candidates, constraints, prefer_local=prefer_local)[0]
        return RoutingDecision(
            selected=best,
            alternatives=[],
            trade_offs=[f"No backend supports all constraints. Relaxed to: {best.backend}:{best.model}"],
        )

    ranked = _rank(candidates, constraints, prefer_local=prefer_local)
    selected = ranked[0]
    alternatives = ranked[1:]

    return RoutingDecision(selected=selected, alternatives=alternatives, trade_offs=None)


def _find_best(
    constraints: Constraints,
    *,
    duration_seconds: float,
    backends: list[BackendInfo],
    prefer_local: bool,
) -> Estimate | None:
    candidates = _build_candidates(constraints, duration_seconds=duration_seconds, backends=backends)
    if not candidates:
        return None
    return _rank(candidates, constraints, prefer_local=prefer_local)[0]


def _build_candidates(
    constraints: Constraints,
    *,
    duration_seconds: float,
    backends: list[BackendInfo],
) -> list[Estimate]:
    duration_minutes = duration_seconds / 60.0
    candidates: list[Estimate] = []

    for backend in backends:
        if not backend.available:
            continue

        # Filter: diarization
        if constraints.diarize and not backend.supports_diarize:
            continue

        # Select model for quality
        model_map = QUALITY_MODEL_MAP.get(backend.name, {})
        model = model_map.get(constraints.quality)
        if model is None:
            continue

        # Filter: diarization model support (OpenAI whisper-1 doesn't support diarize)
        if constraints.diarize and backend.diarize_models and model not in backend.diarize_models:
            # Try to upgrade to a diarize-capable model
            for dm in backend.diarize_models:
                if dm in backend.models:
                    model = dm
                    break
            else:
                continue

        # Filter: budget
        cost = estimate_cost_cents(backend.name, model, duration_minutes=duration_minutes)
        if constraints.budget_cents is not None and cost > constraints.budget_cents:
            continue

        # Filter: timeout
        time_est = estimate_time_seconds(backend.name, model, duration_seconds=duration_seconds)
        if constraints.timeout_seconds is not None and time_est > constraints.timeout_seconds:
            continue

        candidates.append(Estimate(
            backend=backend.name,
            model=model,
            time_seconds=time_est,
            cost_cents=cost,
            available=True,
            recommended=False,
            reason_unavailable=None,
        ))

    return candidates


def _rank(
    candidates: list[Estimate],
    constraints: Constraints,
    *,
    prefer_local: bool,
) -> list[Estimate]:
    """Rank candidates. Lower score = better."""

    def score(e: Estimate) -> tuple[int, float, float]:
        locality = 0 if prefer_local and e.backend != "openai_stt" else 1
        return (locality, e.cost_cents, e.time_seconds)

    ranked = sorted(candidates, key=score)
    if ranked:
        ranked[0] = Estimate(
            backend=ranked[0].backend,
            model=ranked[0].model,
            time_seconds=ranked[0].time_seconds,
            cost_cents=ranked[0].cost_cents,
            available=ranked[0].available,
            recommended=True,
            reason_unavailable=ranked[0].reason_unavailable,
        )
    return ranked
```

- [ ] **Step 8: Run all router tests**

Run: `python -m pytest tests/test_constraints.py tests/test_router.py -v`
Expected: all 16 tests PASS

- [ ] **Step 9: Commit**

```bash
git add src/scriba/router/ tests/test_constraints.py tests/test_router.py
git commit -m "feat: add constraint-based router with cost model integration"
```

---

### Task 8: Backend Adapters (Protocol + MLX Whisper)

**Files:**
- Create: `src/scriba/backends/__init__.py`
- Create: `src/scriba/backends/base.py`
- Create: `src/scriba/backends/mlx_whisper.py`
- Create: `tests/test_backends/test_mlx.py`

- [ ] **Step 1: Write failing tests**

`tests/test_backends/__init__.py`: empty file.

`tests/test_backends/test_mlx.py`:
```python
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from scriba.backends.mlx_whisper import MlxWhisperBackend
from scriba.contracts import TranscriptionConfig


def test_is_available_when_installed():
    with patch("scriba.backends.mlx_whisper._import_mlx_whisper", return_value=MagicMock()):
        backend = MlxWhisperBackend()
        assert backend.is_available() is True


def test_is_unavailable_when_not_installed():
    with patch("scriba.backends.mlx_whisper._import_mlx_whisper", return_value=None):
        backend = MlxWhisperBackend()
        assert backend.is_available() is False


@pytest.mark.asyncio
async def test_transcribe_returns_result():
    mock_mlx = MagicMock()
    mock_mlx.transcribe.return_value = {
        "text": "hello world",
        "segments": [
            {"start": 0.0, "end": 1.5, "text": "hello world"},
        ],
    }

    with patch("scriba.backends.mlx_whisper._import_mlx_whisper", return_value=mock_mlx):
        backend = MlxWhisperBackend()
        config = TranscriptionConfig(model="tiny")
        result = await backend.transcribe(Path("/fake/audio.wav"), config)

    assert result.text == "hello world"
    assert result.backend == "mlx_whisper"
    assert result.cost_cents == 0.0
    assert result.diarized is False
    assert len(result.segments) == 1
    assert result.segments[0].start == 0.0
    assert result.segments[0].speaker is None


def test_estimate():
    with patch("scriba.backends.mlx_whisper._import_mlx_whisper", return_value=MagicMock()):
        backend = MlxWhisperBackend()
        config = TranscriptionConfig(model="tiny")
        est = backend.estimate(60.0, config)
    assert est.backend == "mlx_whisper"
    assert est.cost_cents == 0.0
    assert est.time_seconds > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_backends/test_mlx.py -v`
Expected: FAIL

- [ ] **Step 3: Implement backend base and MLX adapter**

`src/scriba/backends/__init__.py`:
```python
"""Backend adapter registry."""
from __future__ import annotations

from scriba.backends.base import BackendAdapter


def discover_backends() -> list[BackendAdapter]:
    """Return all backend adapters, regardless of availability."""
    from scriba.backends.mlx_whisper import MlxWhisperBackend
    from scriba.backends.whisperx import WhisperXBackend
    from scriba.backends.openai_stt import OpenAISTTBackend
    return [MlxWhisperBackend(), WhisperXBackend(), OpenAISTTBackend()]


__all__ = ["BackendAdapter", "discover_backends"]
```

`src/scriba/backends/base.py`:
```python
"""Backend adapter protocol."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from scriba.contracts import Estimate, TranscriptionConfig, TranscriptionResult


@runtime_checkable
class BackendAdapter(Protocol):
    name: str

    async def transcribe(self, audio_path: Path, config: TranscriptionConfig) -> TranscriptionResult: ...
    def estimate(self, duration_seconds: float, config: TranscriptionConfig) -> Estimate: ...
    def is_available(self) -> bool: ...
```

`src/scriba/backends/mlx_whisper.py`:
```python
"""MLX Whisper backend adapter."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from scriba.contracts import Estimate, Segment, TranscriptionConfig, TranscriptionResult
from scriba.router.cost_model import estimate_cost_cents, estimate_time_seconds


def _import_mlx_whisper() -> Any | None:
    try:
        import mlx_whisper
        return mlx_whisper
    except ImportError:
        return None


class MlxWhisperBackend:
    name = "mlx_whisper"

    def is_available(self) -> bool:
        return _import_mlx_whisper() is not None

    def estimate(self, duration_seconds: float, config: TranscriptionConfig) -> Estimate:
        return Estimate(
            backend=self.name,
            model=config.model,
            time_seconds=estimate_time_seconds(self.name, config.model, duration_seconds=duration_seconds),
            cost_cents=estimate_cost_cents(self.name, config.model, duration_minutes=duration_seconds / 60),
            available=self.is_available(),
            recommended=False,
        )

    async def transcribe(self, audio_path: Path, config: TranscriptionConfig) -> TranscriptionResult:
        mlx_whisper = _import_mlx_whisper()
        if mlx_whisper is None:
            from scriba.errors import DependencyMissing
            raise DependencyMissing("mlx-whisper", hint='pip install "nyqst-scriba[mlx]"')

        raw = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=f"mlx-community/whisper-{config.model}-mlx",
            language=config.language,
        )

        segments = [
            Segment(
                start=s.get("start", 0.0),
                end=s.get("end", 0.0),
                text=s.get("text", "").strip(),
            )
            for s in raw.get("segments", [])
            if s.get("text", "").strip()
        ]

        return TranscriptionResult(
            text=raw.get("text", "").strip(),
            segments=segments,
            duration_seconds=sum(s.end - s.start for s in segments) if segments else 0.0,
            model_used=config.model,
            backend=self.name,
            cost_cents=0.0,
            diarized=False,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_backends/test_mlx.py -v`
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/scriba/backends/ tests/test_backends/
git commit -m "feat: add backend protocol and MLX Whisper adapter"
```

---

### Task 9: WhisperX Backend Adapter

**Files:**
- Create: `src/scriba/backends/whisperx.py`
- Create: `tests/test_backends/test_whisperx.py`

- [ ] **Step 1: Write failing tests**

`tests/test_backends/test_whisperx.py`:
```python
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from scriba.backends.whisperx import WhisperXBackend
from scriba.contracts import TranscriptionConfig


def test_is_available():
    with patch("scriba.backends.whisperx._import_whisperx", return_value=MagicMock()):
        assert WhisperXBackend().is_available() is True


def test_is_unavailable():
    with patch("scriba.backends.whisperx._import_whisperx", return_value=None):
        assert WhisperXBackend().is_available() is False


@pytest.mark.asyncio
async def test_transcribe_with_diarization():
    mock_wx = MagicMock()
    mock_model = MagicMock()
    mock_wx.load_model.return_value = mock_model
    mock_model.transcribe.return_value = {
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "hello", "speaker": "SPEAKER_00"},
            {"start": 2.5, "end": 4.0, "text": "world", "speaker": "SPEAKER_01"},
        ],
    }
    mock_wx.load_audio.return_value = MagicMock()
    mock_wx.load_align_model.return_value = (MagicMock(), MagicMock())
    mock_wx.align.return_value = mock_model.transcribe.return_value
    mock_diarize = MagicMock()
    mock_diarize_pipeline = MagicMock()
    mock_wx.DiarizationPipeline.return_value = mock_diarize_pipeline
    mock_wx.assign_word_speakers.return_value = mock_model.transcribe.return_value

    with patch("scriba.backends.whisperx._import_whisperx", return_value=mock_wx):
        backend = WhisperXBackend()
        config = TranscriptionConfig(model="large-v3", diarize=True, speakers=2)
        result = await backend.transcribe(Path("/fake/audio.wav"), config)

    assert result.diarized is True
    assert result.backend == "whisperx"
    assert len(result.segments) == 2
    assert result.segments[0].speaker == "SPEAKER_00"
    assert result.segments[1].speaker == "SPEAKER_01"


def test_estimate():
    with patch("scriba.backends.whisperx._import_whisperx", return_value=MagicMock()):
        backend = WhisperXBackend()
        config = TranscriptionConfig(model="large-v3", diarize=True)
        est = backend.estimate(60.0, config)
    assert est.backend == "whisperx"
    assert est.cost_cents == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_backends/test_whisperx.py -v`
Expected: FAIL

- [ ] **Step 3: Implement WhisperX adapter**

`src/scriba/backends/whisperx.py`:
```python
"""WhisperX + pyannote backend adapter."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from scriba.contracts import Estimate, Segment, TranscriptionConfig, TranscriptionResult
from scriba.router.cost_model import estimate_cost_cents, estimate_time_seconds


def _import_whisperx() -> Any | None:
    try:
        import whisperx
        return whisperx
    except ImportError:
        return None


class WhisperXBackend:
    name = "whisperx"

    def is_available(self) -> bool:
        return _import_whisperx() is not None

    def estimate(self, duration_seconds: float, config: TranscriptionConfig) -> Estimate:
        return Estimate(
            backend=self.name,
            model=config.model,
            time_seconds=estimate_time_seconds(self.name, config.model, duration_seconds=duration_seconds),
            cost_cents=estimate_cost_cents(self.name, config.model, duration_minutes=duration_seconds / 60),
            available=self.is_available(),
            recommended=False,
        )

    async def transcribe(self, audio_path: Path, config: TranscriptionConfig) -> TranscriptionResult:
        wx = _import_whisperx()
        if wx is None:
            from scriba.errors import DependencyMissing
            raise DependencyMissing("whisperx", hint='pip install "nyqst-scriba[whisperx]"')

        import torch
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        compute_type = "float16" if device != "cpu" else "int8"

        model = wx.load_model(config.model, device, compute_type=compute_type, language=config.language)
        audio = wx.load_audio(str(audio_path))
        result = model.transcribe(audio, batch_size=16)

        # Align
        align_model, align_metadata = wx.load_align_model(
            language_code=config.language or result.get("language", "en"),
            device=device,
        )
        result = wx.align(result["segments"], align_model, align_metadata, audio, device)

        # Diarize if requested
        if config.diarize:
            diarize_model = wx.DiarizationPipeline(device=device)
            diarize_kwargs = {}
            if config.speakers:
                diarize_kwargs["num_speakers"] = config.speakers
            diarize_segments = diarize_model(audio, **diarize_kwargs)
            result = wx.assign_word_speakers(diarize_segments, result)

        segments = [
            Segment(
                start=s.get("start", 0.0),
                end=s.get("end", 0.0),
                text=s.get("text", "").strip(),
                speaker=s.get("speaker") if config.diarize else None,
            )
            for s in result.get("segments", [])
            if s.get("text", "").strip()
        ]

        full_text = " ".join(s.text for s in segments)

        return TranscriptionResult(
            text=full_text,
            segments=segments,
            duration_seconds=segments[-1].end if segments else 0.0,
            model_used=config.model,
            backend=self.name,
            cost_cents=0.0,
            diarized=config.diarize,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_backends/test_whisperx.py -v`
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/scriba/backends/whisperx.py tests/test_backends/test_whisperx.py
git commit -m "feat: add WhisperX + pyannote backend adapter"
```

---

### Task 10: OpenAI STT Backend Adapter

**Files:**
- Create: `src/scriba/backends/openai_stt.py`
- Create: `tests/test_backends/test_openai.py`

- [ ] **Step 1: Write failing tests**

`tests/test_backends/test_openai.py`:
```python
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from scriba.backends.openai_stt import OpenAISTTBackend
from scriba.contracts import TranscriptionConfig


def test_is_available():
    with patch("scriba.backends.openai_stt._import_openai", return_value=MagicMock()):
        assert OpenAISTTBackend(api_key="sk-test").is_available() is True


def test_is_unavailable_no_lib():
    with patch("scriba.backends.openai_stt._import_openai", return_value=None):
        assert OpenAISTTBackend(api_key="sk-test").is_available() is False


def test_is_unavailable_no_key():
    with patch("scriba.backends.openai_stt._import_openai", return_value=MagicMock()):
        assert OpenAISTTBackend(api_key=None).is_available() is False


@pytest.mark.asyncio
async def test_transcribe_verbose_json():
    mock_resp = MagicMock()
    mock_resp.model_dump.return_value = {
        "text": "hello world",
        "duration": 2.0,
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "hello"},
            {"start": 1.0, "end": 2.0, "text": "world"},
        ],
    }

    mock_client = AsyncMock()
    mock_client.audio.transcriptions.create.return_value = mock_resp

    with patch("scriba.backends.openai_stt._import_openai") as mock_openai:
        mock_openai.return_value = MagicMock()
        mock_openai.return_value.AsyncOpenAI.return_value = mock_client

        backend = OpenAISTTBackend(api_key="sk-test")
        backend._make_client = lambda: mock_client
        config = TranscriptionConfig(model="whisper-1")
        result = await backend.transcribe(Path("/fake/audio.wav"), config)

    assert result.text == "hello world"
    assert result.backend == "openai_stt"
    assert result.diarized is False
    assert len(result.segments) == 2


def test_estimate_cost():
    with patch("scriba.backends.openai_stt._import_openai", return_value=MagicMock()):
        backend = OpenAISTTBackend(api_key="sk-test")
        config = TranscriptionConfig(model="whisper-1")
        est = backend.estimate(600.0, config)
    assert est.cost_cents == pytest.approx(6.0)  # 10 min * 0.6 cents/min
    assert est.backend == "openai_stt"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_backends/test_openai.py -v`
Expected: FAIL

- [ ] **Step 3: Implement OpenAI adapter**

`src/scriba/backends/openai_stt.py`:
```python
"""OpenAI STT backend adapter."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from scriba.contracts import Estimate, Segment, TranscriptionConfig, TranscriptionResult
from scriba.errors import BackendError
from scriba.router.cost_model import estimate_cost_cents, estimate_time_seconds

DIARIZE_MODELS = {"gpt-4o-mini-transcribe", "gpt-4o-transcribe"}
MAX_UPLOAD_BYTES = 24 * 1024 * 1024


def _import_openai() -> Any | None:
    try:
        import openai
        return openai
    except ImportError:
        return None


class OpenAISTTBackend:
    name = "openai_stt"

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key

    def is_available(self) -> bool:
        return _import_openai() is not None and self._api_key is not None

    def _make_client(self) -> Any:
        openai = _import_openai()
        return openai.AsyncOpenAI(api_key=self._api_key, timeout=180)

    def estimate(self, duration_seconds: float, config: TranscriptionConfig) -> Estimate:
        return Estimate(
            backend=self.name,
            model=config.model,
            time_seconds=estimate_time_seconds(self.name, config.model, duration_seconds=duration_seconds),
            cost_cents=estimate_cost_cents(self.name, config.model, duration_minutes=duration_seconds / 60),
            available=self.is_available(),
            recommended=False,
        )

    async def transcribe(self, audio_path: Path, config: TranscriptionConfig) -> TranscriptionResult:
        openai_mod = _import_openai()
        if openai_mod is None:
            from scriba.errors import DependencyMissing
            raise DependencyMissing("openai", hint='pip install "nyqst-scriba[openai]"')

        # Compress if too large
        prepared_path = audio_path
        tmp_dir = None
        try:
            if audio_path.stat().st_size > MAX_UPLOAD_BYTES:
                from scriba.media.ingest import compress_for_upload
                tmp_dir = tempfile.mkdtemp(prefix="scriba-")
                prepared_path = compress_for_upload(
                    audio_path, Path(tmp_dir) / f"{audio_path.stem}.mp3"
                )

            use_diarize = config.diarize and config.model in DIARIZE_MODELS
            response_format = "diarized_json" if use_diarize else "verbose_json"

            client = self._make_client()
            kwargs: dict[str, Any] = {
                "model": config.model,
                "file": prepared_path.open("rb"),
                "response_format": response_format,
            }
            if config.language:
                kwargs["language"] = config.language
            if config.model != "whisper-1":
                kwargs["chunking_strategy"] = "auto"

            try:
                resp = await client.audio.transcriptions.create(**kwargs)
            except Exception as e:
                raise BackendError(self.name, cause=e, suggestion="check API key and model access")
            finally:
                kwargs["file"].close()

        finally:
            if tmp_dir:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

        raw = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)

        segments = [
            Segment(
                start=s.get("start", 0.0),
                end=s.get("end", 0.0),
                text=s.get("text", "").strip(),
                speaker=s.get("speaker") if use_diarize else None,
            )
            for s in raw.get("segments", [])
            if isinstance(s, dict) and s.get("text", "").strip()
        ]

        return TranscriptionResult(
            text=raw.get("text", "").strip(),
            segments=segments,
            duration_seconds=float(raw.get("duration", 0)),
            model_used=config.model,
            backend=self.name,
            cost_cents=estimate_cost_cents(
                self.name, config.model,
                duration_minutes=float(raw.get("duration", 0)) / 60,
            ),
            diarized=use_diarize,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_backends/test_openai.py -v`
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/scriba/backends/openai_stt.py tests/test_backends/test_openai.py
git commit -m "feat: add OpenAI STT backend adapter with compression and diarization"
```

---

### Task 11: Output Formatter

**Files:**
- Create: `src/scriba/output/__init__.py`
- Create: `src/scriba/output/formatter.py`
- Create: `tests/test_formatter.py`

- [ ] **Step 1: Write failing tests**

`tests/test_formatter.py`:
```python
import json

import pytest

from scriba.contracts import Segment, TranscriptionResult
from scriba.output.formatter import format_result


def _make_result(diarized: bool = False) -> TranscriptionResult:
    segments = [
        Segment(start=0.0, end=1.5, text="Hello there.", speaker="SPEAKER_00" if diarized else None),
        Segment(start=2.0, end=4.0, text="How are you?", speaker="SPEAKER_01" if diarized else None),
    ]
    return TranscriptionResult(
        text="Hello there. How are you?",
        segments=segments,
        duration_seconds=4.0,
        model_used="large-v3",
        backend="mlx_whisper",
        cost_cents=0.0,
        diarized=diarized,
    )


def test_format_json():
    result = _make_result()
    output = format_result(result, output_format="json", output_tier="timestamped")
    parsed = json.loads(output)
    assert parsed["text"] == "Hello there. How are you?"
    assert len(parsed["segments"]) == 2


def test_format_text_raw():
    result = _make_result()
    output = format_result(result, output_format="text", output_tier="raw")
    assert output == "Hello there. How are you?"


def test_format_text_timestamped():
    result = _make_result()
    output = format_result(result, output_format="text", output_tier="timestamped")
    assert "[0:00:00" in output
    assert "Hello there." in output


def test_format_text_diarized():
    result = _make_result(diarized=True)
    output = format_result(result, output_format="text", output_tier="diarized")
    assert "SPEAKER_00" in output
    assert "SPEAKER_01" in output


def test_format_srt():
    result = _make_result()
    output = format_result(result, output_format="srt", output_tier="timestamped")
    assert "1\n" in output
    assert "00:00:00,000 --> 00:00:01,500" in output
    assert "Hello there." in output


def test_format_vtt():
    result = _make_result()
    output = format_result(result, output_format="vtt", output_tier="timestamped")
    assert output.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.500" in output


def test_format_md_diarized():
    result = _make_result(diarized=True)
    output = format_result(result, output_format="md", output_tier="diarized")
    assert "**SPEAKER_00**" in output
    assert "**SPEAKER_01**" in output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_formatter.py -v`
Expected: FAIL

- [ ] **Step 3: Implement formatter**

`src/scriba/output/__init__.py`:
```python
from scriba.output.formatter import format_result

__all__ = ["format_result"]
```

`src/scriba/output/formatter.py`:
```python
"""Output formatting for transcription results."""
from __future__ import annotations

import json
from dataclasses import asdict

from scriba.contracts import Segment, TranscriptionResult


def format_result(result: TranscriptionResult, *, output_format: str, output_tier: str) -> str:
    """Render a TranscriptionResult into the requested format and tier."""
    segments = _filter_tier(result, output_tier)

    match output_format:
        case "json":
            return _to_json(result, segments, output_tier)
        case "text":
            return _to_text(result, segments, output_tier)
        case "srt":
            return _to_srt(segments)
        case "vtt":
            return _to_vtt(segments)
        case "md":
            return _to_md(result, segments, output_tier)
        case _:
            raise ValueError(f"Unknown format: {output_format}")


def _filter_tier(result: TranscriptionResult, tier: str) -> list[Segment]:
    if tier == "raw":
        return []
    return result.segments


def _to_json(result: TranscriptionResult, segments: list[Segment], tier: str) -> str:
    data: dict = {"text": result.text}
    if tier != "raw":
        data["segments"] = [asdict(s) for s in segments]
    data["duration_seconds"] = result.duration_seconds
    data["backend"] = result.backend
    data["model"] = result.model_used
    data["cost_cents"] = result.cost_cents
    data["diarized"] = result.diarized
    return json.dumps(data, indent=2, ensure_ascii=False)


def _to_text(result: TranscriptionResult, segments: list[Segment], tier: str) -> str:
    if tier == "raw":
        return result.text
    lines = []
    for s in segments:
        ts = _fmt_ts_text(s.start)
        if tier == "diarized" and s.speaker:
            lines.append(f"[{ts}] {s.speaker}: {s.text}")
        else:
            lines.append(f"[{ts}] {s.text}")
    return "\n".join(lines)


def _to_srt(segments: list[Segment]) -> str:
    lines = []
    for i, s in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_fmt_ts_srt(s.start)} --> {_fmt_ts_srt(s.end)}")
        if s.speaker:
            lines.append(f"[{s.speaker}] {s.text}")
        else:
            lines.append(s.text)
        lines.append("")
    return "\n".join(lines)


def _to_vtt(segments: list[Segment]) -> str:
    lines = ["WEBVTT", ""]
    for s in segments:
        lines.append(f"{_fmt_ts_vtt(s.start)} --> {_fmt_ts_vtt(s.end)}")
        if s.speaker:
            lines.append(f"[{s.speaker}] {s.text}")
        else:
            lines.append(s.text)
        lines.append("")
    return "\n".join(lines)


def _to_md(result: TranscriptionResult, segments: list[Segment], tier: str) -> str:
    lines = [f"# Transcript", ""]
    if tier == "raw":
        lines.append(result.text)
        return "\n".join(lines)
    current_speaker = None
    for s in segments:
        if tier == "diarized" and s.speaker and s.speaker != current_speaker:
            current_speaker = s.speaker
            lines.append(f"\n**{s.speaker}** ({_fmt_ts_text(s.start)})")
        ts = _fmt_ts_text(s.start)
        if tier == "diarized" and s.speaker:
            lines.append(f"> {s.text}")
        else:
            lines.append(f"[{ts}] {s.text}")
    return "\n".join(lines)


def _fmt_ts_text(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def _fmt_ts_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_ts_vtt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_formatter.py -v`
Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/scriba/output/ tests/test_formatter.py
git commit -m "feat: add output formatter (JSON, text, SRT, VTT, Markdown)"
```

---

### Task 12: Subtitle Burner

**Files:**
- Create: `src/scriba/media/subtitle.py`
- Create: `tests/test_subtitle.py`

- [ ] **Step 1: Write failing tests**

`tests/test_subtitle.py`:
```python
from pathlib import Path

import pytest

from scriba.contracts import Segment
from scriba.media.subtitle import generate_srt, generate_vtt, burn_subtitles

FIXTURES = Path(__file__).parent / "fixtures"


def test_generate_srt():
    segments = [
        Segment(start=0.0, end=1.5, text="Hello"),
        Segment(start=2.0, end=3.5, text="World"),
    ]
    srt = generate_srt(segments)
    assert "1\n" in srt
    assert "00:00:00,000 --> 00:00:01,500" in srt
    assert "Hello" in srt
    assert "2\n" in srt


def test_generate_srt_with_overlap():
    segments = [
        Segment(start=0.0, end=2.0, text="Hello", speaker="SPEAKER_00"),
        Segment(start=1.0, end=2.5, text="World", speaker="SPEAKER_01"),
    ]
    srt = generate_srt(segments)
    assert "SPEAKER_00" in srt
    assert "SPEAKER_01" in srt


def test_generate_vtt():
    segments = [Segment(start=0.0, end=1.5, text="Hello")]
    vtt = generate_vtt(segments)
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.500" in vtt


def test_burn_subtitles_soft(tmp_path: Path):
    video = FIXTURES / "test_video.mp4"
    if not video.exists():
        pytest.skip("test video fixture not available")
    srt_path = tmp_path / "subs.srt"
    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n\n")
    out = burn_subtitles(video, srt_path, tmp_path / "out.mp4", mode="soft")
    assert out.exists()
    assert out.stat().st_size > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_subtitle.py -v`
Expected: FAIL

- [ ] **Step 3: Implement subtitle.py**

`src/scriba/media/subtitle.py`:
```python
"""SRT/VTT generation and video subtitle burn-in."""
from __future__ import annotations

import subprocess
from pathlib import Path

from scriba.contracts import Segment
from scriba.errors import AudioError, DependencyMissing


def generate_srt(segments: list[Segment]) -> str:
    lines = []
    for i, s in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_fmt_srt(s.start)} --> {_fmt_srt(s.end)}")
        prefix = f"[{s.speaker}] " if s.speaker else ""
        lines.append(f"{prefix}{s.text}")
        lines.append("")
    return "\n".join(lines)


def generate_vtt(segments: list[Segment]) -> str:
    lines = ["WEBVTT", ""]
    for s in segments:
        lines.append(f"{_fmt_vtt(s.start)} --> {_fmt_vtt(s.end)}")
        prefix = f"[{s.speaker}] " if s.speaker else ""
        lines.append(f"{prefix}{s.text}")
        lines.append("")
    return "\n".join(lines)


def burn_subtitles(
    video_path: Path,
    srt_path: Path,
    output_path: Path,
    *,
    mode: str = "soft",
) -> Path:
    """Burn subtitles into video. mode='soft' embeds as stream, 'hard' overlays."""
    import shutil
    if not shutil.which("ffmpeg"):
        raise DependencyMissing("ffmpeg", hint="brew install ffmpeg")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = video_path.suffix.lower()

    if mode == "hard":
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", f"subtitles={srt_path}",
            "-c:a", "copy",
            str(output_path),
        ]
    else:
        # Soft subtitles: container-specific codec
        sub_codec = {
            ".mp4": "mov_text", ".mov": "mov_text",
            ".mkv": "srt", ".webm": "webvtt",
        }.get(ext, "mov_text")
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-i", str(srt_path),
            "-c:v", "copy", "-c:a", "copy", "-c:s", sub_codec,
            str(output_path),
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AudioError(f"Subtitle burn failed: {result.stderr.strip()}")
    return output_path


def _fmt_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_vtt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_subtitle.py -v`
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/scriba/media/subtitle.py tests/test_subtitle.py
git commit -m "feat: add subtitle generator and video burn-in"
```

---

### Task 13: MCP Server

**Files:**
- Create: `src/scriba/mcp/__init__.py`
- Create: `src/scriba/mcp/server.py`
- Create: `tests/test_mcp/__init__.py`
- Create: `tests/test_mcp/test_server.py`

- [ ] **Step 1: Write failing tests**

`tests/test_mcp/__init__.py`: empty file.

`tests/test_mcp/test_server.py`:
```python
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from scriba.mcp.server import create_server


def test_server_has_tools():
    server = create_server()
    tool_names = [t.name for t in server._tool_manager.list_tools()]
    assert "transcribe" in tool_names
    assert "estimate" in tool_names
    assert "backends" in tool_names


@pytest.mark.asyncio
async def test_backends_tool():
    server = create_server()
    # Call the backends tool handler directly
    from scriba.mcp.server import handle_backends
    result = await handle_backends()
    assert isinstance(result, list)
    # Each entry should have name and available keys
    for entry in result:
        assert "name" in entry
        assert "available" in entry
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_mcp/test_server.py -v`
Expected: FAIL

- [ ] **Step 3: Implement MCP server**

`src/scriba/mcp/__init__.py`:
```python
"""Scriba MCP server."""
```

`src/scriba/mcp/server.py`:
```python
"""FastMCP server exposing transcription tools."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from scriba.backends import discover_backends
from scriba.config import ScribaConfig, load_config
from scriba.contracts import TranscriptionConfig
from scriba.errors import ScribaError
from scriba.media.ingest import extract_audio, probe_media
from scriba.output.formatter import format_result
from scriba.router.constraints import Constraints
from scriba.router.engine import BackendInfo, route
from scriba.secrets import EnvProvider, KeychainProvider, SecretsChain

mcp = FastMCP("scriba")


def create_server() -> FastMCP:
    return mcp


async def _get_secrets() -> SecretsChain:
    return SecretsChain([KeychainProvider(), EnvProvider()])


def _backend_to_info(b: Any) -> BackendInfo:
    supports_diarize = b.name in ("whisperx", "openai_stt")
    diarize_models = set()
    if b.name == "openai_stt":
        diarize_models = {"gpt-4o-mini-transcribe", "gpt-4o-transcribe"}
    models = {
        "mlx_whisper": ["tiny", "base", "small", "medium", "large-v3"],
        "whisperx": ["tiny", "base", "small", "medium", "large-v3"],
        "openai_stt": ["whisper-1", "gpt-4o-mini-transcribe", "gpt-4o-transcribe"],
    }.get(b.name, [])
    return BackendInfo(
        name=b.name,
        available=b.is_available(),
        models=models,
        supports_diarize=supports_diarize,
        diarize_models=diarize_models,
    )


@mcp.tool()
async def transcribe(
    file_path: str,
    quality: str = "balanced",
    budget_cents: int | None = None,
    timeout_seconds: int | None = None,
    diarize: bool = False,
    speakers: int | None = None,
    language: str | None = None,
    output_tier: str = "timestamped",
    output_format: str = "json",
    subtitle_video: bool = False,
    intent: str | None = None,
) -> str:
    """Transcribe an audio or video file."""
    try:
        path = Path(file_path)
        config = load_config()
        secrets = await _get_secrets()

        # Probe media
        info = probe_media(path)

        # Extract audio to temp
        with tempfile.TemporaryDirectory(prefix="scriba-") as tmpdir:
            audio_path = extract_audio(path, Path(tmpdir) / "audio.wav")

            # Build constraints
            constraints = Constraints(
                quality=quality,
                budget_cents=budget_cents,
                timeout_seconds=timeout_seconds,
                diarize=diarize,
                speakers=speakers,
                output_tier=output_tier,
                language=language,
            )

            # Discover backends and set up API keys
            all_backends = discover_backends()
            for b in all_backends:
                if b.name == "openai_stt":
                    key = await secrets.get("openai-api-key")
                    if key:
                        b._api_key = key

            backend_infos = [_backend_to_info(b) for b in all_backends]

            # Route
            decision = route(constraints, duration_seconds=info.duration_seconds, backends=backend_infos)

            # Find the selected backend adapter
            adapter = next(b for b in all_backends if b.name == decision.selected.backend)
            tc = TranscriptionConfig(
                model=decision.selected.model,
                language=language,
                diarize=diarize,
                speakers=speakers,
                output_tier=output_tier,
            )

            # Transcribe
            result = await adapter.transcribe(audio_path, tc)
            result.routing = decision
            if output_tier == "enriched":
                result.enrichment_available = True

            # Subtitle burn
            if subtitle_video and info.has_video:
                from scriba.media.subtitle import generate_srt, burn_subtitles
                srt_content = generate_srt(result.segments)
                srt_path = Path(tmpdir) / "subs.srt"
                srt_path.write_text(srt_content)
                out_video = path.parent / f"{path.stem}.subtitled{path.suffix}"
                burn_subtitles(path, srt_path, out_video)

            return format_result(result, output_format=output_format, output_tier=output_tier)

    except ScribaError as e:
        return json.dumps({"error": str(e), "type": type(e).__name__})


@mcp.tool()
async def estimate(
    file_path: str,
    quality: str = "balanced",
    budget_cents: int | None = None,
    timeout_seconds: int | None = None,
    diarize: bool = False,
    speakers: int | None = None,
    output_tier: str = "timestamped",
) -> str:
    """Estimate cost and time for transcription without running it."""
    try:
        info = probe_media(Path(file_path))
        constraints = Constraints(
            quality=quality,
            budget_cents=budget_cents,
            timeout_seconds=timeout_seconds,
            diarize=diarize,
            speakers=speakers,
            output_tier=output_tier,
        )

        secrets = await _get_secrets()
        all_backends = discover_backends()
        for b in all_backends:
            if b.name == "openai_stt":
                key = await secrets.get("openai-api-key")
                if key:
                    b._api_key = key

        backend_infos = [_backend_to_info(b) for b in all_backends]
        decision = route(constraints, duration_seconds=info.duration_seconds, backends=backend_infos)

        result = {
            "selected": {
                "backend": decision.selected.backend,
                "model": decision.selected.model,
                "time_seconds": decision.selected.time_seconds,
                "cost_cents": decision.selected.cost_cents,
            },
            "alternatives": [
                {
                    "backend": a.backend,
                    "model": a.model,
                    "time_seconds": a.time_seconds,
                    "cost_cents": a.cost_cents,
                }
                for a in decision.alternatives
            ],
            "trade_offs": decision.trade_offs,
        }
        return json.dumps(result, indent=2)

    except ScribaError as e:
        return json.dumps({"error": str(e), "type": type(e).__name__})


async def handle_backends() -> list[dict]:
    """List available backends."""
    all_backends = discover_backends()
    return [
        {
            "name": b.name,
            "available": b.is_available(),
        }
        for b in all_backends
    ]


@mcp.tool()
async def backends() -> str:
    """List available transcription backends and their status."""
    result = await handle_backends()
    return json.dumps(result, indent=2)


def main() -> None:
    transport = os.environ.get("SCRIBA_MCP_TRANSPORT", "stdio")
    host = os.environ.get("SCRIBA_MCP_HOST", "127.0.0.1")
    port = int(os.environ.get("SCRIBA_MCP_PORT", "8787"))

    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "sse":
        mcp.run(transport="sse", host=host, port=port)
    else:
        mcp.run(transport="streamable-http", host=host, port=port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_mcp/test_server.py -v`
Expected: all 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/scriba/mcp/ tests/test_mcp/
git commit -m "feat: add MCP server with transcribe, estimate, backends tools"
```

---

### Task 14: CLI

**Files:**
- Create: `src/scriba/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

`tests/test_cli.py`:
```python
from pathlib import Path

from click.testing import CliRunner

from scriba.cli import main

FIXTURES = Path(__file__).parent / "fixtures"


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "transcribe" in result.output
    assert "estimate" in result.output
    assert "backends" in result.output
    assert "configure" in result.output


def test_cli_backends():
    runner = CliRunner()
    result = runner.invoke(main, ["backends"])
    assert result.exit_code == 0
    # Should list at least the backend names
    assert "mlx_whisper" in result.output or "whisperx" in result.output or "openai_stt" in result.output


def test_cli_transcribe_missing_file():
    runner = CliRunner()
    result = runner.invoke(main, ["transcribe", "/nonexistent/file.wav"])
    assert result.exit_code != 0


def test_cli_estimate_missing_file():
    runner = CliRunner()
    result = runner.invoke(main, ["estimate", "/nonexistent/file.wav"])
    assert result.exit_code != 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL

- [ ] **Step 3: Implement CLI**

`src/scriba/cli.py`:
```python
"""Click CLI for Scriba."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click

from scriba.backends import discover_backends
from scriba.config import load_config
from scriba.contracts import TranscriptionConfig
from scriba.errors import ScribaError
from scriba.media.ingest import extract_audio, probe_media
from scriba.output.formatter import format_result
from scriba.router.constraints import Constraints
from scriba.router.engine import BackendInfo, route
from scriba.secrets import EnvProvider, KeychainProvider, SecretsChain


@click.group()
def main():
    """Scriba: intent-aware transcription tool."""


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("-q", "--quality", type=click.Choice(["fast", "balanced", "high"]), default="balanced")
@click.option("-t", "--tier", type=click.Choice(["raw", "timestamped", "diarized", "enriched"]), default="timestamped")
@click.option("-f", "--format", "fmt", type=click.Choice(["json", "text", "srt", "vtt", "md"]), default="json")
@click.option("-d", "--diarize", is_flag=True, default=False)
@click.option("-s", "--speakers", type=int, default=None)
@click.option("-l", "--language", type=str, default=None)
@click.option("--budget", type=int, default=None, help="Max spend in cents")
@click.option("--timeout", type=int, default=None, help="Max wall-clock seconds")
@click.option("--subtitle", is_flag=True, default=False)
@click.option("--subtitle-mode", type=click.Choice(["soft", "hard"]), default="soft")
@click.option("-o", "--output", type=click.Path(), default=None)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--intent", type=str, default=None)
def transcribe(file, quality, tier, fmt, diarize, speakers, language, budget, timeout, subtitle, subtitle_mode, output, dry_run, intent):
    """Transcribe an audio or video file."""
    try:
        result = asyncio.run(_transcribe(
            Path(file), quality, tier, fmt, diarize, speakers, language,
            budget, timeout, subtitle, subtitle_mode, output, dry_run,
        ))
        if output:
            Path(output).write_text(result)
            click.echo(f"Written to {output}", err=True)
        else:
            click.echo(result)
    except ScribaError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


async def _transcribe(
    path: Path, quality: str, tier: str, fmt: str, diarize: bool,
    speakers: int | None, language: str | None, budget: int | None,
    timeout: int | None, subtitle: bool, subtitle_mode: str,
    output: str | None, dry_run: bool,
) -> str:
    import tempfile

    info = probe_media(path)
    constraints = Constraints(
        quality=quality, budget_cents=budget, timeout_seconds=timeout,
        diarize=diarize, speakers=speakers, output_tier=tier, language=language,
    )

    secrets = SecretsChain([KeychainProvider(), EnvProvider()])
    all_backends = discover_backends()
    for b in all_backends:
        if b.name == "openai_stt":
            key = await secrets.get("openai-api-key")
            if key:
                b._api_key = key

    backend_infos = _to_backend_infos(all_backends)
    decision = route(constraints, duration_seconds=info.duration_seconds, backends=backend_infos)

    # Print trade-offs to stderr
    if decision.trade_offs:
        for t in decision.trade_offs:
            click.echo(f"Note: {t}", err=True)

    if dry_run:
        return json.dumps({
            "selected": {"backend": decision.selected.backend, "model": decision.selected.model,
                         "time_seconds": decision.selected.time_seconds, "cost_cents": decision.selected.cost_cents},
            "alternatives": [{"backend": a.backend, "model": a.model} for a in decision.alternatives],
            "trade_offs": decision.trade_offs,
        }, indent=2)

    adapter = next(b for b in all_backends if b.name == decision.selected.backend)
    tc = TranscriptionConfig(
        model=decision.selected.model, language=language,
        diarize=diarize, speakers=speakers, output_tier=tier,
    )

    with tempfile.TemporaryDirectory(prefix="scriba-") as tmpdir:
        audio_path = extract_audio(path, Path(tmpdir) / "audio.wav")
        result = await adapter.transcribe(audio_path, tc)
        result.routing = decision

        if subtitle and info.has_video:
            from scriba.media.subtitle import generate_srt, burn_subtitles
            srt_content = generate_srt(result.segments)
            srt_path = Path(tmpdir) / "subs.srt"
            srt_path.write_text(srt_content)
            out_video = path.parent / f"{path.stem}.subtitled{path.suffix}"
            burn_subtitles(path, srt_path, out_video, mode=subtitle_mode)
            click.echo(f"Subtitled video: {out_video}", err=True)

    return format_result(result, output_format=fmt, output_tier=tier)


def _to_backend_infos(backends: list) -> list[BackendInfo]:
    infos = []
    for b in backends:
        supports_diarize = b.name in ("whisperx", "openai_stt")
        diarize_models = {"gpt-4o-mini-transcribe", "gpt-4o-transcribe"} if b.name == "openai_stt" else set()
        models = {
            "mlx_whisper": ["tiny", "base", "small", "medium", "large-v3"],
            "whisperx": ["tiny", "base", "small", "medium", "large-v3"],
            "openai_stt": ["whisper-1", "gpt-4o-mini-transcribe", "gpt-4o-transcribe"],
        }.get(b.name, [])
        infos.append(BackendInfo(name=b.name, available=b.is_available(), models=models,
                                 supports_diarize=supports_diarize, diarize_models=diarize_models))
    return infos


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("-q", "--quality", type=click.Choice(["fast", "balanced", "high"]), default="balanced")
@click.option("-d", "--diarize", is_flag=True, default=False)
@click.option("--budget", type=int, default=None)
@click.option("--timeout", type=int, default=None)
def estimate(file, quality, diarize, budget, timeout):
    """Preview cost and time estimates without transcribing."""
    try:
        result = asyncio.run(_estimate(Path(file), quality, diarize, budget, timeout))
        click.echo(result)
    except ScribaError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


async def _estimate(path: Path, quality: str, diarize: bool, budget: int | None, timeout: int | None) -> str:
    info = probe_media(path)
    constraints = Constraints(quality=quality, diarize=diarize, budget_cents=budget, timeout_seconds=timeout)

    secrets = SecretsChain([KeychainProvider(), EnvProvider()])
    all_backends = discover_backends()
    for b in all_backends:
        if b.name == "openai_stt":
            key = await secrets.get("openai-api-key")
            if key:
                b._api_key = key

    backend_infos = _to_backend_infos(all_backends)
    decision = route(constraints, duration_seconds=info.duration_seconds, backends=backend_infos)
    return json.dumps({
        "selected": {"backend": decision.selected.backend, "model": decision.selected.model,
                     "time_seconds": round(decision.selected.time_seconds, 1),
                     "cost_cents": round(decision.selected.cost_cents, 2)},
        "alternatives": [{"backend": a.backend, "model": a.model,
                          "time_seconds": round(a.time_seconds, 1), "cost_cents": round(a.cost_cents, 2)}
                         for a in decision.alternatives],
        "trade_offs": decision.trade_offs,
    }, indent=2)


@main.command()
def backends():
    """List available transcription backends."""
    all_backends = discover_backends()
    for b in all_backends:
        status = "available" if b.is_available() else "not installed"
        click.echo(f"  {b.name}: {status}")


@main.command()
def configure():
    """Interactive setup for API keys and backends."""
    click.echo("Scriba Configuration")
    click.echo("=" * 40)

    # OpenAI
    key = click.prompt("OpenAI API key (enter to skip)", default="", show_default=False)
    if key:
        import keyring
        keyring.set_password("scriba", "openai-api-key", key)
        click.echo("  Stored in keychain.")

    # HuggingFace
    hf = click.prompt("HuggingFace token (enter to skip)", default="", show_default=False)
    if hf:
        import keyring
        keyring.set_password("scriba", "hf-token", hf)
        click.echo("  Stored in keychain.")

    # Check backends
    click.echo("\nBackend status:")
    all_backends = discover_backends()
    for b in all_backends:
        status = "available" if b.is_available() else "not installed"
        click.echo(f"  {b.name}: {status}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cli.py -v`
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/scriba/cli.py tests/test_cli.py
git commit -m "feat: add Click CLI with transcribe, estimate, backends, configure commands"
```

---

### Task 15: Claude Code Plugin Packaging

**Files:**
- Create: `plugin/plugin.json`
- Create: `plugin/.mcp.json`
- Create: `plugin/skills/transcribe.md`

- [ ] **Step 1: Create plugin manifest**

`plugin/plugin.json`:
```json
{
  "name": "scriba",
  "version": "0.1.0",
  "description": "Intent-aware transcription tool with local and cloud backends",
  "author": "NYQST Group"
}
```

- [ ] **Step 2: Create MCP config**

`plugin/.mcp.json`:
```json
{
  "mcpServers": {
    "scriba": {
      "command": "uv",
      "args": ["run", "--directory", "${CLAUDE_PLUGIN_ROOT}/..", "scriba-mcp"],
      "env": {}
    }
  }
}
```

- [ ] **Step 3: Create transcription skill**

`plugin/skills/transcribe.md`:
```markdown
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
```

- [ ] **Step 4: Commit**

```bash
git add plugin/
git commit -m "feat: add Claude Code plugin manifest, MCP config, and transcription skill"
```

---

### Task 16: GitHub Repository and CI

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `README.md`

- [ ] **Step 1: Create CI workflow**

`.github/workflows/ci.yml`:
```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - name: Install dependencies
        run: uv pip install -e ".[dev]" --system
      - name: Lint
        run: ruff check src/ tests/
      - name: Type check
        run: pyright src/
      - name: Unit tests
        run: pytest tests/ -v --ignore=tests/fixtures -k "not integration"

  test-macos:
    runs-on: macos-latest
    if: github.event_name == 'push'
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - name: Install with MLX
        run: uv pip install -e ".[mlx,dev]" --system
      - name: Integration tests (MLX)
        run: pytest tests/test_backends/test_mlx.py -v -k "integration" || true
```

- [ ] **Step 2: Create README**

`README.md`:
```markdown
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
```

- [ ] **Step 3: Create GitHub repo and push**

```bash
gh repo create NYQST-Group/scriba --private --description "Intent-aware transcription tool" --source . --push
```

Confirm with user before running — this creates a remote repo and pushes.

- [ ] **Step 4: Verify CI passes**

Run: `gh run watch` (after push triggers CI)

- [ ] **Step 5: Commit any CI fixes**

If CI fails, fix issues and commit.

---

### Task 17: End-to-End Integration Test

**Files:**
- Update: `tests/conftest.py` (add shared fixtures)

- [ ] **Step 1: Add shared fixtures to conftest**

Update `tests/conftest.py`:
```python
"""Shared test fixtures for Scriba."""
from pathlib import Path

import pytest

from scriba.contracts import Segment, TranscriptionResult

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_segments():
    return [
        Segment(start=0.0, end=1.5, text="Hello there.", speaker="SPEAKER_00"),
        Segment(start=2.0, end=4.0, text="How are you?", speaker="SPEAKER_01"),
    ]


@pytest.fixture
def sample_result(sample_segments):
    return TranscriptionResult(
        text="Hello there. How are you?",
        segments=sample_segments,
        duration_seconds=4.0,
        model_used="large-v3",
        backend="mlx_whisper",
        cost_cents=0.0,
        diarized=True,
    )


@pytest.fixture
def audio_fixture():
    path = FIXTURES / "test_tone.wav"
    if not path.exists():
        pytest.skip("test audio fixture not available")
    return path


@pytest.fixture
def video_fixture():
    path = FIXTURES / "test_video.mp4"
    if not path.exists():
        pytest.skip("test video fixture not available")
    return path
```

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: all tests PASS

- [ ] **Step 3: Final commit**

```bash
git add tests/conftest.py
git commit -m "feat: add shared test fixtures and verify full test suite"
```
