"""FastMCP server exposing transcription tools."""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time as _time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from scriba.backends import backend_to_info, discover_backends
from scriba.contracts import TranscriptionConfig
from scriba.errors import ScribaError
from scriba.media.ingest import extract_audio, probe_media
from scriba.output.formatter import format_result
from scriba.router.constraints import Constraints
from scriba.router.engine import route
from scriba.secrets import EnvProvider, KeychainProvider, SecretsChain

mcp = FastMCP("scriba")

_local_sem = asyncio.Semaphore(1)
_cloud_sem = asyncio.Semaphore(3)


def create_server() -> FastMCP:
    return mcp


async def _get_secrets() -> SecretsChain:
    return SecretsChain([KeychainProvider(), EnvProvider()])


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
        secrets = await _get_secrets()
        info = probe_media(path)

        with tempfile.TemporaryDirectory(prefix="scriba-") as tmpdir:
            audio_path = extract_audio(path, Path(tmpdir) / "audio.wav")
            constraints = Constraints(
                quality=quality, budget_cents=budget_cents, timeout_seconds=timeout_seconds,
                diarize=diarize, speakers=speakers, output_tier=output_tier, language=language,
            )

            all_backends = discover_backends()
            for b in all_backends:
                if b.name == "openai_stt":
                    key = await secrets.get("openai-api-key")
                    if key:
                        b._api_key = key

            backend_infos = [backend_to_info(b) for b in all_backends]
            decision = route(constraints, duration_seconds=info.duration_seconds, backends=backend_infos)
            adapter = next(b for b in all_backends if b.name == decision.selected.backend)

            tc = TranscriptionConfig(
                model=decision.selected.model, language=language,
                diarize=diarize, speakers=speakers, output_tier=output_tier,
            )
            sem = _cloud_sem if adapter.name == "openai_stt" else _local_sem
            _start = _time.monotonic()
            async with sem:
                result = await adapter.transcribe(audio_path, tc)
            _elapsed = _time.monotonic() - _start
            from scriba.router.cost_model import save_calibration_entry
            from scriba.config import load_config as _load_config
            _cfg = _load_config()
            cal_path = Path(_cfg.calibration_path).expanduser()
            save_calibration_entry(cal_path, adapter.name, tc.model, audio_duration=info.duration_seconds, wall_clock=_elapsed)
            result.routing = decision
            if output_tier == "enriched":
                result.enrichment_available = True

            if subtitle_video and info.has_video and result.segments:
                from scriba.formatting import generate_srt
                from scriba.media.subtitle import burn_subtitles
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
            quality=quality, budget_cents=budget_cents, timeout_seconds=timeout_seconds,
            diarize=diarize, speakers=speakers, output_tier=output_tier,
        )
        secrets = await _get_secrets()
        all_backends = discover_backends()
        for b in all_backends:
            if b.name == "openai_stt":
                key = await secrets.get("openai-api-key")
                if key:
                    b._api_key = key

        backend_infos = [backend_to_info(b) for b in all_backends]
        decision = route(constraints, duration_seconds=info.duration_seconds, backends=backend_infos)
        return json.dumps({
            "selected": {
                "backend": decision.selected.backend, "model": decision.selected.model,
                "time_seconds": decision.selected.time_seconds, "cost_cents": decision.selected.cost_cents,
            },
            "alternatives": [
                {"backend": a.backend, "model": a.model, "time_seconds": a.time_seconds, "cost_cents": a.cost_cents}
                for a in decision.alternatives
            ],
            "trade_offs": decision.trade_offs,
            "routing_notes": decision.trade_offs if decision.trade_offs else None,
        }, indent=2)
    except ScribaError as e:
        return json.dumps({"error": str(e), "type": type(e).__name__})


async def handle_backends() -> list[dict]:
    """List available backends."""
    return [
        {
            "name": b.name,
            "available": b.is_available(),
            "models": b.models,
            "supports_diarize": b.supports_diarize,
            "reason_unavailable": None if b.is_available() else f"{b.name} package not installed",
        }
        for b in discover_backends()
    ]


@mcp.tool()
async def backends() -> str:
    """List available transcription backends and their status."""
    return json.dumps(await handle_backends(), indent=2)


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
