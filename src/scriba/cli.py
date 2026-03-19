"""Click CLI for Scriba."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click

from scriba.backends import backend_to_info, discover_backends
from scriba.config import load_config
from scriba.contracts import TranscriptionConfig
from scriba.errors import ScribaError
from scriba.media.ingest import extract_audio, probe_media
from scriba.output.formatter import format_result
from scriba.router.constraints import Constraints
from scriba.router.engine import route
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
@click.option("--enrich", is_flag=True, default=False, help="Summarize via OpenAI (enriched tier)")
def transcribe(file, quality, tier, fmt, diarize, speakers, language, budget, timeout,
               subtitle, subtitle_mode, output, dry_run, intent, enrich):
    """Transcribe an audio or video file."""
    try:
        text = asyncio.run(_transcribe(
            Path(file), quality, tier, fmt, diarize, speakers, language,
            budget, timeout, subtitle, subtitle_mode, output, dry_run, enrich,
        ))
        if output:
            Path(output).write_text(text)
            click.echo(f"Written to {output}", err=True)
        else:
            click.echo(text)
    except ScribaError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


async def _transcribe(
    path: Path, quality: str, tier: str, fmt: str, diarize: bool,
    speakers: int | None, language: str | None, budget: int | None,
    timeout: int | None, subtitle: bool, subtitle_mode: str,
    output: str | None, dry_run: bool, enrich: bool = False,
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

    backend_infos = [backend_to_info(b) for b in all_backends]
    decision = route(constraints, duration_seconds=info.duration_seconds, backends=backend_infos)

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

    import time as _time

    with tempfile.TemporaryDirectory(prefix="scriba-") as tmpdir:
        audio_path = extract_audio(path, Path(tmpdir) / "audio.wav")
        _start = _time.monotonic()
        result = await adapter.transcribe(audio_path, tc)
        _elapsed = _time.monotonic() - _start
        result.routing = decision

        from scriba.router.cost_model import save_calibration_entry
        cal_path = Path(load_config().calibration_path).expanduser()
        save_calibration_entry(cal_path, adapter.name, tc.model, audio_duration=info.duration_seconds, wall_clock=_elapsed)

        if subtitle and info.has_video and result.segments:
            from scriba.formatting import generate_srt
            from scriba.media.subtitle import burn_subtitles
            srt_content = generate_srt(result.segments)
            srt_path = Path(tmpdir) / "subs.srt"
            srt_path.write_text(srt_content)
            out_video = path.parent / f"{path.stem}.subtitled{path.suffix}"
            burn_subtitles(path, srt_path, out_video, mode=subtitle_mode)
            click.echo(f"Subtitled video: {out_video}", err=True)

    if enrich:
        secrets = SecretsChain([KeychainProvider(), EnvProvider()])
        openai_key = await secrets.get("openai-api-key")
        if openai_key:
            try:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=openai_key)
                utterances_text = "\n".join(
                    f"[{s.start:.1f}-{s.end:.1f}] {s.speaker or 'SPEAKER'}: {s.text}"
                    for s in result.segments
                )
                resp = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": f"Summarize this transcript. Provide: title, summary, key decisions, action items, and notable quotes.\n\n{utterances_text}"
                    }],
                    timeout=60,
                )
                enrichment = resp.choices[0].message.content
                click.echo("\n--- Enrichment ---\n", err=True)
                click.echo(enrichment, err=True)
            except Exception as e:
                click.echo(f"Enrichment failed: {e}", err=True)
        else:
            click.echo("Warning: --enrich requires OpenAI API key. Skipping enrichment.", err=True)

    return format_result(result, output_format=fmt, output_tier=tier)


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("-q", "--quality", type=click.Choice(["fast", "balanced", "high"]), default="balanced")
@click.option("-d", "--diarize", is_flag=True, default=False)
@click.option("--budget", type=int, default=None)
@click.option("--timeout", type=int, default=None)
def estimate(file, quality, diarize, budget, timeout):
    """Preview cost and time estimates without transcribing."""
    try:
        text = asyncio.run(_estimate(Path(file), quality, diarize, budget, timeout))
        click.echo(text)
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

    backend_infos = [backend_to_info(b) for b in all_backends]
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
    for b in discover_backends():
        status = "available" if b.is_available() else "not installed"
        click.echo(f"  {b.name}: {status}")


@main.command()
def configure():
    """Interactive setup for API keys and backends."""
    click.echo("Scriba Configuration")
    click.echo("=" * 40)
    key = click.prompt("OpenAI API key (enter to skip)", default="", show_default=False)
    if key:
        import keyring
        keyring.set_password("scriba", "openai-api-key", key)
        click.echo("  Stored in keychain.")
        # Validate
        try:
            import openai
            client = openai.OpenAI(api_key=key, timeout=10)
            client.models.list()
            click.echo("  Validated successfully.")
        except Exception as e:
            click.echo(f"  Warning: validation failed ({e}). Key stored anyway.", err=True)
    hf = click.prompt("HuggingFace token (enter to skip)", default="", show_default=False)
    if hf:
        import keyring
        keyring.set_password("scriba", "hf-token", hf)
        click.echo("  Stored in keychain.")
    click.echo("\nBackend status:")
    for b in discover_backends():
        status = "available" if b.is_available() else "not installed"
        click.echo(f"  {b.name}: {status}")
