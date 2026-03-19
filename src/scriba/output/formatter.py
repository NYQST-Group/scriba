"""Output formatting for transcription results."""
from __future__ import annotations
import json
from dataclasses import asdict
from scriba.contracts import Segment, TranscriptionResult
from scriba.formatting import fmt_text_ts, generate_srt, generate_vtt


def format_result(result: TranscriptionResult, *, output_format: str, output_tier: str) -> str:
    segments = result.segments if output_tier != "raw" else []
    match output_format:
        case "json":
            return _to_json(result, segments, output_tier)
        case "text":
            return _to_text(result, segments, output_tier)
        case "srt":
            return generate_srt(segments)
        case "vtt":
            return generate_vtt(segments)
        case "md":
            return _to_md(result, segments, output_tier)
        case _:
            raise ValueError(f"Unknown format: {output_format}")


def _to_json(result: TranscriptionResult, segments: list[Segment], tier: str) -> str:
    data: dict = {"text": result.text}
    if tier != "raw":
        data["segments"] = [asdict(s) for s in segments]
    data["duration_seconds"] = result.duration_seconds
    data["backend"] = result.backend
    data["model"] = result.model_used
    data["cost_cents"] = result.cost_cents
    data["diarized"] = result.diarized
    if result.enrichment_available:
        data["enrichment_available"] = True
    return json.dumps(data, indent=2, ensure_ascii=False)


def _to_text(result: TranscriptionResult, segments: list[Segment], tier: str) -> str:
    if tier == "raw":
        return result.text
    lines = []
    for s in segments:
        ts = fmt_text_ts(s.start)
        if tier == "diarized" and s.speaker:
            lines.append(f"[{ts}] {s.speaker}: {s.text}")
        else:
            lines.append(f"[{ts}] {s.text}")
    return "\n".join(lines)


def _to_md(result: TranscriptionResult, segments: list[Segment], tier: str) -> str:
    lines = ["# Transcript", ""]
    if tier == "raw":
        lines.append(result.text)
        return "\n".join(lines)
    current_speaker = None
    for s in segments:
        if tier == "diarized" and s.speaker and s.speaker != current_speaker:
            current_speaker = s.speaker
            lines.append(f"\n**{s.speaker}** ({fmt_text_ts(s.start)})")
        ts = fmt_text_ts(s.start)
        if tier == "diarized" and s.speaker:
            lines.append(f"> {s.text}")
        else:
            lines.append(f"[{ts}] {s.text}")
    return "\n".join(lines)
