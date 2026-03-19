"""Shared formatting utilities for SRT, VTT, and timestamp rendering."""
from __future__ import annotations
from scriba.contracts import Segment


def fmt_srt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fmt_vtt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def fmt_text_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def generate_srt(segments: list[Segment]) -> str:
    lines = []
    for i, s in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{fmt_srt_ts(s.start)} --> {fmt_srt_ts(s.end)}")
        prefix = f"[{s.speaker}] " if s.speaker else ""
        lines.append(f"{prefix}{s.text}")
        lines.append("")
    return "\n".join(lines)


def generate_vtt(segments: list[Segment]) -> str:
    lines = ["WEBVTT", ""]
    for s in segments:
        lines.append(f"{fmt_vtt_ts(s.start)} --> {fmt_vtt_ts(s.end)}")
        prefix = f"[{s.speaker}] " if s.speaker else ""
        lines.append(f"{prefix}{s.text}")
        lines.append("")
    return "\n".join(lines)
