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
    merged = _merge_overlapping(segments)
    lines = []
    for i, s in enumerate(merged, 1):
        lines.append(str(i))
        lines.append(f"{fmt_srt_ts(s.start)} --> {fmt_srt_ts(s.end)}")
        prefix = f"[{s.speaker}] " if s.speaker else ""
        lines.append(f"{prefix}{s.text}")
        lines.append("")
    return "\n".join(lines)


def generate_vtt(segments: list[Segment]) -> str:
    merged = _merge_overlapping(segments)
    lines = ["WEBVTT", ""]
    for s in merged:
        lines.append(f"{fmt_vtt_ts(s.start)} --> {fmt_vtt_ts(s.end)}")
        prefix = f"[{s.speaker}] " if s.speaker else ""
        lines.append(f"{prefix}{s.text}")
        lines.append("")
    return "\n".join(lines)


def _merge_overlapping(segments: list[Segment]) -> list[Segment]:
    """Merge overlapping segments from different speakers into combined entries."""
    if len(segments) < 2:
        return segments
    result = []
    i = 0
    while i < len(segments):
        current = segments[i]
        # Check if next segment overlaps with current
        if (i + 1 < len(segments)
            and segments[i + 1].start < current.end
            and segments[i + 1].speaker != current.speaker
            and current.speaker and segments[i + 1].speaker):
            nxt = segments[i + 1]
            merged = Segment(
                start=min(current.start, nxt.start),
                end=max(current.end, nxt.end),
                text=f"{current.text} / {nxt.text}",
                speaker=f"{current.speaker} & {nxt.speaker}",
            )
            result.append(merged)
            i += 2
        else:
            result.append(current)
            i += 1
    return result
