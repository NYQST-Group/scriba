"""Pricing tables and time estimation for transcription backends."""
from __future__ import annotations

import json
import time
from pathlib import Path

# Cost per minute in cents
PRICING: dict[str, float] = {
    "mlx_whisper:tiny": 0.0, "mlx_whisper:base": 0.0, "mlx_whisper:small": 0.0,
    "mlx_whisper:medium": 0.0, "mlx_whisper:large-v3": 0.0,
    "whisperx:tiny": 0.0, "whisperx:base": 0.0, "whisperx:small": 0.0,
    "whisperx:medium": 0.0, "whisperx:large-v3": 0.0,
    "openai_stt:whisper-1": 0.6,
    "openai_stt:gpt-4o-mini-transcribe": 0.3,
    "openai_stt:gpt-4o-transcribe": 0.6,
}

# Time as fraction of audio realtime
TIME_MULTIPLIERS: dict[str, float] = {
    "mlx_whisper:tiny": 0.05, "mlx_whisper:base": 0.08, "mlx_whisper:small": 0.12,
    "mlx_whisper:medium": 0.20, "mlx_whisper:large-v3": 0.30,
    "whisperx:tiny": 0.15, "whisperx:base": 0.20, "whisperx:small": 0.25,
    "whisperx:medium": 0.35, "whisperx:large-v3": 0.50,
    "openai_stt:whisper-1": 0.15,
    "openai_stt:gpt-4o-mini-transcribe": 0.15,
    "openai_stt:gpt-4o-transcribe": 0.15,
}


def estimate_cost_cents(backend: str, model: str, *, duration_minutes: float) -> float:
    key = f"{backend}:{model}"
    rate = PRICING.get(key, 0.0)
    return rate * duration_minutes


def estimate_time_seconds(
    backend: str, model: str, *, duration_seconds: float,
    calibration_path: Path | None = None,
) -> float:
    key = f"{backend}:{model}"
    if calibration_path is not None:
        data = load_calibration(calibration_path)
        entries = [e for e in data.get(key, []) if e.get("audio_duration", 0) > 0]
        if entries:
            avg_ratio = sum(e["wall_clock"] / e["audio_duration"] for e in entries) / len(entries)
            return avg_ratio * duration_seconds
    multiplier = TIME_MULTIPLIERS.get(key, 1.0)
    return multiplier * duration_seconds


def load_calibration(path: Path) -> dict[str, list[dict]]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_calibration_entry(
    path: Path, backend: str, model: str, *,
    audio_duration: float, wall_clock: float,
    max_samples: int = 10, stale_days: int = 30,
) -> None:
    data = load_calibration(path)
    key = f"{backend}:{model}"
    entries = data.get(key, [])
    cutoff = time.time() - (stale_days * 86400)
    entries = [e for e in entries if e.get("ts", 0) > cutoff]
    entries.append({"audio_duration": audio_duration, "wall_clock": wall_clock, "ts": time.time()})
    entries = entries[-max_samples:]
    data[key] = entries
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
