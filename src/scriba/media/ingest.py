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
        duration_seconds=duration, has_audio=has_audio, has_video=has_video,
        file_size_bytes=path.stat().st_size, sample_rate=sample_rate,
        channels=channels, codec=codec,
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
