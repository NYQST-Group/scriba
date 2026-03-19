"""Video subtitle burn-in via ffmpeg."""
from __future__ import annotations
import subprocess
from pathlib import Path
from scriba.errors import AudioError, DependencyMissing


def burn_subtitles(
    video_path: Path, srt_path: Path, output_path: Path, *, mode: str = "soft",
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
            "-vf", f"subtitles={srt_path}", "-c:a", "copy", str(output_path),
        ]
    else:
        sub_codec = {
            ".mp4": "mov_text", ".mov": "mov_text",
            ".mkv": "srt", ".webm": "webvtt",
        }.get(ext, "mov_text")
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path), "-i", str(srt_path),
            "-c:v", "copy", "-c:a", "copy", "-c:s", sub_codec, str(output_path),
        ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AudioError(f"Subtitle burn failed: {result.stderr.strip()}")
    return output_path
