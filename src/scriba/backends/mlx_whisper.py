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
    models = ["tiny", "base", "small", "medium", "large-v3"]
    supports_diarize = False
    diarize_models: set[str] = set()

    def is_available(self) -> bool:
        return _import_mlx_whisper() is not None

    def estimate(self, duration_seconds: float, config: TranscriptionConfig) -> Estimate:
        return Estimate(
            backend=self.name, model=config.model,
            time_seconds=estimate_time_seconds(self.name, config.model, duration_seconds=duration_seconds),
            cost_cents=estimate_cost_cents(self.name, config.model, duration_minutes=duration_seconds / 60),
            available=self.is_available(), recommended=False,
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
            Segment(start=s.get("start", 0.0), end=s.get("end", 0.0), text=s.get("text", "").strip())
            for s in raw.get("segments", [])
            if s.get("text", "").strip()
        ]
        return TranscriptionResult(
            text=raw.get("text", "").strip(), segments=segments,
            duration_seconds=sum(s.end - s.start for s in segments) if segments else 0.0,
            model_used=config.model, backend=self.name, cost_cents=0.0, diarized=False,
        )
