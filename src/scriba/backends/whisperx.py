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
    models = ["tiny", "base", "small", "medium", "large-v3"]
    supports_diarize = True
    diarize_models: set[str] = set()

    def is_available(self) -> bool:
        return _import_whisperx() is not None

    def estimate(self, duration_seconds: float, config: TranscriptionConfig) -> Estimate:
        return Estimate(
            backend=self.name, model=config.model,
            time_seconds=estimate_time_seconds(self.name, config.model, duration_seconds=duration_seconds),
            cost_cents=estimate_cost_cents(self.name, config.model, duration_minutes=duration_seconds / 60),
            available=self.is_available(), recommended=False,
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

        align_model, align_metadata = wx.load_align_model(
            language_code=config.language or result.get("language", "en"), device=device,
        )
        result = wx.align(result["segments"], align_model, align_metadata, audio, device)

        if config.diarize:
            diarize_model = wx.DiarizationPipeline(device=device)
            diarize_kwargs = {}
            if config.speakers:
                diarize_kwargs["num_speakers"] = config.speakers
            diarize_segments = diarize_model(audio, **diarize_kwargs)
            result = wx.assign_word_speakers(diarize_segments, result)

        segments = [
            Segment(
                start=s.get("start", 0.0), end=s.get("end", 0.0),
                text=s.get("text", "").strip(),
                speaker=s.get("speaker") if config.diarize else None,
            )
            for s in result.get("segments", [])
            if s.get("text", "").strip()
        ]
        full_text = " ".join(s.text for s in segments)
        return TranscriptionResult(
            text=full_text, segments=segments,
            duration_seconds=segments[-1].end if segments else 0.0,
            model_used=config.model, backend=self.name, cost_cents=0.0, diarized=config.diarize,
        )
