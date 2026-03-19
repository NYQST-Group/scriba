"""Backend adapter protocol."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from scriba.contracts import Estimate, TranscriptionConfig, TranscriptionResult


@runtime_checkable
class BackendAdapter(Protocol):
    name: str
    models: list[str]
    supports_diarize: bool
    diarize_models: set[str]

    async def transcribe(self, audio_path: Path, config: TranscriptionConfig) -> TranscriptionResult: ...
    def estimate(self, duration_seconds: float, config: TranscriptionConfig) -> Estimate: ...
    def is_available(self) -> bool: ...
