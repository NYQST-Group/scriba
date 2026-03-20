"""Backend adapter registry."""
from __future__ import annotations

from scriba.backends.base import BackendAdapter
from scriba.router.engine import BackendInfo


def backend_to_info(adapter: BackendAdapter) -> BackendInfo:
    """Convert a backend adapter to a BackendInfo for the router."""
    return BackendInfo(
        name=adapter.name,
        available=adapter.is_available(),
        models=adapter.models,
        supports_diarize=adapter.supports_diarize,
        diarize_models=adapter.diarize_models,
    )


def discover_backends() -> list[BackendAdapter]:
    """Return all backend adapters, regardless of availability."""
    backends: list[BackendAdapter] = []

    try:
        from scriba.backends.mlx_whisper import MlxWhisperBackend
        backends.append(MlxWhisperBackend())
    except ImportError:
        pass

    try:
        from scriba.backends.whisperx import WhisperXBackend
        backends.append(WhisperXBackend())
    except ImportError:
        pass

    try:
        from scriba.backends.openai_stt import OpenAISTTBackend
        backends.append(OpenAISTTBackend())
    except ImportError:
        pass

    return backends


__all__ = ["BackendAdapter", "backend_to_info", "discover_backends"]
