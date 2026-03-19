from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from scriba.backends.mlx_whisper import MlxWhisperBackend
from scriba.contracts import TranscriptionConfig


def test_is_available_when_installed():
    with patch("scriba.backends.mlx_whisper._import_mlx_whisper", return_value=MagicMock()):
        assert MlxWhisperBackend().is_available() is True


def test_is_unavailable_when_not_installed():
    with patch("scriba.backends.mlx_whisper._import_mlx_whisper", return_value=None):
        assert MlxWhisperBackend().is_available() is False


def test_backend_metadata():
    b = MlxWhisperBackend()
    assert b.name == "mlx_whisper"
    assert "large-v3" in b.models
    assert "tiny" in b.models
    assert b.supports_diarize is False
    assert b.diarize_models == set()


@pytest.mark.asyncio
async def test_transcribe_returns_result():
    mock_mlx = MagicMock()
    mock_mlx.transcribe.return_value = {
        "text": "hello world",
        "segments": [
            {"start": 0.0, "end": 1.5, "text": "hello world"},
        ],
    }
    with patch("scriba.backends.mlx_whisper._import_mlx_whisper", return_value=mock_mlx):
        backend = MlxWhisperBackend()
        config = TranscriptionConfig(model="tiny")
        result = await backend.transcribe(Path("/fake/audio.wav"), config)
    assert result.text == "hello world"
    assert result.backend == "mlx_whisper"
    assert result.cost_cents == 0.0
    assert result.diarized is False
    assert len(result.segments) == 1
    assert result.segments[0].speaker is None


@pytest.mark.asyncio
async def test_transcribe_empty_segments():
    mock_mlx = MagicMock()
    mock_mlx.transcribe.return_value = {"text": "", "segments": []}
    with patch("scriba.backends.mlx_whisper._import_mlx_whisper", return_value=mock_mlx):
        result = await MlxWhisperBackend().transcribe(
            Path("/fake/audio.wav"), TranscriptionConfig(model="tiny")
        )
    assert result.text == ""
    assert result.segments == []


@pytest.mark.asyncio
async def test_transcribe_strips_whitespace():
    mock_mlx = MagicMock()
    mock_mlx.transcribe.return_value = {
        "text": "  hello  ",
        "segments": [{"start": 0.0, "end": 1.0, "text": "  hello  "}],
    }
    with patch("scriba.backends.mlx_whisper._import_mlx_whisper", return_value=mock_mlx):
        result = await MlxWhisperBackend().transcribe(
            Path("/fake/audio.wav"), TranscriptionConfig(model="tiny")
        )
    assert result.text == "hello"
    assert result.segments[0].text == "hello"


def test_estimate():
    with patch("scriba.backends.mlx_whisper._import_mlx_whisper", return_value=MagicMock()):
        est = MlxWhisperBackend().estimate(60.0, TranscriptionConfig(model="tiny"))
    assert est.backend == "mlx_whisper"
    assert est.cost_cents == 0.0
    assert est.time_seconds > 0


@pytest.mark.asyncio
async def test_transcribe_raises_when_unavailable():
    from scriba.errors import DependencyMissing
    with patch("scriba.backends.mlx_whisper._import_mlx_whisper", return_value=None):
        with pytest.raises(DependencyMissing):
            await MlxWhisperBackend().transcribe(
                Path("/fake/audio.wav"), TranscriptionConfig(model="tiny")
            )
