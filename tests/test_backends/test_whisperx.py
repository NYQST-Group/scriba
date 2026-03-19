from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from scriba.backends.whisperx import WhisperXBackend
from scriba.contracts import TranscriptionConfig


def test_is_available():
    with patch("scriba.backends.whisperx._import_whisperx", return_value=MagicMock()):
        assert WhisperXBackend().is_available() is True


def test_is_unavailable():
    with patch("scriba.backends.whisperx._import_whisperx", return_value=None):
        assert WhisperXBackend().is_available() is False


@pytest.mark.asyncio
async def test_transcribe_with_diarization():
    mock_wx = MagicMock()
    mock_model = MagicMock()
    mock_wx.load_model.return_value = mock_model
    mock_model.transcribe.return_value = {
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "hello", "speaker": "SPEAKER_00"},
            {"start": 2.5, "end": 4.0, "text": "world", "speaker": "SPEAKER_01"},
        ],
    }
    mock_wx.load_audio.return_value = MagicMock()
    mock_wx.load_align_model.return_value = (MagicMock(), MagicMock())
    mock_wx.align.return_value = mock_model.transcribe.return_value
    mock_wx.DiarizationPipeline.return_value = MagicMock()
    mock_wx.assign_word_speakers.return_value = mock_model.transcribe.return_value

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False

    with patch("scriba.backends.whisperx._import_whisperx", return_value=mock_wx), \
         patch.dict("sys.modules", {"torch": mock_torch}):
        backend = WhisperXBackend()
        config = TranscriptionConfig(model="large-v3", diarize=True, speakers=2)
        result = await backend.transcribe(Path("/fake/audio.wav"), config)

    assert result.diarized is True
    assert result.backend == "whisperx"
    assert len(result.segments) == 2
    assert result.segments[0].speaker == "SPEAKER_00"


def test_estimate():
    with patch("scriba.backends.whisperx._import_whisperx", return_value=MagicMock()):
        est = WhisperXBackend().estimate(60.0, TranscriptionConfig(model="large-v3", diarize=True))
    assert est.backend == "whisperx"
    assert est.cost_cents == 0.0
