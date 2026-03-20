from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
import pytest
from scriba.backends.openai_stt import OpenAISTTBackend
from scriba.contracts import TranscriptionConfig


def test_is_available():
    with patch("scriba.backends.openai_stt._import_openai", return_value=MagicMock()):
        assert OpenAISTTBackend(api_key="sk-test").is_available() is True


def test_is_unavailable_no_lib():
    with patch("scriba.backends.openai_stt._import_openai", return_value=None):
        assert OpenAISTTBackend(api_key="sk-test").is_available() is False


def test_is_unavailable_no_key():
    with patch("scriba.backends.openai_stt._import_openai", return_value=MagicMock()):
        assert OpenAISTTBackend(api_key=None).is_available() is False


@pytest.mark.asyncio
async def test_transcribe_verbose_json():
    mock_resp = MagicMock()
    mock_resp.model_dump.return_value = {
        "text": "hello world", "duration": 2.0,
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "hello"},
            {"start": 1.0, "end": 2.0, "text": "world"},
        ],
    }
    mock_client = AsyncMock()
    mock_client.audio.transcriptions.create.return_value = mock_resp

    with patch("scriba.backends.openai_stt._import_openai", return_value=MagicMock()), \
         patch("pathlib.Path.stat") as mock_stat, \
         patch("pathlib.Path.open", create=True) as mock_open:
        mock_stat.return_value = MagicMock(st_size=1000)
        mock_open.return_value = MagicMock()
        backend = OpenAISTTBackend(api_key="sk-test")
        backend._make_client = lambda: mock_client
        backend._call_api = AsyncMock(return_value=mock_resp)
        config = TranscriptionConfig(model="whisper-1")
        result = await backend.transcribe(Path("/fake/audio.wav"), config)

    assert result.text == "hello world"
    assert result.backend == "openai_stt"
    assert result.diarized is False
    assert len(result.segments) == 2


def test_estimate_cost():
    with patch("scriba.backends.openai_stt._import_openai", return_value=MagicMock()):
        est = OpenAISTTBackend(api_key="sk-test").estimate(600.0, TranscriptionConfig(model="whisper-1"))
    assert est.cost_cents == pytest.approx(6.0)
    assert est.backend == "openai_stt"


def test_set_api_key():
    with patch("scriba.backends.openai_stt._import_openai", return_value=MagicMock()):
        backend = OpenAISTTBackend(api_key=None)
        assert backend.is_available() is False
        backend.set_api_key("sk-new-key")
        assert backend.is_available() is True
        assert backend._api_key == "sk-new-key"


@pytest.mark.asyncio
async def test_call_api_seeks_file():
    mock_file = MagicMock()
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_resp)

    backend = OpenAISTTBackend(api_key="sk-test")
    result = await backend._call_api(mock_client, file=mock_file, model="whisper-1")

    mock_file.seek.assert_called_with(0)
    assert result is mock_resp
