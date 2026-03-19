from pathlib import Path

import pytest

from scriba.media.ingest import probe_media, extract_audio, compress_for_upload, MediaInfo
from scriba.errors import AudioError

FIXTURES = Path(__file__).parent / "fixtures"


def test_probe_wav():
    info = probe_media(FIXTURES / "test_tone.wav")
    assert info.duration_seconds == pytest.approx(2.0, abs=0.5)
    assert info.has_audio is True
    assert info.has_video is False
    assert info.file_size_bytes > 0


def test_probe_video():
    info = probe_media(FIXTURES / "test_video.mp4")
    assert info.has_audio is True
    assert info.has_video is True
    assert info.duration_seconds == pytest.approx(2.0, abs=0.5)


def test_probe_nonexistent():
    with pytest.raises(AudioError, match="not found"):
        probe_media(Path("/nonexistent/file.wav"))


def test_extract_audio_from_wav(tmp_path: Path):
    out = extract_audio(FIXTURES / "test_tone.wav", tmp_path / "out.wav")
    assert out.exists()
    assert out.stat().st_size > 0
    info = probe_media(out)
    assert info.sample_rate == 16000
    assert info.channels == 1


def test_extract_audio_from_video(tmp_path: Path):
    out = extract_audio(FIXTURES / "test_video.mp4", tmp_path / "out.wav")
    assert out.exists()
    info = probe_media(out)
    assert info.has_audio is True
    assert info.has_video is False
    assert info.sample_rate == 16000


def test_compress_for_upload(tmp_path: Path):
    out = compress_for_upload(FIXTURES / "test_tone.wav", tmp_path / "out.mp3")
    assert out.exists()
    assert out.stat().st_size > 0
    assert out.stat().st_size < (FIXTURES / "test_tone.wav").stat().st_size
