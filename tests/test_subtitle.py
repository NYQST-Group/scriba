from pathlib import Path
import pytest
from scriba.contracts import Segment
from scriba.formatting import generate_srt, generate_vtt
from scriba.media.subtitle import burn_subtitles

FIXTURES = Path(__file__).parent / "fixtures"


def test_generate_srt():
    segments = [Segment(start=0.0, end=1.5, text="Hello"), Segment(start=2.0, end=3.5, text="World")]
    srt = generate_srt(segments)
    assert "1\n" in srt
    assert "00:00:00,000 --> 00:00:01,500" in srt
    assert "Hello" in srt
    assert "2\n" in srt


def test_generate_srt_with_speakers():
    segments = [
        Segment(start=0.0, end=2.0, text="Hello", speaker="SPEAKER_00"),
        Segment(start=1.0, end=2.5, text="World", speaker="SPEAKER_01"),
    ]
    srt = generate_srt(segments)
    assert "SPEAKER_00" in srt
    assert "SPEAKER_01" in srt


def test_generate_vtt():
    segments = [Segment(start=0.0, end=1.5, text="Hello")]
    vtt = generate_vtt(segments)
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.500" in vtt


def test_hard_burn_escapes_special_path_chars(tmp_path):
    """SRT path with colons is escaped for ffmpeg filter."""
    from scriba.media.subtitle import _escape_ffmpeg_path
    special_path = tmp_path / "has:colon.srt"
    escaped = _escape_ffmpeg_path(special_path)
    assert "\\:" in escaped
    assert "has:colon" not in escaped


def test_burn_subtitles_soft(tmp_path: Path):
    video = FIXTURES / "test_video.mp4"
    if not video.exists():
        pytest.skip("test video fixture not available")
    srt_path = tmp_path / "subs.srt"
    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n\n")
    out = burn_subtitles(video, srt_path, tmp_path / "out.mp4", mode="soft")
    assert out.exists()
    assert out.stat().st_size > 0
