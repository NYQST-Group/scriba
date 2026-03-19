from scriba.contracts import Segment
from scriba.formatting import (
    fmt_srt_ts, fmt_vtt_ts, fmt_text_ts,
    generate_srt, generate_vtt, _merge_overlapping,
)


def test_fmt_srt_ts_zero():
    assert fmt_srt_ts(0.0) == "00:00:00,000"


def test_fmt_srt_ts_complex():
    assert fmt_srt_ts(3661.5) == "01:01:01,500"


def test_fmt_vtt_ts_zero():
    assert fmt_vtt_ts(0.0) == "00:00:00.000"


def test_fmt_vtt_ts_complex():
    assert fmt_vtt_ts(3661.5) == "01:01:01.500"


def test_fmt_text_ts():
    assert fmt_text_ts(0.0) == "0:00:00"
    assert fmt_text_ts(3661.0) == "1:01:01"


def test_generate_srt_basic():
    segs = [Segment(start=0.0, end=1.5, text="Hello")]
    srt = generate_srt(segs)
    assert "1\n" in srt
    assert "00:00:00,000 --> 00:00:01,500" in srt
    assert "Hello" in srt


def test_generate_srt_empty():
    assert generate_srt([]) == ""


def test_generate_vtt_basic():
    segs = [Segment(start=0.0, end=1.5, text="Hello")]
    vtt = generate_vtt(segs)
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.500" in vtt


def test_generate_vtt_empty():
    vtt = generate_vtt([])
    assert vtt.startswith("WEBVTT")


def test_merge_overlapping_no_overlap():
    segs = [
        Segment(start=0.0, end=1.0, text="A", speaker="S0"),
        Segment(start=2.0, end=3.0, text="B", speaker="S1"),
    ]
    merged = _merge_overlapping(segs)
    assert len(merged) == 2


def test_merge_overlapping_same_speaker():
    """Overlapping segments from same speaker should NOT merge."""
    segs = [
        Segment(start=0.0, end=2.0, text="A", speaker="S0"),
        Segment(start=1.0, end=3.0, text="B", speaker="S0"),
    ]
    merged = _merge_overlapping(segs)
    assert len(merged) == 2


def test_merge_overlapping_different_speakers():
    """Overlapping segments from different speakers SHOULD merge."""
    segs = [
        Segment(start=0.0, end=2.0, text="Hello", speaker="SPEAKER_00"),
        Segment(start=1.0, end=2.5, text="World", speaker="SPEAKER_01"),
    ]
    merged = _merge_overlapping(segs)
    assert len(merged) == 1
    assert "SPEAKER_00" in merged[0].speaker
    assert "SPEAKER_01" in merged[0].speaker
    assert "Hello" in merged[0].text
    assert "World" in merged[0].text


def test_merge_overlapping_no_speakers():
    """Segments without speakers should not merge even if overlapping."""
    segs = [
        Segment(start=0.0, end=2.0, text="A"),
        Segment(start=1.0, end=3.0, text="B"),
    ]
    merged = _merge_overlapping(segs)
    assert len(merged) == 2


def test_srt_uses_merge():
    """generate_srt should merge overlapping segments."""
    segs = [
        Segment(start=0.0, end=2.0, text="Hello", speaker="S0"),
        Segment(start=1.0, end=2.5, text="World", speaker="S1"),
    ]
    srt = generate_srt(segs)
    assert "S0 & S1" in srt
    # Should only have 1 subtitle entry
    assert "2\n" not in srt
