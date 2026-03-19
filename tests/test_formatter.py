import json
import pytest
from scriba.contracts import Segment, TranscriptionResult
from scriba.output.formatter import format_result


def _make_result(diarized=False):
    segments = [
        Segment(start=0.0, end=1.5, text="Hello there.", speaker="SPEAKER_00" if diarized else None),
        Segment(start=2.0, end=4.0, text="How are you?", speaker="SPEAKER_01" if diarized else None),
    ]
    return TranscriptionResult(
        text="Hello there. How are you?", segments=segments,
        duration_seconds=4.0, model_used="large-v3", backend="mlx_whisper",
        cost_cents=0.0, diarized=diarized,
    )


def test_format_json():
    output = format_result(_make_result(), output_format="json", output_tier="timestamped")
    parsed = json.loads(output)
    assert parsed["text"] == "Hello there. How are you?"
    assert len(parsed["segments"]) == 2


def test_format_json_enriched():
    r = _make_result()
    r.enrichment_available = True
    output = format_result(r, output_format="json", output_tier="enriched")
    parsed = json.loads(output)
    assert parsed["enrichment_available"] is True


def test_format_text_raw():
    assert format_result(_make_result(), output_format="text", output_tier="raw") == "Hello there. How are you?"


def test_format_text_timestamped():
    output = format_result(_make_result(), output_format="text", output_tier="timestamped")
    assert "[0:00:00" in output
    assert "Hello there." in output


def test_format_text_diarized():
    output = format_result(_make_result(diarized=True), output_format="text", output_tier="diarized")
    assert "SPEAKER_00" in output
    assert "SPEAKER_01" in output


def test_format_srt():
    output = format_result(_make_result(), output_format="srt", output_tier="timestamped")
    assert "1\n" in output
    assert "00:00:00,000 --> 00:00:01,500" in output
    assert "Hello there." in output


def test_format_vtt():
    output = format_result(_make_result(), output_format="vtt", output_tier="timestamped")
    assert output.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.500" in output


def test_format_md_diarized():
    output = format_result(_make_result(diarized=True), output_format="md", output_tier="diarized")
    assert "**SPEAKER_00**" in output
    assert "**SPEAKER_01**" in output
