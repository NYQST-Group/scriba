from scriba.contracts import (
    Segment, TranscriptionConfig, TranscriptionResult, Estimate, RoutingDecision,
)


def test_segment_defaults():
    s = Segment(start=0.0, end=1.5, text="hello")
    assert s.speaker is None


def test_segment_with_speaker():
    s = Segment(start=0.0, end=1.5, text="hello", speaker="SPEAKER_00")
    assert s.speaker == "SPEAKER_00"


def test_transcription_config_defaults():
    c = TranscriptionConfig(model="large-v3")
    assert c.language is None
    assert c.diarize is False
    assert c.speakers is None
    assert c.output_tier == "timestamped"


def test_transcription_result_defaults():
    r = TranscriptionResult(
        text="hello", segments=[], duration_seconds=1.0,
        model_used="large-v3", backend="mlx_whisper", cost_cents=0.0, diarized=False,
    )
    assert r.enrichment_available is False
    assert r.routing is None


def test_estimate_fields():
    e = Estimate(
        backend="mlx_whisper", model="large-v3", time_seconds=5.0,
        cost_cents=0.0, available=True, recommended=True, reason_unavailable=None,
    )
    assert e.recommended is True


def test_routing_decision():
    selected = Estimate(backend="mlx_whisper", model="large-v3", time_seconds=5.0,
                        cost_cents=0.0, available=True, recommended=True)
    rd = RoutingDecision(selected=selected, alternatives=[], trade_offs=None)
    assert rd.trade_offs is None
    assert rd.alternatives == []


def test_routing_decision_with_trade_offs():
    selected = Estimate(backend="openai_stt", model="gpt-4o-transcribe", time_seconds=10.0,
                        cost_cents=12.0, available=True, recommended=True)
    alt = Estimate(backend="whisperx", model="large-v3", time_seconds=60.0,
                   cost_cents=0.0, available=True, recommended=False)
    rd = RoutingDecision(
        selected=selected, alternatives=[alt],
        trade_offs=["Cloud selected to meet 30s timeout; local would take ~60s"],
    )
    assert len(rd.trade_offs) == 1
    assert len(rd.alternatives) == 1
