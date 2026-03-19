import pytest

from scriba.contracts import Estimate, RoutingDecision
from scriba.errors import RoutingError
from scriba.router.constraints import Constraints
from scriba.router.engine import route, BackendInfo


def _mlx(available=True):
    return BackendInfo(name="mlx_whisper", available=available,
                       models=["tiny", "base", "small", "medium", "large-v3"], supports_diarize=False)


def _whisperx(available=True):
    return BackendInfo(name="whisperx", available=available,
                       models=["tiny", "base", "small", "medium", "large-v3"], supports_diarize=True)


def _openai(available=True):
    return BackendInfo(name="openai_stt", available=available,
                       models=["whisper-1", "gpt-4o-mini-transcribe", "gpt-4o-transcribe"],
                       supports_diarize=True, diarize_models={"gpt-4o-mini-transcribe", "gpt-4o-transcribe"})


def test_default_selects_mlx_balanced():
    d = route(Constraints(), duration_seconds=60, backends=[_mlx(), _whisperx()])
    assert d.selected.backend == "mlx_whisper"
    assert d.selected.model == "medium"


def test_diarize_selects_whisperx():
    d = route(Constraints(diarize=True, quality="balanced"), duration_seconds=60, backends=[_mlx(), _whisperx()])
    assert d.selected.backend == "whisperx"


def test_diarize_excludes_mlx():
    d = route(Constraints(diarize=True), duration_seconds=60, backends=[_mlx()])
    assert d.trade_offs is not None
    assert len(d.trade_offs) > 0


def test_budget_zero_excludes_cloud():
    d = route(Constraints(budget_cents=0), duration_seconds=60, backends=[_mlx(), _openai()])
    assert d.selected.backend == "mlx_whisper"


def test_high_quality_selects_large_model():
    d = route(Constraints(quality="high"), duration_seconds=60, backends=[_mlx()])
    assert d.selected.model == "large-v3"


def test_fast_selects_tiny():
    d = route(Constraints(quality="fast"), duration_seconds=60, backends=[_mlx()])
    assert d.selected.model == "tiny"


def test_timeout_filters_slow_options():
    # balanced->medium: 0.20*60=12s, fits in 15s
    d = route(Constraints(timeout_seconds=15), duration_seconds=60, backends=[_mlx()])
    assert d.selected.model == "medium"
    # With 5s timeout, medium doesn't fit — router falls back with trade-off
    d2 = route(Constraints(timeout_seconds=5), duration_seconds=60, backends=[_mlx()])
    assert d2.trade_offs is not None or d2.selected.time_seconds <= 5


def test_openai_diarize_excludes_whisper1():
    d = route(Constraints(diarize=True, quality="balanced"), duration_seconds=60, backends=[_openai()])
    assert d.selected.model != "whisper-1"


def test_no_backends_available():
    with pytest.raises(RoutingError):
        route(Constraints(), duration_seconds=60, backends=[])


def test_all_backends_unavailable():
    with pytest.raises(RoutingError):
        route(Constraints(), duration_seconds=60, backends=[_mlx(available=False)])
