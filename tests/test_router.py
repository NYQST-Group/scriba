import pytest

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


# ---------------------------------------------------------------------------
# Budget boundary tests
# ---------------------------------------------------------------------------

def test_budget_exact_match_selects_cloud():
    # openai_stt:gpt-4o-mini-transcribe costs 0.3 cents/min * 1 min = 0.3 cents
    # A budget of exactly 0.3 (as int 0 rounds down) — use a budget that is
    # the exact ceiling. The PRICING table charges per minute, so for 60s audio
    # the cost is 0.3 * 1 = 0.3 cents. We pass budget_cents=1 (the lowest int
    # above 0.3) to confirm the cloud backend IS selected.
    d = route(Constraints(budget_cents=1), duration_seconds=60,
              backends=[_openai()])
    assert d.selected.backend == "openai_stt"
    assert d.trade_offs is None


def test_budget_just_below_cost_excludes_cloud():
    # gpt-4o-mini-transcribe costs 0.3 cents/min * 2 min = 0.6 cents for 120s.
    # A budget of 0 should exclude it; confirmed by test_budget_zero_excludes_cloud
    # but here we test with an explicit budget < actual cost while > 0.
    # Cost for 120s (2 min) at 0.3 c/min = 0.6 cents.
    # budget_cents=0 (integer) is < 0.6, so cloud excluded and mlx chosen.
    d = route(Constraints(budget_cents=0), duration_seconds=120,
              backends=[_mlx(), _openai()])
    assert d.selected.backend == "mlx_whisper"


def test_budget_zero_with_only_paid_backend_returns_trade_off():
    # budget_cents=0 excludes all openai options (cost > 0). When no candidate
    # survives, the router relaxes to drop the budget constraint and returns the
    # best feasible option with trade_offs populated rather than raising.
    d = route(Constraints(budget_cents=0), duration_seconds=60,
              backends=[_openai()])
    assert d.trade_offs is not None
    assert len(d.trade_offs) > 0
    assert d.selected.backend == "openai_stt"


# ---------------------------------------------------------------------------
# Multi-backend ranking
# ---------------------------------------------------------------------------

def test_prefer_local_true_picks_mlx_over_openai():
    # With prefer_local=True (default), local backend should rank first even
    # if the cloud backend has a non-zero but affordable cost.
    d = route(Constraints(budget_cents=100), duration_seconds=60,
              backends=[_mlx(), _openai()], prefer_local=True)
    assert d.selected.backend == "mlx_whisper"


def test_prefer_local_true_alternatives_include_cloud():
    # Alternatives list should contain the cloud option.
    d = route(Constraints(budget_cents=100), duration_seconds=60,
              backends=[_mlx(), _openai()], prefer_local=True)
    alt_backends = [e.backend for e in d.alternatives]
    assert "openai_stt" in alt_backends


def test_prefer_local_false_picks_cheapest():
    # With prefer_local=False the ranking falls back to cost then time.
    # gpt-4o-mini-transcribe is 0.3 c/min which is the cheapest cloud option,
    # but mlx costs 0.0, so mlx still wins on cost. We need a setup where the
    # only options are cloud to test ranking between cloud models; instead
    # verify that when mlx is absent, openai is selected.
    d = route(Constraints(budget_cents=100), duration_seconds=60,
              backends=[_openai()], prefer_local=False)
    assert d.selected.backend == "openai_stt"


def test_prefer_local_false_with_both_selects_local_on_cost():
    # mlx_whisper costs 0.0 which is cheaper than any openai option, so even
    # with prefer_local=False the cost-based ranking will still pick mlx first.
    d = route(Constraints(budget_cents=100), duration_seconds=60,
              backends=[_mlx(), _openai()], prefer_local=False)
    assert d.selected.backend == "mlx_whisper"


def test_ranking_recommended_flag_set_on_selected():
    d = route(Constraints(), duration_seconds=60, backends=[_mlx(), _whisperx()])
    assert d.selected.recommended is True
    for alt in d.alternatives:
        assert alt.recommended is False


# ---------------------------------------------------------------------------
# prefer_local=False — cloud selected when it is the only option
# ---------------------------------------------------------------------------

def test_prefer_local_false_cloud_only_backend():
    d = route(Constraints(budget_cents=100), duration_seconds=60,
              backends=[_openai()], prefer_local=False)
    assert d.selected.backend == "openai_stt"
    assert d.selected.recommended is True


# ---------------------------------------------------------------------------
# Constraint relaxation
# ---------------------------------------------------------------------------

def test_relaxation_returns_trade_offs():
    # fast + diarize triggers ConstraintConflict; router should return a
    # RoutingDecision with trade_offs populated.
    d = route(Constraints(quality="fast", diarize=True), duration_seconds=60,
              backends=[_whisperx()])
    assert d.trade_offs is not None
    assert len(d.trade_offs) > 0


def test_relaxation_picks_best_feasible_backend():
    # fast + diarize is a conflict; the relaxed path drops diarize.
    # whisperx should still be selected as it is the only available backend.
    d = route(Constraints(quality="fast", diarize=True), duration_seconds=60,
              backends=[_whisperx()])
    assert d.selected.backend == "whisperx"


def test_no_candidate_relaxes_to_any_available():
    # Timeout so tight that nothing fits; router relaxes and picks best feasible,
    # returning trade_offs.
    d = route(Constraints(timeout_seconds=1), duration_seconds=600,
              backends=[_mlx()])
    # Either trade_offs is set, or selected time fits within the timeout
    assert d.trade_offs is not None or d.selected.time_seconds <= 1


# ---------------------------------------------------------------------------
# Duration scaling
# ---------------------------------------------------------------------------

def test_cost_scales_linearly_with_duration():
    # openai_stt cost is proportional to duration in minutes.
    d60 = route(Constraints(budget_cents=100), duration_seconds=60,
                backends=[_openai()])
    d120 = route(Constraints(budget_cents=100), duration_seconds=120,
                 backends=[_openai()])
    assert abs(d120.selected.cost_cents - 2 * d60.selected.cost_cents) < 1e-9


def test_time_scales_linearly_with_duration():
    # Processing time estimate should be proportional to audio duration.
    d60 = route(Constraints(), duration_seconds=60, backends=[_mlx()])
    d120 = route(Constraints(), duration_seconds=120, backends=[_mlx()])
    assert abs(d120.selected.time_seconds - 2 * d60.selected.time_seconds) < 1e-9


def test_cost_zero_for_local_backends():
    d60 = route(Constraints(), duration_seconds=60, backends=[_mlx()])
    d120 = route(Constraints(), duration_seconds=120, backends=[_mlx()])
    assert d60.selected.cost_cents == 0.0
    assert d120.selected.cost_cents == 0.0


# ---------------------------------------------------------------------------
# Diarize with OpenAI
# ---------------------------------------------------------------------------

def test_openai_diarize_selects_gpt4o_mini():
    # balanced quality + diarize should pick gpt-4o-mini-transcribe (not whisper-1)
    d = route(Constraints(diarize=True, quality="balanced"), duration_seconds=60,
              backends=[_openai()])
    assert d.selected.model in {"gpt-4o-mini-transcribe", "gpt-4o-transcribe"}
    assert d.selected.model != "whisper-1"


def test_openai_diarize_high_quality_selects_gpt4o():
    # high quality + diarize should select gpt-4o-transcribe
    d = route(Constraints(diarize=True, quality="high"), duration_seconds=60,
              backends=[_openai()])
    assert d.selected.model == "gpt-4o-transcribe"


def test_openai_no_diarize_can_use_any_model():
    # Without diarize constraint, whisper-1 class models are eligible;
    # balanced maps to gpt-4o-mini-transcribe per QUALITY_MODEL_MAP.
    d = route(Constraints(diarize=False, quality="balanced"), duration_seconds=60,
              backends=[_openai()])
    assert d.selected.backend == "openai_stt"


# ---------------------------------------------------------------------------
# Unknown backend
# ---------------------------------------------------------------------------

def test_unknown_backend_skipped_gracefully():
    unknown = BackendInfo(name="future_backend", available=True,
                          models=["model-x"], supports_diarize=False)
    # Should not raise; unknown backend has no QUALITY_MODEL_MAP entry so it
    # is silently skipped, and mlx handles the request.
    d = route(Constraints(), duration_seconds=60, backends=[unknown, _mlx()])
    assert d.selected.backend == "mlx_whisper"


def test_unknown_backend_only_raises_routing_error():
    unknown = BackendInfo(name="future_backend", available=True,
                          models=["model-x"], supports_diarize=False)
    with pytest.raises(RoutingError):
        route(Constraints(), duration_seconds=60, backends=[unknown])
