from scriba.errors import (
    ScribaError, DependencyMissing, BackendError, AudioError,
    SecretsError, RoutingError, BudgetExceeded, ConstraintConflict,
)


def test_hierarchy():
    for cls in [DependencyMissing, BackendError, AudioError, SecretsError,
                RoutingError, BudgetExceeded, ConstraintConflict]:
        err = cls("test") if cls in (AudioError, SecretsError, ConstraintConflict) else None
        if err is None:
            continue
        assert isinstance(err, ScribaError)
        assert isinstance(err, Exception)


def test_dependency_missing_with_hint():
    err = DependencyMissing("ffmpeg", hint="brew install ffmpeg")
    assert "ffmpeg" in str(err)
    assert err.hint == "brew install ffmpeg"
    assert isinstance(err, ScribaError)


def test_backend_error_wraps_cause():
    cause = RuntimeError("OOM")
    err = BackendError("mlx_whisper", cause=cause, suggestion="try a smaller model")
    assert err.backend == "mlx_whisper"
    assert err.cause is cause
    assert err.suggestion == "try a smaller model"
    assert isinstance(err, ScribaError)


def test_routing_error_lists_missing():
    err = RoutingError(missing=["mlx-whisper", "whisperx", "openai"])
    assert len(err.missing) == 3
    assert isinstance(err, ScribaError)


def test_budget_exceeded():
    err = BudgetExceeded(estimated_cents=25.0, budget_cents=10)
    assert err.estimated_cents == 25.0
    assert err.budget_cents == 10
    assert isinstance(err, ScribaError)


def test_constraint_conflict_is_scriba_error():
    err = ConstraintConflict("fast + diarize conflict")
    assert isinstance(err, ScribaError)
