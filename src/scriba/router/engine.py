"""Constraint-based backend router."""
from __future__ import annotations

from dataclasses import dataclass, field

from scriba.contracts import Estimate, RoutingDecision
from scriba.errors import ConstraintConflict, RoutingError
from scriba.router.constraints import Constraints, normalize, validate
from scriba.router.cost_model import estimate_cost_cents, estimate_time_seconds

QUALITY_MODEL_MAP: dict[str, dict[str, str]] = {
    "mlx_whisper": {"fast": "tiny", "balanced": "medium", "high": "large-v3"},
    "whisperx": {"fast": "small", "balanced": "medium", "high": "large-v3"},
    "openai_stt": {"fast": "gpt-4o-mini-transcribe", "balanced": "gpt-4o-mini-transcribe", "high": "gpt-4o-transcribe"},
}


@dataclass
class BackendInfo:
    name: str
    available: bool
    models: list[str]
    supports_diarize: bool = False
    diarize_models: set[str] = field(default_factory=set)


def route(
    constraints: Constraints,
    *,
    duration_seconds: float,
    backends: list[BackendInfo],
    prefer_local: bool = True,
) -> RoutingDecision:
    """Select the best backend and model for the given constraints."""
    try:
        constraints = normalize(constraints)
        validate(constraints)
    except ConstraintConflict as e:
        # Return best feasible with trade-off
        fallback_constraints = normalize(Constraints(
            quality=constraints.quality, budget_cents=constraints.budget_cents,
            timeout_seconds=constraints.timeout_seconds, diarize=False,
            output_tier="timestamped", language=constraints.language,
        ))
        fallback = _find_best(fallback_constraints, duration_seconds=duration_seconds,
                              backends=backends, prefer_local=prefer_local)
        if fallback is None:
            raise RoutingError(missing=[b.name for b in backends if not b.available])
        return RoutingDecision(selected=fallback, alternatives=[], trade_offs=[str(e)])

    candidates = _build_candidates(constraints, duration_seconds=duration_seconds, backends=backends)

    if not candidates:
        available_names = [b.name for b in backends if b.available]
        if not available_names:
            raise RoutingError(missing=[b.name for b in backends])
        relaxed = Constraints(quality=constraints.quality, language=constraints.language)
        fallback_candidates = _build_candidates(relaxed, duration_seconds=duration_seconds, backends=backends)
        if not fallback_candidates:
            raise RoutingError(missing=[b.name for b in backends if not b.available])
        best = _rank(fallback_candidates, constraints, prefer_local=prefer_local)[0]
        return RoutingDecision(
            selected=best, alternatives=[],
            trade_offs=[f"No backend supports all constraints. Relaxed to: {best.backend}:{best.model}"],
        )

    ranked = _rank(candidates, constraints, prefer_local=prefer_local)
    selected = ranked[0]
    alternatives = ranked[1:]
    return RoutingDecision(selected=selected, alternatives=alternatives, trade_offs=None)


def _find_best(constraints, *, duration_seconds, backends, prefer_local) -> Estimate | None:
    candidates = _build_candidates(constraints, duration_seconds=duration_seconds, backends=backends)
    if not candidates:
        return None
    return _rank(candidates, constraints, prefer_local=prefer_local)[0]


def _build_candidates(constraints, *, duration_seconds, backends) -> list[Estimate]:
    duration_minutes = duration_seconds / 60.0
    candidates: list[Estimate] = []

    for backend in backends:
        if not backend.available:
            continue
        if constraints.diarize and not backend.supports_diarize:
            continue

        model_map = QUALITY_MODEL_MAP.get(backend.name, {})
        model = model_map.get(constraints.quality)
        if model is None:
            continue

        if constraints.diarize and backend.diarize_models and model not in backend.diarize_models:
            for dm in backend.diarize_models:
                if dm in backend.models:
                    model = dm
                    break
            else:
                continue

        cost = estimate_cost_cents(backend.name, model, duration_minutes=duration_minutes)
        if constraints.budget_cents is not None and cost > constraints.budget_cents:
            continue

        time_est = estimate_time_seconds(backend.name, model, duration_seconds=duration_seconds)
        if constraints.timeout_seconds is not None and time_est > constraints.timeout_seconds:
            continue

        candidates.append(Estimate(
            backend=backend.name, model=model, time_seconds=time_est,
            cost_cents=cost, available=True, recommended=False,
        ))
    return candidates


def _rank(candidates, constraints, *, prefer_local) -> list[Estimate]:
    def score(e: Estimate) -> tuple[int, float, float]:
        locality = 0 if prefer_local and e.backend != "openai_stt" else 1
        return (locality, e.cost_cents, e.time_seconds)

    ranked = sorted(candidates, key=score)
    if ranked:
        r = ranked[0]
        ranked[0] = Estimate(
            backend=r.backend, model=r.model, time_seconds=r.time_seconds,
            cost_cents=r.cost_cents, available=r.available, recommended=True,
            reason_unavailable=r.reason_unavailable,
        )
    return ranked
