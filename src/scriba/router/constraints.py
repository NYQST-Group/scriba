"""Constraint model for routing decisions."""
from __future__ import annotations

from dataclasses import dataclass

from scriba.errors import ConstraintConflict


@dataclass
class Constraints:
    quality: str = "balanced"
    budget_cents: int | None = None
    timeout_seconds: int | None = None
    diarize: bool = False
    speakers: int | None = None
    output_tier: str = "timestamped"
    language: str | None = None
    _diarize_explicit: bool = False


def normalize(c: Constraints) -> Constraints:
    """Resolve parameter interactions."""
    diarize = c.diarize
    if c.output_tier in ("diarized", "enriched") and not c.diarize:
        if c._diarize_explicit:
            return c
        diarize = True
    return Constraints(
        quality=c.quality, budget_cents=c.budget_cents, timeout_seconds=c.timeout_seconds,
        diarize=diarize, speakers=c.speakers, output_tier=c.output_tier,
        language=c.language, _diarize_explicit=c._diarize_explicit,
    )


def validate(c: Constraints) -> None:
    """Raise ConstraintConflict if constraints are contradictory."""
    if c._diarize_explicit and not c.diarize and c.output_tier in ("diarized", "enriched"):
        raise ConstraintConflict(f"Cannot set diarize=false with output_tier={c.output_tier}")
    if c.quality == "fast" and c.diarize:
        raise ConstraintConflict("fast quality with diarization is not available — diarization is inherently slower")
