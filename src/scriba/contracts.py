"""Data models for Scriba transcription pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass
class TranscriptionConfig:
    """Resolved configuration passed to a backend adapter."""
    model: str
    language: str | None = None
    diarize: bool = False
    speakers: int | None = None
    output_tier: str = "timestamped"


@dataclass
class Estimate:
    backend: str
    model: str
    time_seconds: float
    cost_cents: float
    available: bool
    recommended: bool
    reason_unavailable: str | None = None


@dataclass
class RoutingDecision:
    """Returned by the router engine."""
    selected: Estimate
    alternatives: list[Estimate] = field(default_factory=list)
    trade_offs: list[str] | None = None


@dataclass
class TranscriptionResult:
    text: str
    segments: list[Segment]
    duration_seconds: float
    model_used: str
    backend: str
    cost_cents: float
    diarized: bool
    enrichment_available: bool = False
    routing: RoutingDecision | None = None
