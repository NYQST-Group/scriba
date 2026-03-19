"""Scriba error hierarchy with actionable recovery messages."""
from __future__ import annotations


class ScribaError(Exception):
    """Base error for all Scriba operations."""


class DependencyMissing(ScribaError):
    def __init__(self, name: str, *, hint: str | None = None):
        self.name = name
        self.hint = hint
        msg = f"Missing dependency: {name}"
        if hint:
            msg += f" — {hint}"
        super().__init__(msg)


class BackendError(ScribaError):
    def __init__(self, backend: str, *, cause: Exception | None = None, suggestion: str | None = None):
        self.backend = backend
        self.cause = cause
        self.suggestion = suggestion
        msg = f"Backend '{backend}' failed"
        if cause:
            msg += f": {cause}"
        if suggestion:
            msg += f" — {suggestion}"
        super().__init__(msg)


class AudioError(ScribaError):
    pass


class SecretsError(ScribaError):
    pass


class RoutingError(ScribaError):
    def __init__(self, msg: str = "No backends available", *, missing: list[str] | None = None):
        self.missing = missing or []
        if self.missing:
            msg += f". Missing: {', '.join(self.missing)}"
        super().__init__(msg)


class BudgetExceeded(ScribaError):
    def __init__(self, *, estimated_cents: float, budget_cents: int):
        self.estimated_cents = estimated_cents
        self.budget_cents = budget_cents
        super().__init__(f"Estimated cost {estimated_cents:.1f}c exceeds budget {budget_cents}c")


class ConstraintConflict(ScribaError):
    """Raised when constraints are contradictory."""
    pass
