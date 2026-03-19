import pytest

from scriba.errors import ConstraintConflict
from scriba.router.constraints import Constraints, normalize, validate


def test_defaults():
    c = Constraints()
    assert c.quality == "balanced"
    assert c.diarize is False
    assert c.budget_cents is None


def test_normalize_diarized_tier_implies_diarize():
    c = Constraints(output_tier="diarized", diarize=False)
    n = normalize(c)
    assert n.diarize is True


def test_normalize_enriched_tier_implies_diarize():
    c = Constraints(output_tier="enriched")
    n = normalize(c)
    assert n.diarize is True


def test_validate_diarize_false_with_diarized_tier():
    c = Constraints(output_tier="diarized", diarize=False, _diarize_explicit=True)
    with pytest.raises(ConstraintConflict, match="diarize"):
        validate(c)


def test_validate_fast_diarize_conflict():
    c = Constraints(quality="fast", diarize=True)
    with pytest.raises(ConstraintConflict, match="fast.*diariz"):
        validate(c)


def test_validate_ok():
    c = Constraints(quality="high", diarize=True)
    validate(c)


def test_constraint_conflict_is_scriba_error():
    from scriba.errors import ScribaError
    with pytest.raises(ScribaError):
        raise ConstraintConflict("test")
