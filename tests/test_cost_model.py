from pathlib import Path

import pytest

from scriba.router.cost_model import (
    estimate_cost_cents, estimate_time_seconds,
    PRICING, load_calibration, save_calibration_entry,
)


def test_pricing_table_has_all_models():
    assert "openai_stt:whisper-1" in PRICING
    assert "openai_stt:gpt-4o-mini-transcribe" in PRICING
    assert "openai_stt:gpt-4o-transcribe" in PRICING


def test_local_cost_is_zero():
    assert estimate_cost_cents("mlx_whisper", "large-v3", duration_minutes=60) == 0.0
    assert estimate_cost_cents("whisperx", "large-v3", duration_minutes=60) == 0.0


def test_openai_cost():
    cost = estimate_cost_cents("openai_stt", "whisper-1", duration_minutes=10)
    assert cost == pytest.approx(6.0)


def test_openai_mini_cost():
    cost = estimate_cost_cents("openai_stt", "gpt-4o-mini-transcribe", duration_minutes=10)
    assert cost == pytest.approx(3.0)


def test_time_estimate_mlx_tiny():
    t = estimate_time_seconds("mlx_whisper", "tiny", duration_seconds=600)
    assert t == pytest.approx(30.0)


def test_time_estimate_whisperx_diarize():
    t = estimate_time_seconds("whisperx", "large-v3", duration_seconds=600)
    assert t == pytest.approx(300.0)


def test_calibration_round_trip(tmp_path: Path):
    cal_path = tmp_path / "calibration.json"
    save_calibration_entry(cal_path, "mlx_whisper", "large-v3", audio_duration=60.0, wall_clock=20.0)
    save_calibration_entry(cal_path, "mlx_whisper", "large-v3", audio_duration=120.0, wall_clock=35.0)
    cal = load_calibration(cal_path)
    key = "mlx_whisper:large-v3"
    assert key in cal
    assert len(cal[key]) == 2


def test_calibration_missing_file(tmp_path: Path):
    cal = load_calibration(tmp_path / "nonexistent.json")
    assert cal == {}


def test_estimate_time_uses_calibration(tmp_path):
    cal_path = tmp_path / "calibration.json"
    for _ in range(3):
        save_calibration_entry(cal_path, "mlx_whisper", "medium", audio_duration=60.0, wall_clock=6.0)
    result = estimate_time_seconds("mlx_whisper", "medium", duration_seconds=60.0, calibration_path=cal_path)
    assert result < 10.0  # Calibrated ~6s, not default 12s


def test_estimate_time_calibration_ignores_zero_duration(tmp_path):
    cal_path = tmp_path / "calibration.json"
    save_calibration_entry(cal_path, "mlx_whisper", "medium", audio_duration=0.0, wall_clock=1.0)
    save_calibration_entry(cal_path, "mlx_whisper", "medium", audio_duration=60.0, wall_clock=6.0)
    result = estimate_time_seconds("mlx_whisper", "medium", duration_seconds=60.0, calibration_path=cal_path)
    assert result < 10.0


def test_estimate_time_no_calibration_uses_default():
    result = estimate_time_seconds("mlx_whisper", "medium", duration_seconds=60.0, calibration_path=None)
    assert result == 12.0  # 0.20 * 60
