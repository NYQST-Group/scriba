"""Configuration loader for Scriba."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "scriba" / "config.toml"


@dataclass
class ScribaConfig:
    quality: str = "balanced"
    output_tier: str = "timestamped"
    output_format: str = "json"
    diarize: bool = False
    prefer_local: bool = True
    max_local_concurrency: int = 1
    max_cloud_concurrency: int = 3
    calibration_path: str = "~/.config/scriba/calibration.json"
    calibration_max_samples: int = 10
    calibration_stale_days: int = 30
    openai_model: str = "gpt-4o-mini-transcribe"
    max_budget_cents_per_job: int = 50
    mlx_model: str = "large-v3"
    mlx_cache_dir: str = "~/.cache/scriba/models"
    whisperx_model: str = "large-v3"


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> ScribaConfig:
    if not path.exists():
        return ScribaConfig()
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    defaults = raw.get("defaults", {})
    backends = raw.get("backends", {})
    concurrency = raw.get("concurrency", {})
    calibration = raw.get("calibration", {})
    openai = raw.get("openai", {})
    mlx = raw.get("mlx", {})
    whisperx = raw.get("whisperx", {})
    return ScribaConfig(
        quality=defaults.get("quality", "balanced"),
        output_tier=defaults.get("output_tier", "timestamped"),
        output_format=defaults.get("output_format", "json"),
        diarize=defaults.get("diarize", False),
        prefer_local=backends.get("prefer_local", True),
        max_local_concurrency=concurrency.get("max_local", 1),
        max_cloud_concurrency=concurrency.get("max_cloud", 3),
        calibration_path=calibration.get("path", "~/.config/scriba/calibration.json"),
        calibration_max_samples=calibration.get("max_samples", 10),
        calibration_stale_days=calibration.get("stale_days", 30),
        openai_model=openai.get("model", "gpt-4o-mini-transcribe"),
        max_budget_cents_per_job=openai.get("max_budget_cents_per_job", 50),
        mlx_model=mlx.get("default_model", "large-v3"),
        mlx_cache_dir=mlx.get("cache_dir", "~/.cache/scriba/models"),
        whisperx_model=whisperx.get("default_model", "large-v3"),
    )
