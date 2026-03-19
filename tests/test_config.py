import textwrap
from pathlib import Path
from scriba.config import ScribaConfig, load_config


def test_default_config():
    cfg = ScribaConfig()
    assert cfg.quality == "balanced"
    assert cfg.output_tier == "timestamped"
    assert cfg.output_format == "json"
    assert cfg.diarize is False
    assert cfg.prefer_local is True
    assert cfg.max_local_concurrency == 1
    assert cfg.max_cloud_concurrency == 3
    assert cfg.openai_model == "gpt-4o-mini-transcribe"
    assert cfg.mlx_model == "large-v3"
    assert cfg.whisperx_model == "large-v3"
    assert cfg.max_budget_cents_per_job == 50


def test_load_config_missing_file(tmp_path: Path):
    cfg = load_config(tmp_path / "nonexistent.toml")
    assert cfg == ScribaConfig()


def test_load_config_from_toml(tmp_path: Path):
    toml_file = tmp_path / "config.toml"
    toml_file.write_text(textwrap.dedent("""\
        [defaults]
        quality = "high"
        diarize = true
        [backends]
        prefer_local = false
        [concurrency]
        max_cloud = 5
        [openai]
        model = "gpt-4o-transcribe"
        max_budget_cents_per_job = 100
        [mlx]
        default_model = "medium"
    """))
    cfg = load_config(toml_file)
    assert cfg.quality == "high"
    assert cfg.diarize is True
    assert cfg.prefer_local is False
    assert cfg.max_cloud_concurrency == 5
    assert cfg.openai_model == "gpt-4o-transcribe"
    assert cfg.max_budget_cents_per_job == 100
    assert cfg.mlx_model == "medium"
    assert cfg.output_format == "json"
    assert cfg.whisperx_model == "large-v3"


def test_load_config_partial_toml(tmp_path: Path):
    toml_file = tmp_path / "config.toml"
    toml_file.write_text("[defaults]\nquality = \"fast\"\n")
    cfg = load_config(toml_file)
    assert cfg.quality == "fast"
    assert cfg.output_tier == "timestamped"
