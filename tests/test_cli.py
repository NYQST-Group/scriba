from pathlib import Path

from click.testing import CliRunner

from scriba.cli import main

FIXTURES = Path(__file__).parent / "fixtures"


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "transcribe" in result.output
    assert "estimate" in result.output
    assert "backends" in result.output
    assert "configure" in result.output


def test_cli_backends():
    runner = CliRunner()
    result = runner.invoke(main, ["backends"])
    assert result.exit_code == 0
    assert "mlx_whisper" in result.output or "whisperx" in result.output or "openai_stt" in result.output


def test_cli_transcribe_missing_file():
    runner = CliRunner()
    result = runner.invoke(main, ["transcribe", "/nonexistent/file.wav"])
    assert result.exit_code != 0


def test_cli_estimate_missing_file():
    runner = CliRunner()
    result = runner.invoke(main, ["estimate", "/nonexistent/file.wav"])
    assert result.exit_code != 0
