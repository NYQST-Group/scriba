from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from scriba.cli import main
from scriba.contracts import Estimate, RoutingDecision, Segment, TranscriptionResult
from scriba.media.ingest import MediaInfo
from scriba.router.engine import BackendInfo

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

def _make_media_info(duration=60.0, has_video=False):
    return MediaInfo(
        duration_seconds=duration,
        has_audio=True,
        has_video=has_video,
        file_size_bytes=1_000_000,
        sample_rate=16000,
        channels=1,
        codec="pcm_s16le",
    )


def _make_routing_decision(backend="mlx_whisper", model="medium", trade_offs=None):
    selected = Estimate(
        backend=backend,
        model=model,
        time_seconds=30.0,
        cost_cents=0.0,
        available=True,
        recommended=True,
    )
    alternatives = [
        Estimate(
            backend="openai_stt",
            model="gpt-4o-mini-transcribe",
            time_seconds=10.0,
            cost_cents=5.0,
            available=True,
            recommended=False,
        )
    ]
    return RoutingDecision(
        selected=selected,
        alternatives=alternatives,
        trade_offs=trade_offs,
    )


def _make_backend_info(name="mlx_whisper", available=True):
    return BackendInfo(
        name=name,
        available=available,
        models=["tiny", "medium", "large-v3"],
        supports_diarize=False,
    )


def _make_transcription_result():
    return TranscriptionResult(
        text="Hello world.",
        segments=[Segment(start=0.0, end=2.0, text="Hello world.", speaker=None)],
        duration_seconds=60.0,
        model_used="medium",
        backend="mlx_whisper",
        cost_cents=0.0,
        diarized=False,
        routing=_make_routing_decision(),
    )


# ---------------------------------------------------------------------------
# Existing tests (kept as-is)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# transcribe --dry-run
# ---------------------------------------------------------------------------

def test_transcribe_dry_run_json_structure(tmp_path):
    """--dry-run returns JSON with selected/alternatives/trade_offs keys."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00" * 100)

    decision = _make_routing_decision(trade_offs=["Note: budget relaxed"])
    backend_info = _make_backend_info()

    runner = CliRunner()
    with patch("scriba.cli.probe_media", return_value=_make_media_info()), \
         patch("scriba.cli.discover_backends", return_value=[MagicMock(name="mlx_whisper", is_available=lambda: True)]), \
         patch("scriba.cli.backend_to_info", return_value=backend_info), \
         patch("scriba.cli.route", return_value=decision), \
         patch("scriba.cli.SecretsChain"):

        result = runner.invoke(main, ["transcribe", str(audio), "--dry-run"])

    assert result.exit_code == 0, result.output
    import json
    data = json.loads(result.stdout)
    assert "selected" in data
    assert "alternatives" in data
    assert "trade_offs" in data
    assert data["selected"]["backend"] == "mlx_whisper"
    assert data["selected"]["model"] == "medium"
    assert data["selected"]["time_seconds"] == 30.0
    assert data["selected"]["cost_cents"] == 0.0
    assert isinstance(data["alternatives"], list)
    assert data["alternatives"][0]["backend"] == "openai_stt"


def test_transcribe_dry_run_no_trade_offs(tmp_path):
    """--dry-run with no trade_offs returns null for trade_offs."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00" * 100)

    decision = _make_routing_decision(trade_offs=None)
    backend_info = _make_backend_info()

    runner = CliRunner()
    with patch("scriba.cli.probe_media", return_value=_make_media_info()), \
         patch("scriba.cli.discover_backends", return_value=[MagicMock(name="mlx_whisper", is_available=lambda: True)]), \
         patch("scriba.cli.backend_to_info", return_value=backend_info), \
         patch("scriba.cli.route", return_value=decision), \
         patch("scriba.cli.SecretsChain"):

        result = runner.invoke(main, ["transcribe", str(audio), "--dry-run"])

    assert result.exit_code == 0
    import json
    data = json.loads(result.output)
    assert data["trade_offs"] is None


def test_transcribe_dry_run_trade_offs_in_stderr(tmp_path):
    """Trade-off notes are written to stderr, not stdout."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00" * 100)

    decision = _make_routing_decision(trade_offs=["diarize not available, falling back"])
    backend_info = _make_backend_info()

    runner = CliRunner()
    with patch("scriba.cli.probe_media", return_value=_make_media_info()), \
         patch("scriba.cli.discover_backends", return_value=[MagicMock(name="mlx_whisper", is_available=lambda: True)]), \
         patch("scriba.cli.backend_to_info", return_value=backend_info), \
         patch("scriba.cli.route", return_value=decision), \
         patch("scriba.cli.SecretsChain"):

        result = runner.invoke(main, ["transcribe", str(audio), "--dry-run"])

    assert result.exit_code == 0
    assert "diarize not available" in result.stderr


# ---------------------------------------------------------------------------
# transcribe with output file
# ---------------------------------------------------------------------------

def test_transcribe_writes_output_file(tmp_path):
    """With -o, transcription text is written to the file and a message goes to stderr."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00" * 100)
    out_file = tmp_path / "result.json"

    decision = _make_routing_decision()
    backend_info = _make_backend_info()
    tr = _make_transcription_result()

    mock_adapter = MagicMock()
    mock_adapter.name = "mlx_whisper"
    mock_adapter.transcribe = AsyncMock(return_value=tr)

    runner = CliRunner()
    with patch("scriba.cli.probe_media", return_value=_make_media_info()), \
         patch("scriba.cli.discover_backends", return_value=[mock_adapter]), \
         patch("scriba.cli.backend_to_info", return_value=backend_info), \
         patch("scriba.cli.route", return_value=decision), \
         patch("scriba.cli.SecretsChain"), \
         patch("scriba.cli.extract_audio", return_value=audio), \
         patch("scriba.cli.format_result", return_value='{"text":"Hello world."}'), \
         patch("scriba.cli.load_config", return_value=MagicMock(calibration_path="~/.scriba_cal.json")), \
         patch("scriba.router.cost_model.save_calibration_entry"):

        result = runner.invoke(main, ["transcribe", str(audio), "-o", str(out_file)])

    assert result.exit_code == 0, result.output
    assert out_file.exists()
    assert out_file.read_text() == '{"text":"Hello world."}'
    assert f"Written to {out_file}" in result.output


def test_transcribe_stdout_when_no_output_file(tmp_path):
    """Without -o, transcription text goes to stdout."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00" * 100)

    decision = _make_routing_decision()
    backend_info = _make_backend_info()
    tr = _make_transcription_result()

    mock_adapter = MagicMock()
    mock_adapter.name = "mlx_whisper"
    mock_adapter.transcribe = AsyncMock(return_value=tr)

    runner = CliRunner()
    with patch("scriba.cli.probe_media", return_value=_make_media_info()), \
         patch("scriba.cli.discover_backends", return_value=[mock_adapter]), \
         patch("scriba.cli.backend_to_info", return_value=backend_info), \
         patch("scriba.cli.route", return_value=decision), \
         patch("scriba.cli.SecretsChain"), \
         patch("scriba.cli.extract_audio", return_value=audio), \
         patch("scriba.cli.format_result", return_value="Hello world."), \
         patch("scriba.cli.load_config", return_value=MagicMock(calibration_path="~/.scriba_cal.json")), \
         patch("scriba.router.cost_model.save_calibration_entry"):

        result = runner.invoke(main, ["transcribe", str(audio)])

    assert result.exit_code == 0
    assert "Hello world." in result.output


# ---------------------------------------------------------------------------
# estimate command
# ---------------------------------------------------------------------------

def test_estimate_json_structure(tmp_path):
    """estimate outputs valid JSON with selected/alternatives/trade_offs."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00" * 100)

    decision = _make_routing_decision()
    backend_info = _make_backend_info()

    runner = CliRunner()
    with patch("scriba.cli.probe_media", return_value=_make_media_info(duration=120.0)), \
         patch("scriba.cli.discover_backends", return_value=[MagicMock(name="mlx_whisper", is_available=lambda: True)]), \
         patch("scriba.cli.backend_to_info", return_value=backend_info), \
         patch("scriba.cli.route", return_value=decision), \
         patch("scriba.cli.SecretsChain"):

        result = runner.invoke(main, ["estimate", str(audio)])

    assert result.exit_code == 0, result.output
    import json
    data = json.loads(result.output)
    assert "selected" in data
    assert "alternatives" in data
    assert "trade_offs" in data

    sel = data["selected"]
    assert sel["backend"] == "mlx_whisper"
    assert sel["model"] == "medium"
    assert "time_seconds" in sel
    assert "cost_cents" in sel


def test_estimate_alternatives_include_time_and_cost(tmp_path):
    """Each alternative in estimate output includes time_seconds and cost_cents."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00" * 100)

    decision = _make_routing_decision()
    backend_info = _make_backend_info()

    runner = CliRunner()
    with patch("scriba.cli.probe_media", return_value=_make_media_info()), \
         patch("scriba.cli.discover_backends", return_value=[MagicMock(name="mlx_whisper", is_available=lambda: True)]), \
         patch("scriba.cli.backend_to_info", return_value=backend_info), \
         patch("scriba.cli.route", return_value=decision), \
         patch("scriba.cli.SecretsChain"):

        result = runner.invoke(main, ["estimate", str(audio)])

    assert result.exit_code == 0
    import json
    data = json.loads(result.output)
    for alt in data["alternatives"]:
        assert "backend" in alt
        assert "model" in alt
        assert "time_seconds" in alt
        assert "cost_cents" in alt


def test_estimate_with_quality_flag(tmp_path):
    """estimate passes quality flag through to route."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00" * 100)

    decision = _make_routing_decision(model="large-v3")
    backend_info = _make_backend_info()

    runner = CliRunner()
    with patch("scriba.cli.probe_media", return_value=_make_media_info()), \
         patch("scriba.cli.discover_backends", return_value=[MagicMock(name="mlx_whisper", is_available=lambda: True)]), \
         patch("scriba.cli.backend_to_info", return_value=backend_info), \
         patch("scriba.cli.route", return_value=decision) as mock_route, \
         patch("scriba.cli.SecretsChain"):

        result = runner.invoke(main, ["estimate", str(audio), "--quality", "high"])

    assert result.exit_code == 0
    called_constraints = mock_route.call_args[0][0]
    assert called_constraints.quality == "high"


def test_estimate_with_diarize_flag(tmp_path):
    """estimate passes --diarize flag through to route constraints."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00" * 100)

    decision = _make_routing_decision()
    backend_info = _make_backend_info()

    runner = CliRunner()
    with patch("scriba.cli.probe_media", return_value=_make_media_info()), \
         patch("scriba.cli.discover_backends", return_value=[MagicMock(name="mlx_whisper", is_available=lambda: True)]), \
         patch("scriba.cli.backend_to_info", return_value=backend_info), \
         patch("scriba.cli.route", return_value=decision) as mock_route, \
         patch("scriba.cli.SecretsChain"):

        result = runner.invoke(main, ["estimate", str(audio), "--diarize"])

    assert result.exit_code == 0
    called_constraints = mock_route.call_args[0][0]
    assert called_constraints.diarize is True


# ---------------------------------------------------------------------------
# backends command — extra tests
# ---------------------------------------------------------------------------

def test_backends_no_backends_available():
    """backends command still exits 0 even when no backends are installed."""
    runner = CliRunner()
    with patch("scriba.cli.discover_backends", return_value=[]):
        result = runner.invoke(main, ["backends"])
    assert result.exit_code == 0
    # Output should be empty or only whitespace
    assert result.output.strip() == ""


def test_backends_output_format():
    """Each backend line contains the backend name and its status."""
    mock_avail = MagicMock()
    mock_avail.name = "mlx_whisper"
    mock_avail.is_available = MagicMock(return_value=True)

    mock_unavail = MagicMock()
    mock_unavail.name = "whisperx"
    mock_unavail.is_available = MagicMock(return_value=False)

    runner = CliRunner()
    with patch("scriba.cli.discover_backends", return_value=[mock_avail, mock_unavail]):
        result = runner.invoke(main, ["backends"])

    assert result.exit_code == 0
    assert "mlx_whisper" in result.output
    assert "available" in result.output
    assert "whisperx" in result.output
    assert "not installed" in result.output


def test_backends_single_available():
    """A single available backend is listed with 'available' status."""
    mock_b = MagicMock()
    mock_b.name = "openai_stt"
    mock_b.is_available = MagicMock(return_value=True)

    runner = CliRunner()
    with patch("scriba.cli.discover_backends", return_value=[mock_b]):
        result = runner.invoke(main, ["backends"])

    assert result.exit_code == 0
    assert "openai_stt" in result.output
    assert "available" in result.output
    assert "not installed" not in result.output


# ---------------------------------------------------------------------------
# configure command
# ---------------------------------------------------------------------------

def test_configure_skip_both_keys():
    """Pressing enter for both prompts skips storage and still shows backend status."""
    mock_b = MagicMock()
    mock_b.name = "mlx_whisper"
    mock_b.is_available = MagicMock(return_value=True)

    runner = CliRunner()
    with patch("scriba.cli.discover_backends", return_value=[mock_b]):
        # Two empty enters: skip OpenAI key, skip HF token
        result = runner.invoke(main, ["configure"], input="\n\n")

    assert result.exit_code == 0
    assert "Scriba Configuration" in result.output
    assert "Backend status:" in result.output
    assert "mlx_whisper" in result.output


def test_configure_stores_openai_key():
    """Providing an OpenAI key calls keyring.set_password with correct args."""
    mock_b = MagicMock()
    mock_b.name = "mlx_whisper"
    mock_b.is_available = MagicMock(return_value=True)

    runner = CliRunner()
    with patch("scriba.cli.discover_backends", return_value=[mock_b]), \
         patch("keyring.set_password") as mock_set_pw, \
         patch("openai.OpenAI") as mock_openai:

        # Simulate validation failure so we don't need a real key
        mock_openai.return_value.models.list.side_effect = Exception("no network")

        result = runner.invoke(main, ["configure"], input="sk-test-key\n\n")

    assert result.exit_code == 0
    mock_set_pw.assert_any_call("scriba", "openai-api-key", "sk-test-key")
    assert "Stored in keychain." in result.output


def test_configure_stores_hf_token():
    """Providing an HF token calls keyring.set_password for hf-token."""
    mock_b = MagicMock()
    mock_b.name = "mlx_whisper"
    mock_b.is_available = MagicMock(return_value=True)

    runner = CliRunner()
    with patch("scriba.cli.discover_backends", return_value=[mock_b]), \
         patch("keyring.set_password") as mock_set_pw:

        # Empty OpenAI key, then provide HF token
        result = runner.invoke(main, ["configure"], input="\nhf-my-token\n")

    assert result.exit_code == 0
    mock_set_pw.assert_any_call("scriba", "hf-token", "hf-my-token")


def test_configure_openai_validation_success():
    """When OpenAI validation succeeds, 'Validated successfully.' is printed."""
    mock_b = MagicMock()
    mock_b.name = "mlx_whisper"
    mock_b.is_available = MagicMock(return_value=True)

    runner = CliRunner()
    with patch("scriba.cli.discover_backends", return_value=[mock_b]), \
         patch("keyring.set_password"), \
         patch("openai.OpenAI") as mock_openai:

        mock_openai.return_value.models.list.return_value = []

        result = runner.invoke(main, ["configure"], input="sk-valid-key\n\n")

    assert result.exit_code == 0
    assert "Validated successfully." in result.output


def test_configure_openai_validation_failure_warns():
    """When OpenAI validation fails, a warning is printed but key is still stored."""
    mock_b = MagicMock()
    mock_b.name = "mlx_whisper"
    mock_b.is_available = MagicMock(return_value=True)

    runner = CliRunner()
    with patch("scriba.cli.discover_backends", return_value=[mock_b]), \
         patch("keyring.set_password") as mock_set_pw, \
         patch("openai.OpenAI") as mock_openai:

        mock_openai.return_value.models.list.side_effect = Exception("auth error")

        result = runner.invoke(main, ["configure"], input="sk-bad-key\n\n")

    assert result.exit_code == 0
    mock_set_pw.assert_any_call("scriba", "openai-api-key", "sk-bad-key")
    assert "Warning: validation failed" in result.output


def test_configure_shows_all_backend_statuses():
    """configure shows status for every discovered backend at the end."""
    backends = []
    for name, avail in [("mlx_whisper", True), ("whisperx", False), ("openai_stt", True)]:
        m = MagicMock()
        m.name = name
        m.is_available = MagicMock(return_value=avail)
        backends.append(m)

    runner = CliRunner()
    with patch("scriba.cli.discover_backends", return_value=backends):
        result = runner.invoke(main, ["configure"], input="\n\n")

    assert result.exit_code == 0
    assert "mlx_whisper" in result.output
    assert "whisperx" in result.output
    assert "openai_stt" in result.output


# ---------------------------------------------------------------------------
# ScribaError propagation
# ---------------------------------------------------------------------------

def test_transcribe_scriba_error_exits_nonzero(tmp_path):
    """ScribaError during transcription causes non-zero exit and prints to stderr."""
    from scriba.errors import ScribaError

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00" * 100)

    runner = CliRunner()
    with patch("scriba.cli.probe_media", side_effect=ScribaError("ffprobe not found")):
        result = runner.invoke(main, ["transcribe", str(audio)])

    assert result.exit_code != 0
    assert "ffprobe not found" in result.output


def test_estimate_scriba_error_exits_nonzero(tmp_path):
    """ScribaError during estimate causes non-zero exit and prints to stderr."""
    from scriba.errors import ScribaError

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00" * 100)

    runner = CliRunner()
    with patch("scriba.cli.probe_media", side_effect=ScribaError("no audio stream")):
        result = runner.invoke(main, ["estimate", str(audio)])

    assert result.exit_code != 0
    assert "no audio stream" in result.output
