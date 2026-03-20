from unittest.mock import patch, AsyncMock, MagicMock
import json

import pytest

from scriba.mcp.server import create_server, handle_backends
from scriba.contracts import (
    Estimate,
    RoutingDecision,
    Segment,
    TranscriptionResult,
)
from scriba.media.ingest import MediaInfo
from scriba.errors import AudioError, BackendError, RoutingError


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_media_info(duration: float = 120.0, has_video: bool = False) -> MagicMock:
    info = MagicMock(spec=MediaInfo)
    info.duration_seconds = duration
    info.has_audio = True
    info.has_video = has_video
    info.file_size_bytes = 1_000_000
    return info


def _make_fake_backend(name: str = "mlx_whisper", available: bool = True) -> MagicMock:
    backend = MagicMock()
    backend.name = name
    backend.models = ["tiny", "medium", "large-v3"]
    backend.supports_diarize = False
    backend.diarize_models = set()
    backend.is_available.return_value = available
    return backend


def _make_routing_decision(
    backend: str = "mlx_whisper",
    model: str = "medium",
    cost: float = 0.0,
    time: float = 30.0,
    alternatives: list | None = None,
    trade_offs: list | None = None,
) -> RoutingDecision:
    selected = Estimate(
        backend=backend,
        model=model,
        time_seconds=time,
        cost_cents=cost,
        available=True,
        recommended=True,
    )
    return RoutingDecision(
        selected=selected,
        alternatives=alternatives or [],
        trade_offs=trade_offs,
    )


def _make_transcription_result(
    backend: str = "mlx_whisper",
    model: str = "medium",
    decision: RoutingDecision | None = None,
) -> TranscriptionResult:
    seg = Segment(start=0.0, end=5.0, text="Hello world.")
    return TranscriptionResult(
        text="Hello world.",
        segments=[seg],
        duration_seconds=5.0,
        model_used=model,
        backend=backend,
        cost_cents=0.0,
        diarized=False,
        enrichment_available=False,
        routing=decision,
    )


# ---------------------------------------------------------------------------
# Existing tests (kept intact)
# ---------------------------------------------------------------------------


def test_server_has_tools():
    server = create_server()
    tool_names = [t.name for t in server._tool_manager.list_tools()]
    assert "transcribe" in tool_names
    assert "estimate" in tool_names
    assert "backends" in tool_names


@pytest.mark.asyncio
async def test_backends_tool():
    result = await handle_backends()
    assert isinstance(result, list)
    for entry in result:
        assert "name" in entry
        assert "available" in entry


# ---------------------------------------------------------------------------
# transcribe tool tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transcribe_returns_json_with_text():
    """Happy-path: mocked local backend returns a result; tool returns valid JSON."""
    from scriba.mcp.server import transcribe

    fake_backend = _make_fake_backend("mlx_whisper")
    decision = _make_routing_decision("mlx_whisper", "medium")
    tr = _make_transcription_result(decision=decision)
    fake_backend.transcribe = AsyncMock(return_value=tr)

    with (
        patch("scriba.mcp.server.discover_backends", return_value=[fake_backend]),
        patch("scriba.mcp.server.probe_media", return_value=_make_media_info()),
        patch("scriba.mcp.server.extract_audio", return_value=MagicMock()),
        patch("scriba.mcp.server.route", return_value=decision),
        patch("scriba.mcp.server._get_secrets", new_callable=AsyncMock,
              return_value=MagicMock(get=AsyncMock(return_value=None))),
        patch("scriba.mcp.server.save_calibration_entry", create=True),
        patch("scriba.router.cost_model.save_calibration_entry"),
    ):
        result = await transcribe(file_path="/fake/audio.wav")

    data = json.loads(result)
    assert data["text"] == "Hello world."
    assert data["backend"] == "mlx_whisper"
    assert data["model"] == "medium"
    assert "duration_seconds" in data


@pytest.mark.asyncio
async def test_transcribe_local_backend_uses_local_semaphore():
    """Non-openai_stt backend should use _local_sem (Semaphore(1))."""
    from scriba.mcp.server import transcribe, _local_sem, _cloud_sem

    fake_backend = _make_fake_backend("mlx_whisper")
    decision = _make_routing_decision("mlx_whisper", "medium")
    tr = _make_transcription_result(decision=decision)
    fake_backend.transcribe = AsyncMock(return_value=tr)

    acquired_sems = []

    original_local_acquire = _local_sem.acquire
    original_cloud_acquire = _cloud_sem.acquire

    async def track_local():
        acquired_sems.append("local")
        return await original_local_acquire()

    async def track_cloud():
        acquired_sems.append("cloud")
        return await original_cloud_acquire()

    with (
        patch("scriba.mcp.server.discover_backends", return_value=[fake_backend]),
        patch("scriba.mcp.server.probe_media", return_value=_make_media_info()),
        patch("scriba.mcp.server.extract_audio", return_value=MagicMock()),
        patch("scriba.mcp.server.route", return_value=decision),
        patch("scriba.mcp.server._get_secrets", new_callable=AsyncMock,
              return_value=MagicMock(get=AsyncMock(return_value=None))),
        patch.object(_local_sem, "acquire", side_effect=track_local),
        patch.object(_cloud_sem, "acquire", side_effect=track_cloud),
        patch("scriba.router.cost_model.save_calibration_entry"),
    ):
        await transcribe(file_path="/fake/audio.wav")

    assert "local" in acquired_sems
    assert "cloud" not in acquired_sems


@pytest.mark.asyncio
async def test_transcribe_cloud_backend_uses_cloud_semaphore():
    """openai_stt backend should use _cloud_sem (Semaphore(3))."""
    from scriba.mcp.server import transcribe, _local_sem, _cloud_sem

    fake_backend = _make_fake_backend("openai_stt")
    decision = _make_routing_decision("openai_stt", "gpt-4o-mini-transcribe", cost=1.2, time=5.0)
    tr = _make_transcription_result("openai_stt", "gpt-4o-mini-transcribe", decision=decision)
    fake_backend.transcribe = AsyncMock(return_value=tr)

    acquired_sems = []

    original_local_acquire = _local_sem.acquire
    original_cloud_acquire = _cloud_sem.acquire

    async def track_local():
        acquired_sems.append("local")
        return await original_local_acquire()

    async def track_cloud():
        acquired_sems.append("cloud")
        return await original_cloud_acquire()

    with (
        patch("scriba.mcp.server.discover_backends", return_value=[fake_backend]),
        patch("scriba.mcp.server.probe_media", return_value=_make_media_info()),
        patch("scriba.mcp.server.extract_audio", return_value=MagicMock()),
        patch("scriba.mcp.server.route", return_value=decision),
        patch("scriba.mcp.server._get_secrets", new_callable=AsyncMock,
              return_value=MagicMock(get=AsyncMock(return_value=None))),
        patch.object(_local_sem, "acquire", side_effect=track_local),
        patch.object(_cloud_sem, "acquire", side_effect=track_cloud),
        patch("scriba.router.cost_model.save_calibration_entry"),
    ):
        await transcribe(file_path="/fake/audio.wav")

    assert "cloud" in acquired_sems
    assert "local" not in acquired_sems


@pytest.mark.asyncio
async def test_transcribe_enriched_tier_sets_flag():
    """output_tier=enriched should set enrichment_available on the result."""
    from scriba.mcp.server import transcribe

    fake_backend = _make_fake_backend("mlx_whisper")
    decision = _make_routing_decision()
    tr = _make_transcription_result(decision=decision)
    fake_backend.transcribe = AsyncMock(return_value=tr)

    with (
        patch("scriba.mcp.server.discover_backends", return_value=[fake_backend]),
        patch("scriba.mcp.server.probe_media", return_value=_make_media_info()),
        patch("scriba.mcp.server.extract_audio", return_value=MagicMock()),
        patch("scriba.mcp.server.route", return_value=decision),
        patch("scriba.mcp.server._get_secrets", new_callable=AsyncMock,
              return_value=MagicMock(get=AsyncMock(return_value=None))),
        patch("scriba.router.cost_model.save_calibration_entry"),
    ):
        result = await transcribe(file_path="/fake/audio.wav", output_tier="enriched")

    data = json.loads(result)
    assert data.get("enrichment_available") is True


@pytest.mark.asyncio
async def test_transcribe_text_format():
    """output_format=text returns a plain string (no JSON parsing needed)."""
    from scriba.mcp.server import transcribe

    fake_backend = _make_fake_backend("mlx_whisper")
    decision = _make_routing_decision()
    tr = _make_transcription_result(decision=decision)
    fake_backend.transcribe = AsyncMock(return_value=tr)

    with (
        patch("scriba.mcp.server.discover_backends", return_value=[fake_backend]),
        patch("scriba.mcp.server.probe_media", return_value=_make_media_info()),
        patch("scriba.mcp.server.extract_audio", return_value=MagicMock()),
        patch("scriba.mcp.server.route", return_value=decision),
        patch("scriba.mcp.server._get_secrets", new_callable=AsyncMock,
              return_value=MagicMock(get=AsyncMock(return_value=None))),
        patch("scriba.router.cost_model.save_calibration_entry"),
    ):
        result = await transcribe(file_path="/fake/audio.wav", output_format="text")

    # text format returns timestamped lines, not JSON
    assert isinstance(result, str)
    assert "Hello world." in result


# ---------------------------------------------------------------------------
# estimate tool tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_estimate_returns_expected_structure():
    """estimate returns JSON with selected/alternatives/trade_offs keys."""
    from scriba.mcp.server import estimate

    fake_backend = _make_fake_backend("mlx_whisper")
    alt = Estimate(
        backend="openai_stt", model="gpt-4o-mini-transcribe",
        time_seconds=5.0, cost_cents=1.5, available=True, recommended=False,
    )
    decision = _make_routing_decision(alternatives=[alt], trade_offs=None)

    with (
        patch("scriba.mcp.server.discover_backends", return_value=[fake_backend]),
        patch("scriba.mcp.server.probe_media", return_value=_make_media_info()),
        patch("scriba.mcp.server.route", return_value=decision),
        patch("scriba.mcp.server._get_secrets", new_callable=AsyncMock,
              return_value=MagicMock(get=AsyncMock(return_value=None))),
    ):
        result = await estimate(file_path="/fake/audio.wav")

    data = json.loads(result)
    assert "selected" in data
    assert "alternatives" in data
    assert "trade_offs" in data

    sel = data["selected"]
    assert sel["backend"] == "mlx_whisper"
    assert sel["model"] == "medium"
    assert "time_seconds" in sel
    assert "cost_cents" in sel

    assert len(data["alternatives"]) == 1
    assert data["alternatives"][0]["backend"] == "openai_stt"


@pytest.mark.asyncio
async def test_estimate_with_no_alternatives():
    """estimate works fine when there are no alternative backends."""
    from scriba.mcp.server import estimate

    fake_backend = _make_fake_backend("mlx_whisper")
    decision = _make_routing_decision(alternatives=[], trade_offs=None)

    with (
        patch("scriba.mcp.server.discover_backends", return_value=[fake_backend]),
        patch("scriba.mcp.server.probe_media", return_value=_make_media_info()),
        patch("scriba.mcp.server.route", return_value=decision),
        patch("scriba.mcp.server._get_secrets", new_callable=AsyncMock,
              return_value=MagicMock(get=AsyncMock(return_value=None))),
    ):
        result = await estimate(file_path="/fake/audio.wav")

    data = json.loads(result)
    assert data["alternatives"] == []
    assert data["trade_offs"] is None
    assert data["routing_notes"] is None


@pytest.mark.asyncio
async def test_estimate_with_trade_offs():
    """estimate populates trade_offs when the router provides them."""
    from scriba.mcp.server import estimate

    fake_backend = _make_fake_backend("mlx_whisper")
    trade_offs = ["Diarization not available on selected backend, disabled."]
    decision = _make_routing_decision(trade_offs=trade_offs)

    with (
        patch("scriba.mcp.server.discover_backends", return_value=[fake_backend]),
        patch("scriba.mcp.server.probe_media", return_value=_make_media_info()),
        patch("scriba.mcp.server.route", return_value=decision),
        patch("scriba.mcp.server._get_secrets", new_callable=AsyncMock,
              return_value=MagicMock(get=AsyncMock(return_value=None))),
    ):
        result = await estimate(file_path="/fake/audio.wav")

    data = json.loads(result)
    assert data["trade_offs"] == trade_offs
    assert data["routing_notes"] == trade_offs


@pytest.mark.asyncio
async def test_estimate_passes_constraints_to_router():
    """estimate passes quality, budget, timeout, diarize to route()."""
    from scriba.mcp.server import estimate

    fake_backend = _make_fake_backend("mlx_whisper")
    decision = _make_routing_decision()
    mock_route = MagicMock(return_value=decision)

    with (
        patch("scriba.mcp.server.discover_backends", return_value=[fake_backend]),
        patch("scriba.mcp.server.probe_media", return_value=_make_media_info(duration=300.0)),
        patch("scriba.mcp.server.route", mock_route),
        patch("scriba.mcp.server._get_secrets", new_callable=AsyncMock,
              return_value=MagicMock(get=AsyncMock(return_value=None))),
    ):
        await estimate(
            file_path="/fake/audio.wav",
            quality="high",
            budget_cents=50,
            timeout_seconds=120,
            diarize=True,
        )

    call_kwargs = mock_route.call_args
    constraints = call_kwargs.args[0]
    assert constraints.quality == "high"
    assert constraints.budget_cents == 50
    assert constraints.timeout_seconds == 120
    assert constraints.diarize is True
    assert call_kwargs.kwargs["duration_seconds"] == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transcribe_scriba_error_returns_json_error():
    """ScribaError during transcription is caught and returned as JSON."""
    from scriba.mcp.server import transcribe

    with (
        patch("scriba.mcp.server.probe_media",
              side_effect=AudioError("File not found: /fake/missing.wav")),
        patch("scriba.mcp.server._get_secrets", new_callable=AsyncMock,
              return_value=MagicMock(get=AsyncMock(return_value=None))),
    ):
        result = await transcribe(file_path="/fake/missing.wav")

    data = json.loads(result)
    assert "error" in data
    assert "type" in data
    assert data["type"] == "AudioError"
    assert "File not found" in data["error"]


@pytest.mark.asyncio
async def test_transcribe_routing_error_returns_json_error():
    """RoutingError (no backends) is caught and returned as JSON."""
    from scriba.mcp.server import transcribe

    with (
        patch("scriba.mcp.server.probe_media", return_value=_make_media_info()),
        patch("scriba.mcp.server.extract_audio", return_value=MagicMock()),
        patch("scriba.mcp.server.discover_backends", return_value=[]),
        patch("scriba.mcp.server.route",
              side_effect=RoutingError("No backends available")),
        patch("scriba.mcp.server._get_secrets", new_callable=AsyncMock,
              return_value=MagicMock(get=AsyncMock(return_value=None))),
    ):
        result = await transcribe(file_path="/fake/audio.wav")

    data = json.loads(result)
    assert data["type"] == "RoutingError"
    assert "No backends available" in data["error"]


@pytest.mark.asyncio
async def test_transcribe_backend_error_returns_json_error():
    """BackendError raised during transcription is caught and returned as JSON."""
    from scriba.mcp.server import transcribe

    fake_backend = _make_fake_backend("mlx_whisper")
    fake_backend.transcribe = AsyncMock(
        side_effect=BackendError("mlx_whisper", cause=RuntimeError("OOM"))
    )
    decision = _make_routing_decision()

    with (
        patch("scriba.mcp.server.discover_backends", return_value=[fake_backend]),
        patch("scriba.mcp.server.probe_media", return_value=_make_media_info()),
        patch("scriba.mcp.server.extract_audio", return_value=MagicMock()),
        patch("scriba.mcp.server.route", return_value=decision),
        patch("scriba.mcp.server._get_secrets", new_callable=AsyncMock,
              return_value=MagicMock(get=AsyncMock(return_value=None))),
    ):
        result = await transcribe(file_path="/fake/audio.wav")

    data = json.loads(result)
    assert data["type"] == "BackendError"
    assert "mlx_whisper" in data["error"]


@pytest.mark.asyncio
async def test_estimate_scriba_error_returns_json_error():
    """ScribaError during estimate is caught and returned as JSON."""
    from scriba.mcp.server import estimate

    with (
        patch("scriba.mcp.server.probe_media",
              side_effect=AudioError("File not found: /fake/missing.wav")),
        patch("scriba.mcp.server._get_secrets", new_callable=AsyncMock,
              return_value=MagicMock(get=AsyncMock(return_value=None))),
    ):
        result = await estimate(file_path="/fake/missing.wav")

    data = json.loads(result)
    assert "error" in data
    assert data["type"] == "AudioError"


@pytest.mark.asyncio
async def test_estimate_routing_error_returns_json_error():
    """RoutingError during estimate is caught and returned as JSON."""
    from scriba.mcp.server import estimate

    with (
        patch("scriba.mcp.server.probe_media", return_value=_make_media_info()),
        patch("scriba.mcp.server.discover_backends", return_value=[]),
        patch("scriba.mcp.server.route",
              side_effect=RoutingError("No backends available")),
        patch("scriba.mcp.server._get_secrets", new_callable=AsyncMock,
              return_value=MagicMock(get=AsyncMock(return_value=None))),
    ):
        result = await estimate(file_path="/fake/audio.wav")

    data = json.loads(result)
    assert data["type"] == "RoutingError"


# ---------------------------------------------------------------------------
# Secrets integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transcribe_wires_openai_api_key_to_backend():
    """When an openai_stt backend exists, its _api_key is set from secrets."""
    from scriba.mcp.server import transcribe

    openai_backend = _make_fake_backend("openai_stt")
    decision = _make_routing_decision("openai_stt", "gpt-4o-mini-transcribe")
    tr = _make_transcription_result("openai_stt", "gpt-4o-mini-transcribe", decision=decision)
    openai_backend.transcribe = AsyncMock(return_value=tr)

    fake_secrets = MagicMock()
    fake_secrets.get = AsyncMock(return_value="sk-test-1234")

    with (
        patch("scriba.mcp.server.discover_backends", return_value=[openai_backend]),
        patch("scriba.mcp.server.probe_media", return_value=_make_media_info()),
        patch("scriba.mcp.server.extract_audio", return_value=MagicMock()),
        patch("scriba.mcp.server.route", return_value=decision),
        patch("scriba.mcp.server._get_secrets", new_callable=AsyncMock,
              return_value=fake_secrets),
        patch("scriba.router.cost_model.save_calibration_entry"),
    ):
        await transcribe(file_path="/fake/audio.wav")

    fake_secrets.get.assert_called_with("openai-api-key")
    openai_backend.set_api_key.assert_called_with("sk-test-1234")


@pytest.mark.asyncio
async def test_transcribe_does_not_set_api_key_when_secret_missing():
    """When secrets returns None, the backend _api_key is not modified."""
    from scriba.mcp.server import transcribe

    openai_backend = _make_fake_backend("openai_stt")
    decision = _make_routing_decision("openai_stt", "gpt-4o-mini-transcribe")
    tr = _make_transcription_result("openai_stt", "gpt-4o-mini-transcribe", decision=decision)
    openai_backend.transcribe = AsyncMock(return_value=tr)

    fake_secrets = MagicMock()
    fake_secrets.get = AsyncMock(return_value=None)

    with (
        patch("scriba.mcp.server.discover_backends", return_value=[openai_backend]),
        patch("scriba.mcp.server.probe_media", return_value=_make_media_info()),
        patch("scriba.mcp.server.extract_audio", return_value=MagicMock()),
        patch("scriba.mcp.server.route", return_value=decision),
        patch("scriba.mcp.server._get_secrets", new_callable=AsyncMock,
              return_value=fake_secrets),
        patch("scriba.router.cost_model.save_calibration_entry"),
    ):
        await transcribe(file_path="/fake/audio.wav")

    # set_api_key should NOT have been called when no key was available
    openai_backend.set_api_key.assert_not_called()


@pytest.mark.asyncio
async def test_estimate_wires_openai_api_key_to_backend():
    """estimate also wires the OpenAI key from secrets to the openai_stt backend."""
    from scriba.mcp.server import estimate

    openai_backend = _make_fake_backend("openai_stt")
    decision = _make_routing_decision("openai_stt", "gpt-4o-mini-transcribe")

    fake_secrets = MagicMock()
    fake_secrets.get = AsyncMock(return_value="sk-estimate-key")

    with (
        patch("scriba.mcp.server.discover_backends", return_value=[openai_backend]),
        patch("scriba.mcp.server.probe_media", return_value=_make_media_info()),
        patch("scriba.mcp.server.route", return_value=decision),
        patch("scriba.mcp.server._get_secrets", new_callable=AsyncMock,
              return_value=fake_secrets),
    ):
        await estimate(file_path="/fake/audio.wav")

    fake_secrets.get.assert_called_with("openai-api-key")
    openai_backend.set_api_key.assert_called_with("sk-estimate-key")


@pytest.mark.asyncio
async def test_transcribe_non_openai_backend_does_not_fetch_api_key():
    """For local-only backends, secrets.get should not be called with openai key."""
    from scriba.mcp.server import transcribe

    local_backend = _make_fake_backend("mlx_whisper")
    decision = _make_routing_decision("mlx_whisper", "medium")
    tr = _make_transcription_result(decision=decision)
    local_backend.transcribe = AsyncMock(return_value=tr)

    fake_secrets = MagicMock()
    fake_secrets.get = AsyncMock(return_value=None)

    with (
        patch("scriba.mcp.server.discover_backends", return_value=[local_backend]),
        patch("scriba.mcp.server.probe_media", return_value=_make_media_info()),
        patch("scriba.mcp.server.extract_audio", return_value=MagicMock()),
        patch("scriba.mcp.server.route", return_value=decision),
        patch("scriba.mcp.server._get_secrets", new_callable=AsyncMock,
              return_value=fake_secrets),
        patch("scriba.router.cost_model.save_calibration_entry"),
    ):
        result = await transcribe(file_path="/fake/audio.wav")

    # secrets.get should never be called when there's no openai_stt backend
    fake_secrets.get.assert_not_called()
    data = json.loads(result)
    assert data["backend"] == "mlx_whisper"


# ---------------------------------------------------------------------------
# handle_backends / backends tool tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_backends_structure():
    """handle_backends returns list with required keys per entry."""
    result = await handle_backends()
    assert isinstance(result, list)
    for entry in result:
        assert "name" in entry
        assert "available" in entry
        assert "models" in entry
        assert "supports_diarize" in entry
        assert "reason_unavailable" in entry


@pytest.mark.asyncio
async def test_handle_backends_unavailable_has_reason():
    """Unavailable backends populate reason_unavailable."""
    unavailable = _make_fake_backend("whisperx", available=False)

    with patch("scriba.mcp.server.discover_backends", return_value=[unavailable]):
        result = await handle_backends()

    assert len(result) == 1
    entry = result[0]
    assert entry["available"] is False
    assert entry["reason_unavailable"] is not None
    assert "whisperx" in entry["reason_unavailable"]


@pytest.mark.asyncio
async def test_handle_backends_available_has_no_reason():
    """Available backends have reason_unavailable=None."""
    available = _make_fake_backend("mlx_whisper", available=True)

    with patch("scriba.mcp.server.discover_backends", return_value=[available]):
        result = await handle_backends()

    assert len(result) == 1
    assert result[0]["available"] is True
    assert result[0]["reason_unavailable"] is None
