"""OpenAI STT backend adapter with retry logic."""
from __future__ import annotations
import tempfile
from pathlib import Path
from typing import Any
from scriba.contracts import Estimate, Segment, TranscriptionConfig, TranscriptionResult
from scriba.errors import BackendError
from scriba.router.cost_model import estimate_cost_cents, estimate_time_seconds

DIARIZE_MODELS = {"gpt-4o-mini-transcribe", "gpt-4o-transcribe"}
MAX_UPLOAD_BYTES = 24 * 1024 * 1024


def _import_openai() -> Any | None:
    try:
        import openai
        return openai
    except ImportError:
        return None


class OpenAISTTBackend:
    name = "openai_stt"
    models = ["whisper-1", "gpt-4o-mini-transcribe", "gpt-4o-transcribe"]
    supports_diarize = True
    diarize_models = DIARIZE_MODELS

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key

    def set_api_key(self, key: str) -> None:
        """Set the API key for this backend."""
        self._api_key = key

    def is_available(self) -> bool:
        return _import_openai() is not None and self._api_key is not None

    def _make_client(self) -> Any:
        openai = _import_openai()
        return openai.AsyncOpenAI(api_key=self._api_key, timeout=180)

    def estimate(self, duration_seconds: float, config: TranscriptionConfig) -> Estimate:
        return Estimate(
            backend=self.name, model=config.model,
            time_seconds=estimate_time_seconds(self.name, config.model, duration_seconds=duration_seconds),
            cost_cents=estimate_cost_cents(self.name, config.model, duration_minutes=duration_seconds / 60),
            available=self.is_available(), recommended=False,
        )

    async def _call_api(self, client: Any, **kwargs: Any) -> Any:
        """Call OpenAI API with retry logic."""
        file_handle = kwargs.get("file")
        try:
            from tenacity import retry, stop_after_attempt, wait_exponential_jitter
            @retry(wait=wait_exponential_jitter(initial=1, max=20), stop=stop_after_attempt(3))
            async def _inner():
                if file_handle and hasattr(file_handle, "seek"):
                    file_handle.seek(0)
                return await client.audio.transcriptions.create(**kwargs)
            return await _inner()
        except ImportError:
            return await client.audio.transcriptions.create(**kwargs)

    async def transcribe(self, audio_path: Path, config: TranscriptionConfig) -> TranscriptionResult:
        openai_mod = _import_openai()
        if openai_mod is None:
            from scriba.errors import DependencyMissing
            raise DependencyMissing("openai", hint='pip install "nyqst-scriba[openai]"')

        prepared_path = audio_path
        tmp_dir = None
        try:
            if audio_path.stat().st_size > MAX_UPLOAD_BYTES:
                from scriba.media.ingest import compress_for_upload
                tmp_dir = tempfile.mkdtemp(prefix="scriba-")
                prepared_path = compress_for_upload(audio_path, Path(tmp_dir) / f"{audio_path.stem}.mp3")

            use_diarize = config.diarize and config.model in DIARIZE_MODELS
            response_format = "diarized_json" if use_diarize else "verbose_json"

            client = self._make_client()
            kwargs: dict[str, Any] = {
                "model": config.model,
                "file": prepared_path.open("rb"),
                "response_format": response_format,
            }
            if config.language:
                kwargs["language"] = config.language
            if config.model != "whisper-1":
                kwargs["chunking_strategy"] = "auto"

            try:
                resp = await self._call_api(client, **kwargs)
            except Exception as e:
                raise BackendError(self.name, cause=e, suggestion="check API key and model access")
            finally:
                kwargs["file"].close()
        finally:
            if tmp_dir:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

        raw = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
        segments = [
            Segment(
                start=s.get("start", 0.0), end=s.get("end", 0.0),
                text=s.get("text", "").strip(),
                speaker=s.get("speaker") if use_diarize else None,
            )
            for s in raw.get("segments", [])
            if isinstance(s, dict) and s.get("text", "").strip()
        ]
        return TranscriptionResult(
            text=raw.get("text", "").strip(), segments=segments,
            duration_seconds=float(raw.get("duration", 0)),
            model_used=config.model, backend=self.name,
            cost_cents=estimate_cost_cents(self.name, config.model, duration_minutes=float(raw.get("duration", 0)) / 60),
            diarized=use_diarize,
        )
