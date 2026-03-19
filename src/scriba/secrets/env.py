"""Environment variable secrets provider."""
from __future__ import annotations

import os
from pathlib import Path


class EnvProvider:
    def __init__(self, prefix: str = "SCRIBA_", dotenv_path: Path | None = None):
        self._prefix = prefix
        self._dotenv_path = dotenv_path or Path.cwd() / ".env"
        self._dotenv_cache: dict[str, str] | None = None

    def _env_key(self, key: str) -> str:
        return self._prefix + key.upper().replace("-", "_")

    def _load_dotenv(self) -> dict[str, str]:
        if self._dotenv_cache is not None:
            return self._dotenv_cache
        self._dotenv_cache = {}
        if self._dotenv_path.exists():
            for line in self._dotenv_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, _, v = line.partition("=")
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    self._dotenv_cache[k] = v
        return self._dotenv_cache

    async def get(self, key: str) -> str | None:
        env_key = self._env_key(key)
        # Check os.environ first
        val = os.environ.get(env_key)
        if val is not None:
            return val
        # Fall back to .env file
        return self._load_dotenv().get(env_key)

    async def set(self, key: str, value: str) -> None:
        os.environ[self._env_key(key)] = value

    async def delete(self, key: str) -> None:
        os.environ.pop(self._env_key(key), None)
