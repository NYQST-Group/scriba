"""Environment variable secrets provider."""
from __future__ import annotations

import os


class EnvProvider:
    def __init__(self, prefix: str = "SCRIBA_"):
        self._prefix = prefix

    def _env_key(self, key: str) -> str:
        return self._prefix + key.upper().replace("-", "_")

    async def get(self, key: str) -> str | None:
        return os.environ.get(self._env_key(key))

    async def set(self, key: str, value: str) -> None:
        os.environ[self._env_key(key)] = value

    async def delete(self, key: str) -> None:
        os.environ.pop(self._env_key(key), None)
