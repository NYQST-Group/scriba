"""Secrets provider protocol and chain."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SecretsProvider(Protocol):
    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str) -> None: ...
    async def delete(self, key: str) -> None: ...


class SecretsChain:
    """Try providers in order, return first non-None result."""

    def __init__(self, providers: list[SecretsProvider]):
        self._providers = providers

    async def get(self, key: str) -> str | None:
        for provider in self._providers:
            value = await provider.get(key)
            if value is not None:
                return value
        return None

    async def set(self, key: str, value: str) -> None:
        if self._providers:
            await self._providers[0].set(key, value)

    async def delete(self, key: str) -> None:
        if self._providers:
            await self._providers[0].delete(key)
