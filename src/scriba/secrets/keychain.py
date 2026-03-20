"""macOS Keychain (and cross-platform) secrets via keyring."""
from __future__ import annotations

import asyncio

import keyring


class KeychainProvider:
    def __init__(self, service: str = "scriba"):
        self._service = service

    async def get(self, key: str) -> str | None:
        return await asyncio.to_thread(keyring.get_password, self._service, key)

    async def set(self, key: str, value: str) -> None:
        await asyncio.to_thread(keyring.set_password, self._service, key, value)

    async def delete(self, key: str) -> None:
        try:
            await asyncio.to_thread(keyring.delete_password, self._service, key)
        except keyring.errors.PasswordDeleteError:
            pass
