import os
from unittest.mock import AsyncMock, patch

import pytest

from scriba.secrets.provider import SecretsChain
from scriba.secrets.env import EnvProvider
from scriba.secrets.keychain import KeychainProvider


@pytest.mark.asyncio
async def test_env_provider_reads_env_var():
    prov = EnvProvider(prefix="SCRIBA_")
    with patch.dict(os.environ, {"SCRIBA_OPENAI_API_KEY": "sk-test"}):
        val = await prov.get("openai-api-key")
    assert val == "sk-test"


@pytest.mark.asyncio
async def test_env_provider_returns_none_if_missing():
    prov = EnvProvider(prefix="SCRIBA_")
    with patch.dict(os.environ, {}, clear=True):
        val = await prov.get("openai-api-key")
    assert val is None


@pytest.mark.asyncio
async def test_keychain_provider_delegates_to_keyring():
    with patch("scriba.secrets.keychain.keyring") as mock_kr:
        mock_kr.get_password.return_value = "sk-from-keychain"
        prov = KeychainProvider(service="scriba")
        val = await prov.get("openai-api-key")
    assert val == "sk-from-keychain"
    mock_kr.get_password.assert_called_once_with("scriba", "openai-api-key")


@pytest.mark.asyncio
async def test_keychain_provider_returns_none():
    with patch("scriba.secrets.keychain.keyring") as mock_kr:
        mock_kr.get_password.return_value = None
        prov = KeychainProvider(service="scriba")
        val = await prov.get("openai-api-key")
    assert val is None


@pytest.mark.asyncio
async def test_chain_tries_in_order():
    kc = AsyncMock()
    kc.get.return_value = None
    env = AsyncMock()
    env.get.return_value = "sk-from-env"
    chain = SecretsChain([kc, env])
    val = await chain.get("openai-api-key")
    assert val == "sk-from-env"
    kc.get.assert_called_once_with("openai-api-key")
    env.get.assert_called_once_with("openai-api-key")


@pytest.mark.asyncio
async def test_chain_stops_on_first_hit():
    kc = AsyncMock()
    kc.get.return_value = "sk-from-keychain"
    env = AsyncMock()
    chain = SecretsChain([kc, env])
    val = await chain.get("openai-api-key")
    assert val == "sk-from-keychain"
    env.get.assert_not_called()


@pytest.mark.asyncio
async def test_chain_returns_none_if_all_miss():
    kc = AsyncMock()
    kc.get.return_value = None
    env = AsyncMock()
    env.get.return_value = None
    chain = SecretsChain([kc, env])
    val = await chain.get("openai-api-key")
    assert val is None


@pytest.mark.asyncio
async def test_keychain_uses_to_thread():
    """Keychain get uses asyncio.to_thread to avoid blocking."""
    with patch("scriba.secrets.keychain.asyncio") as mock_asyncio:
        mock_asyncio.to_thread = AsyncMock(return_value="secret")
        provider = KeychainProvider()
        result = await provider.get("test-key")
        mock_asyncio.to_thread.assert_called_once()
        assert result == "secret"
