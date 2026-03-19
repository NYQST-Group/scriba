from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from scriba.mcp.server import create_server, handle_backends


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
