from contextlib import asynccontextmanager
from typing import AsyncIterator

from mcp import ClientSession
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

from teo_ai.config import settings


@asynccontextmanager
async def open_session(token: str) -> AsyncIterator[ClientSession]:
    http_client = create_mcp_http_client(headers={"Authorization": f"Bearer {token}"})

    async with http_client:
        async with streamable_http_client(settings.mcp_url, http_client=http_client) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session


def result_text(result) -> str:
    return "\n".join(item.text for item in result.content if getattr(item, "type", None) == "text")
