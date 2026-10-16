import asyncio
from pathlib import Path

from teo_ai import db, knowledge

KNOWLEDGE_FOLDER = Path("knowledge")


async def run() -> None:
    db.create_schema()
    db.open_pool()

    try:
        for line in await knowledge.ingest(KNOWLEDGE_FOLDER):
            print(line)
    finally:
        db.close_pool()


def main() -> None:
    asyncio.run(run())
