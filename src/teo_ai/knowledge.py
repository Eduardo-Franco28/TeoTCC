import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pypdf import PdfReader

from teo_ai import db, llm
from teo_ai.config import settings

CHUNK_WORDS = 600
CHUNK_OVERLAP = 80
EMBED_BATCH = 16
TEXT_EXTENSIONS = {".md", ".txt"}


@dataclass
class Chunk:
    page: int | None
    content: str


@dataclass
class Source:
    title: str
    link: str | None
    page: int | None
    content: str
    score: float


def read_pages(path: Path) -> list[tuple[int | None, str]]:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        return [(number, page.extract_text() or "") for number, page in enumerate(reader.pages, start=1)]

    if path.suffix.lower() in TEXT_EXTENSIONS:
        return [(None, path.read_text(encoding="utf-8"))]

    raise ValueError(f"Formato não suportado: {path.name}")


def split_chunks(pages: list[tuple[int | None, str]]) -> list[Chunk]:
    words: list[tuple[int | None, str]] = [
        (page, word) for page, text in pages for word in text.split()
    ]

    chunks: list[Chunk] = []
    start = 0

    while start < len(words):
        window = words[start:start + CHUNK_WORDS]
        chunks.append(Chunk(page=window[0][0], content=" ".join(word for _, word in window)))

        if start + CHUNK_WORDS >= len(words):
            break

        start += CHUNK_WORDS - CHUNK_OVERLAP

    return chunks


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def embed_chunks(chunks: list[Chunk]) -> list[list[float]]:
    vectors: list[list[float]] = []

    for start in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[start:start + EMBED_BATCH]
        vectors.extend(await llm.embed([chunk.content for chunk in batch], "passage"))

    return vectors


def load_sources(folder: Path) -> list[dict]:
    sources_file = folder / "fontes.json"

    if not sources_file.exists():
        return []

    return json.loads(sources_file.read_text(encoding="utf-8"))


def _current_hash(file: str) -> str | None:
    with db.pool().connection() as conn:
        row = conn.execute("SELECT content_hash FROM teo.documents WHERE file = %s", (file,)).fetchone()
    return row[0] if row else None


def _save_document(entry: dict, content_hash: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    with db.pool().connection() as conn:
        with conn.transaction():
            row = conn.execute(
                """
                INSERT INTO teo.documents (file, title, link, status, content_hash)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (file) DO UPDATE
                SET title = EXCLUDED.title, link = EXCLUDED.link, status = EXCLUDED.status,
                    content_hash = EXCLUDED.content_hash, ingested_at = now()
                RETURNING id
                """,
                (entry["arquivo"], entry["titulo"], entry.get("link"), entry.get("status", "ativo"), content_hash),
            ).fetchone()

            document_id = row[0]
            conn.execute("DELETE FROM teo.chunks WHERE document_id = %s", (document_id,))

            with conn.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO teo.chunks (document_id, position, page, content, embedding) VALUES (%s, %s, %s, %s, %s::halfvec)",
                    [
                        (document_id, position, chunk.page, chunk.content, np.array(vector, dtype=np.float32))
                        for position, (chunk, vector) in enumerate(zip(chunks, vectors))
                    ],
                )


def _update_metadata(entry: dict) -> None:
    with db.pool().connection() as conn:
        conn.execute(
            "UPDATE teo.documents SET title = %s, link = %s, status = %s WHERE file = %s",
            (entry["titulo"], entry.get("link"), entry.get("status", "ativo"), entry["arquivo"]),
        )


def _deactivate_missing(files: list[str]) -> int:
    with db.pool().connection() as conn:
        result = conn.execute(
            "UPDATE teo.documents SET status = 'inativo' WHERE status <> 'inativo' AND NOT (file = ANY(%s))",
            (files,),
        )
    return result.rowcount


async def ingest(folder: Path) -> list[str]:
    report: list[str] = []
    entries = load_sources(folder)

    for entry in entries:
        path = folder / entry["arquivo"]

        if not path.exists():
            report.append(f"{entry['arquivo']}: arquivo não encontrado")
            continue

        content_hash = file_hash(path)

        if await asyncio.to_thread(_current_hash, entry["arquivo"]) == content_hash:
            await asyncio.to_thread(_update_metadata, entry)
            report.append(f"{entry['arquivo']}: sem mudança")
            continue

        chunks = split_chunks(read_pages(path))

        if not chunks:
            report.append(f"{entry['arquivo']}: nenhum texto encontrado")
            continue

        vectors = await embed_chunks(chunks)
        await asyncio.to_thread(_save_document, entry, content_hash, chunks, vectors)
        report.append(f"{entry['arquivo']}: {len(chunks)} trechos")

    deactivated = await asyncio.to_thread(_deactivate_missing, [entry["arquivo"] for entry in entries])

    if deactivated:
        report.append(f"{deactivated} documento(s) fora do fontes.json marcados como inativos")

    return report


def _search(vector: list[float], limit: int) -> list[Source]:
    with db.pool().connection() as conn:
        rows = conn.execute(
            """
            SELECT d.title, d.link, c.page, c.content, 1 - (c.embedding <=> %s::halfvec) AS score
            FROM teo.chunks c
            JOIN teo.documents d ON d.id = c.document_id
            WHERE d.status = 'ativo'
            ORDER BY c.embedding <=> %s::halfvec
            LIMIT %s
            """,
            (np.array(vector, dtype=np.float32), np.array(vector, dtype=np.float32), limit),
        ).fetchall()

    return [Source(title=row[0], link=row[1], page=row[2], content=row[3], score=float(row[4])) for row in rows]


async def search(query: str) -> list[Source]:
    vector = (await llm.embed([query], "query"))[0]
    sources = await asyncio.to_thread(_search, vector, settings.knowledge_results)
    return [source for source in sources if source.score >= settings.knowledge_min_score]
