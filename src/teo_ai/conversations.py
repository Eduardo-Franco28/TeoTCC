import uuid

from teo_ai import db
from teo_ai.config import settings


def create(user_email: str) -> uuid.UUID:
    conversation_id = uuid.uuid4()

    with db.pool().connection() as conn:
        conn.execute(
            "INSERT INTO teo.conversations (id, user_email) VALUES (%s, %s)",
            (conversation_id, user_email),
        )

    return conversation_id


def belongs_to(conversation_id: uuid.UUID, user_email: str) -> bool:
    with db.pool().connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM teo.conversations WHERE id = %s AND user_email = %s",
            (conversation_id, user_email),
        ).fetchone()

    return row is not None


def history(conversation_id: uuid.UUID) -> list[dict]:
    with db.pool().connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content FROM (
                SELECT id, role, content FROM teo.messages
                WHERE conversation_id = %s
                ORDER BY id DESC
                LIMIT %s
            ) recent
            ORDER BY id
            """,
            (conversation_id, settings.max_history_messages),
        ).fetchall()

    return [{"role": row[0], "content": row[1]} for row in rows]


def all_messages(conversation_id: uuid.UUID) -> list[dict]:
    with db.pool().connection() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM teo.messages WHERE conversation_id = %s ORDER BY id",
            (conversation_id,),
        ).fetchall()

    return [{"role": row[0], "content": row[1], "createdAt": row[2].isoformat()} for row in rows]


def append(conversation_id: uuid.UUID, role: str, content: str) -> None:
    with db.pool().connection() as conn:
        conn.execute(
            "INSERT INTO teo.messages (conversation_id, role, content) VALUES (%s, %s, %s)",
            (conversation_id, role, content),
        )
