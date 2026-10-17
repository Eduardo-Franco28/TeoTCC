import uuid

from teo_ai import db
from teo_ai.config import settings


TITLE_LENGTH = 60
LIST_LIMIT = 50


def make_title(message: str) -> str:
    text = " ".join(message.split())
    return text if len(text) <= TITLE_LENGTH else text[:TITLE_LENGTH - 1].rstrip() + "…"


def create(user_email: str, first_message: str) -> uuid.UUID:
    conversation_id = uuid.uuid4()

    with db.pool().connection() as conn:
        conn.execute(
            "INSERT INTO teo.conversations (id, user_email, title) VALUES (%s, %s, %s)",
            (conversation_id, user_email, make_title(first_message)),
        )

    return conversation_id


def list_for(user_email: str) -> list[dict]:
    with db.pool().connection() as conn:
        rows = conn.execute(
            """
            SELECT c.id, COALESCE(c.title, 'Conversa'), COALESCE(MAX(m.created_at), c.created_at) AS updated_at
            FROM teo.conversations c
            LEFT JOIN teo.messages m ON m.conversation_id = c.id
            WHERE c.user_email = %s
            GROUP BY c.id
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            (user_email, LIST_LIMIT),
        ).fetchall()

    return [{"id": str(row[0]), "title": row[1], "updatedAt": row[2].isoformat()} for row in rows]


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
