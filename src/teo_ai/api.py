import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from teo_ai import agent, conversations, creator, db
from teo_ai.auth import TeoUser, current_user
from teo_ai.config import settings


logger = logging.getLogger("teo")


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.create_schema()
    db.open_pool()
    yield
    db.close_pool()


app = FastAPI(title="Teo", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


class ChatRequest(BaseModel):
    conversation_id: uuid.UUID | None = Field(default=None, alias="conversationId")
    message: str = Field(min_length=1, max_length=2000)


class LettersActivityRequest(BaseModel):
    theme: str | None = Field(default=None, max_length=120)
    words: list[str] = Field(default_factory=list, max_length=10)
    count: int = Field(default=5, ge=1, le=10)
    hidden: int | None = Field(default=None, ge=1, le=3)


class SumActivityRequest(BaseModel):
    count: int = Field(default=5, ge=1, le=10)
    max_total: Literal[10, 20, 30] = Field(default=10, alias="maxTotal")


class GroupsActivityRequest(BaseModel):
    theme: str = Field(min_length=1, max_length=120)
    subject_id: int = Field(alias="subjectId")
    groups: int = Field(default=2, ge=2, le=3)
    items: int = Field(default=2, ge=2, le=3)


class ChatResponse(BaseModel):
    conversationId: uuid.UUID
    reply: str
    drafts: list[dict]
    sources: list[dict]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


async def open_conversation(request: ChatRequest, user: TeoUser) -> tuple[uuid.UUID, list[dict]]:
    if not settings.llm_api_key or not settings.teo_model:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "O Teo ainda não está configurado com uma LLM")

    if request.conversation_id is None:
        conversation_id = await asyncio.to_thread(conversations.create, user.email, request.message)
        return conversation_id, []

    if not await asyncio.to_thread(conversations.belongs_to, request.conversation_id, user.email):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversa não encontrada")

    return request.conversation_id, await asyncio.to_thread(conversations.history, request.conversation_id)


async def save_exchange(conversation_id: uuid.UUID, message: str, reply: str) -> None:
    await asyncio.to_thread(conversations.append, conversation_id, "user", message)
    await asyncio.to_thread(conversations.append, conversation_id, "assistant", reply)


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, user: TeoUser = Depends(current_user)) -> StreamingResponse:
    conversation_id, history = await open_conversation(request, user)

    async def events():
        try:
            async for event in agent.run_stream(user, history, request.message):
                if event["type"] != "done":
                    yield sse(event)
                    continue

                result = event["result"]
                await save_exchange(conversation_id, request.message, result.reply)
                yield sse({
                    "type": "done",
                    "conversationId": str(conversation_id),
                    "reply": result.reply,
                    "drafts": result.drafts,
                    "sources": result.sources,
                })
        except Exception:
            logger.exception("Falha ao responder no chat")
            yield sse({"type": "error", "text": "Não consegui falar com o Teo agora. Tente de novo em instantes."})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user: TeoUser = Depends(current_user)) -> ChatResponse:
    conversation_id, history = await open_conversation(request, user)

    result = await agent.run(user, history, request.message)

    await save_exchange(conversation_id, request.message, result.reply)

    return ChatResponse(
        conversationId=conversation_id,
        reply=result.reply,
        drafts=result.drafts,
        sources=result.sources,
    )


@app.post("/activities/letters")
async def create_letters_activity(request: LettersActivityRequest, user: TeoUser = Depends(current_user)) -> dict:
    if not request.words and not (request.theme and request.theme.strip()):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Informe um tema ou as palavras da atividade")

    try:
        draft = await creator.create_letters(user, request.theme, request.words, request.count, request.hidden)
    except creator.CreationError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(error))

    return {"draft": draft}


@app.post("/activities/sum")
async def create_sum_activity(request: SumActivityRequest, user: TeoUser = Depends(current_user)) -> dict:
    try:
        draft = await creator.create_sum(user, request.count, request.max_total)
    except creator.CreationError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(error))

    return {"draft": draft}


@app.post("/activities/groups")
async def create_groups_activity(request: GroupsActivityRequest, user: TeoUser = Depends(current_user)) -> dict:
    if not request.theme.strip():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Informe o tema da atividade")

    try:
        return await creator.create_groups(user, request.theme, request.subject_id, request.groups, request.items)
    except creator.CreationError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(error))


@app.get("/subjects")
async def list_subjects(user: TeoUser = Depends(current_user)) -> list[dict]:
    return await creator.subjects(user)


@app.get("/conversations")
async def list_conversations(user: TeoUser = Depends(current_user)) -> list[dict]:
    return await asyncio.to_thread(conversations.list_for, user.email)


@app.get("/conversations/{conversation_id}/messages")
async def messages(conversation_id: uuid.UUID, user: TeoUser = Depends(current_user)) -> list[dict]:
    if not await asyncio.to_thread(conversations.belongs_to, conversation_id, user.email):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversa não encontrada")

    return await asyncio.to_thread(conversations.all_messages, conversation_id)
