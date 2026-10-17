import asyncio
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from teo_ai import agent, conversations, creator, db
from teo_ai.auth import TeoUser, current_user
from teo_ai.config import settings


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


class ChatResponse(BaseModel):
    conversationId: uuid.UUID
    reply: str
    drafts: list[dict]
    sources: list[dict]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user: TeoUser = Depends(current_user)) -> ChatResponse:
    if not settings.llm_api_key or not settings.teo_model:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "O Teo ainda não está configurado com uma LLM")

    if request.conversation_id is None:
        conversation_id = await asyncio.to_thread(conversations.create, user.email, request.message)
        history: list[dict] = []
    else:
        conversation_id = request.conversation_id
        if not await asyncio.to_thread(conversations.belongs_to, conversation_id, user.email):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversa não encontrada")
        history = await asyncio.to_thread(conversations.history, conversation_id)

    result = await agent.run(user, history, request.message)

    await asyncio.to_thread(conversations.append, conversation_id, "user", request.message)
    await asyncio.to_thread(conversations.append, conversation_id, "assistant", result.reply)

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


@app.get("/conversations")
async def list_conversations(user: TeoUser = Depends(current_user)) -> list[dict]:
    return await asyncio.to_thread(conversations.list_for, user.email)


@app.get("/conversations/{conversation_id}/messages")
async def messages(conversation_id: uuid.UUID, user: TeoUser = Depends(current_user)) -> list[dict]:
    if not await asyncio.to_thread(conversations.belongs_to, conversation_id, user.email):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversa não encontrada")

    return await asyncio.to_thread(conversations.all_messages, conversation_id)
