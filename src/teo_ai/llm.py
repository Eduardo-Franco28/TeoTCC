from openai import AsyncOpenAI

from teo_ai.config import settings

_client: AsyncOpenAI | None = None


def client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key or "sem-chave")
    return _client


async def embed(texts: list[str], kind: str) -> list[list[float]]:
    extra_body = {"input_type": kind, "truncate": "END"} if settings.teo_embedding_input_type else None

    response = await client().embeddings.create(
        model=settings.teo_embedding_model,
        input=texts,
        encoding_format="float",
        extra_body=extra_body,
    )

    return [item.embedding for item in response.data]
