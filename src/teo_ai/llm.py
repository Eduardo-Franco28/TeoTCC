from typing import Any, AsyncIterator

from openai import APIError, APIStatusError, AsyncOpenAI

from teo_ai.config import settings

MODELS_PER_REQUEST = 3

RETRY_STATUS = {404, 408, 429}

_client: AsyncOpenAI | None = None


def client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key or "sem-chave")
    return _client


def should_retry(error: Exception) -> bool:
    if isinstance(error, APIStatusError):
        return error.status_code >= 500 or error.status_code in RETRY_STATUS
    return isinstance(error, APIError)


def attempts() -> list[list[str]]:
    chain = settings.model_chain
    return [chain[start:start + MODELS_PER_REQUEST] for start in range(len(chain))]


def request_options(models: list[str]) -> dict[str, Any]:
    options: dict[str, Any] = {"model": models[0], "timeout": settings.llm_timeout}
    if len(models) > 1:
        options["extra_body"] = {"models": models}
    return options


async def chat(messages: list[dict], **kwargs: Any):
    last_error: Exception | None = None

    for models in attempts():
        try:
            return await client().chat.completions.create(messages=messages, **request_options(models), **kwargs)
        except Exception as error:
            if not should_retry(error):
                raise
            last_error = error

    raise last_error or RuntimeError("Nenhum modelo configurado")


async def chat_stream(messages: list[dict], **kwargs: Any) -> tuple[Any, AsyncIterator[Any]]:
    last_error: Exception | None = None

    for models in attempts():
        try:
            stream = await client().chat.completions.create(
                messages=messages, stream=True, **request_options(models), **kwargs
            )
            iterator = stream.__aiter__()
            first = await iterator.__anext__()
            return first, iterator
        except StopAsyncIteration:
            last_error = RuntimeError("O modelo não respondeu nada")
        except Exception as error:
            if not should_retry(error):
                raise
            last_error = error

    raise last_error or RuntimeError("Nenhum modelo configurado")


async def embed(texts: list[str], kind: str) -> list[list[float]]:
    extra_body = {"input_type": kind, "truncate": "END"} if settings.teo_embedding_input_type else None

    response = await client().embeddings.create(
        model=settings.teo_embedding_model,
        input=texts,
        encoding_format="float",
        extra_body=extra_body,
    )

    return [item.embedding for item in response.data]
