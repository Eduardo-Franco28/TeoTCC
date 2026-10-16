import uvicorn

from teo_ai.config import settings


def main() -> None:
    uvicorn.run("teo_ai.api:app", host="0.0.0.0", port=settings.port)
