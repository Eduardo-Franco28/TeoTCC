from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    port: int = 8000
    database_url: str = "postgresql://teo:teo@localhost:5434/teo"
    integramente_api_url: str = "http://localhost:8080"
    tcc_jwt_secret: str = ""
    teo_web_origins: str = "http://localhost:5173"

    llm_base_url: str = "https://integrate.api.nvidia.com/v1"
    llm_api_key: str = ""
    teo_model: str = ""

    teo_embedding_model: str = ""
    teo_embedding_dim: int = 1024
    teo_embedding_input_type: bool = True

    max_agent_steps: int = 6
    max_history_messages: int = 12
    knowledge_results: int = 5
    knowledge_min_score: float = 0.35

    @property
    def mcp_url(self) -> str:
        return self.integramente_api_url.rstrip("/") + "/mcp"

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.teo_web_origins.split(",") if origin.strip()]


settings = Settings()
