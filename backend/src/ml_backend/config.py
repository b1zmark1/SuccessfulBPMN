from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="ml-job-backend", alias="APP_NAME")
    app_env: str = Field(default="dev", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    api_prefix: str = Field(default="/api/v1", alias="API_PREFIX")

    postgres_dsn: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/ml_jobs",
        alias="POSTGRES_DSN",
    )
    redis_dsn: str = Field(default="redis://localhost:6379/0", alias="REDIS_DSN")

    redis_stream_name: str = Field(default="jobs:stream", alias="REDIS_STREAM_NAME")
    outbox_batch_size: int = Field(default=100, alias="OUTBOX_BATCH_SIZE")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
