import os
from pathlib import Path

import docker
import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from redis.asyncio import Redis
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer


def _docker_available() -> bool:
    try:
        client = docker.from_env()
        client.ping()
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest_asyncio.fixture(scope="session")
def postgres_container():
    if not _docker_available():
        pytest.skip("Docker daemon is not available for integration tests")
    with PostgresContainer("postgres:16-alpine") as container:
        yield container


@pytest_asyncio.fixture(scope="session")
def redis_container():
    if not _docker_available():
        pytest.skip("Docker daemon is not available for integration tests")
    with RedisContainer("redis:7-alpine") as container:
        yield container


@pytest_asyncio.fixture(scope="session")
def postgres_dsn(postgres_container):
    sync_dsn = postgres_container.get_connection_url()  # postgresql://...
    async_dsn = (
        sync_dsn.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
        .replace("postgresql://", "postgresql+asyncpg://", 1)
        .replace("postgres://", "postgresql+asyncpg://", 1)
    )

    backend_dir = Path(__file__).resolve().parents[1]
    alembic_cfg = Config(str(backend_dir / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(backend_dir / "alembic"))
    alembic_cfg.set_main_option("prepend_sys_path", str(backend_dir))
    alembic_cfg.set_main_option("sqlalchemy.url", async_dsn)
    command.upgrade(alembic_cfg, "head")

    return async_dsn


@pytest_asyncio.fixture(scope="session")
def redis_dsn(redis_container):
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(6379)
    return f"redis://{host}:{port}/0"


@pytest_asyncio.fixture
async def app_client(postgres_dsn, redis_dsn):
    os.environ["POSTGRES_DSN"] = postgres_dsn
    os.environ["REDIS_DSN"] = redis_dsn
    os.environ["REDIS_STREAM_NAME"] = "jobs:test:stream"

    from ml_backend.config import get_settings

    get_settings.cache_clear()

    from ml_backend.main import create_app

    app = create_app()
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield client


@pytest_asyncio.fixture
async def redis_client(redis_dsn):
    client = Redis.from_url(redis_dsn, encoding="utf-8", decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()
