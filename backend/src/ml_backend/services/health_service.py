from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker


class HealthService:
    def __init__(self, *, session_factory: async_sessionmaker, redis_client):
        self._session_factory = session_factory
        self._redis_client = redis_client

    async def postgres_ok(self) -> bool:
        try:
            async with self._session_factory() as session:
                await session.execute(text("select 1"))
            return True
        except Exception:  # noqa: BLE001
            return False

    async def redis_ok(self) -> bool:
        try:
            response = await self._redis_client.ping()
            return bool(response)
        except Exception:  # noqa: BLE001
            return False
