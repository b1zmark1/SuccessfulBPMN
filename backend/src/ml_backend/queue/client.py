from redis.asyncio import Redis

from ml_backend.config import Settings


def build_redis_client(settings: Settings) -> Redis:
    return Redis.from_url(settings.redis_dsn, encoding="utf-8", decode_responses=True)
