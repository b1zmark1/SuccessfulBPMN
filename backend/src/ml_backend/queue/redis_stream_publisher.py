from redis.asyncio import Redis

from ml_backend.queue.base import QueuePublisher
from ml_backend.queue.schemas import JobQueueMessage


class RedisStreamPublisher(QueuePublisher):
    def __init__(self, redis: Redis):
        self._redis = redis

    async def publish(self, *, topic: str, message: JobQueueMessage) -> str:
        stream_id = await self._redis.xadd(name=topic, fields=message.to_stream_fields())
        return stream_id.decode() if isinstance(stream_id, bytes) else str(stream_id)
