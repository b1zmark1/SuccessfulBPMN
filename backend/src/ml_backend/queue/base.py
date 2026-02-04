from typing import Protocol

from ml_backend.queue.schemas import JobQueueMessage


class QueuePublisher(Protocol):
    async def publish(self, *, topic: str, message: JobQueueMessage) -> str:
        """Publish a job message and return queue-specific message id."""
