import asyncio
import logging
from collections.abc import Callable

from ml_backend.db.enums import JobStatus
from ml_backend.db.uow import SqlAlchemyUnitOfWork
from ml_backend.queue.base import QueuePublisher
from ml_backend.queue.schemas import JobQueueMessage

logger = logging.getLogger(__name__)


class OutboxDispatcher:
    def __init__(
        self,
        *,
        uow_factory: Callable[[], SqlAlchemyUnitOfWork],
        publisher: QueuePublisher,
        batch_size: int,
        poll_interval_seconds: float = 0.5,
        max_retry_delay_seconds: int = 30,
    ):
        self._uow_factory = uow_factory
        self._publisher = publisher
        self._batch_size = batch_size
        self._poll_interval_seconds = poll_interval_seconds
        self._max_retry_delay_seconds = max_retry_delay_seconds
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        self._stop_event.clear()
        while not self._stop_event.is_set():
            processed = await self.run_once()
            if processed == 0:
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=self._poll_interval_seconds)
                except TimeoutError:
                    continue

    async def stop(self) -> None:
        self._stop_event.set()

    async def run_once(self) -> int:
        processed = 0
        async with self._uow_factory() as uow:
            assert uow.outbox is not None
            assert uow.jobs is not None

            pending = await uow.outbox.list_pending(limit=self._batch_size)
            for message in pending:
                try:
                    job = await uow.jobs.get(message.aggregate_id)
                    if job is None:
                        raise ValueError(f"Job not found for outbox message {message.outbox_id}")

                    queue_message = JobQueueMessage(
                        job_id=job.job_id,
                        job_type=job.job_type,
                        meta=job.meta,
                    )
                    stream_id = await self._publisher.publish(topic=message.topic, message=queue_message)
                    await uow.outbox.mark_published(message)
                    if job.status.value == JobStatus.PENDING.value:
                        await uow.jobs.set_status(job=job, status=JobStatus.QUEUED)

                    logger.info(
                        "outbox_message_published",
                        extra={
                            "outbox_id": message.outbox_id,
                            "job_id": str(job.job_id),
                            "stream_id": stream_id,
                            "topic": message.topic,
                        },
                    )
                    processed += 1
                except Exception as exc:  # noqa: BLE001
                    retry_delay = min(2 ** max(1, message.attempts + 1), self._max_retry_delay_seconds)
                    await uow.outbox.mark_failed(message, reason=str(exc), retry_after_seconds=retry_delay)
                    logger.exception(
                        "outbox_message_publish_failed",
                        extra={
                            "outbox_id": message.outbox_id,
                            "job_id": str(message.aggregate_id),
                            "topic": message.topic,
                            "retry_after_seconds": retry_delay,
                        },
                    )

            await uow.commit()

        return processed
