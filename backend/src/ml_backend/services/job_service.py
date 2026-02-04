import logging
from collections.abc import Callable
from uuid import UUID

from ml_backend.config import Settings
from ml_backend.db.models import JobModel
from ml_backend.db.uow import SqlAlchemyUnitOfWork
from ml_backend.queue.schemas import JobQueueMessage
from ml_backend.services.outbox_dispatcher import OutboxDispatcher

logger = logging.getLogger(__name__)


class JobService:
    def __init__(
        self,
        *,
        settings: Settings,
        uow_factory: Callable[[], SqlAlchemyUnitOfWork],
        dispatcher: OutboxDispatcher,
    ):
        self._settings = settings
        self._uow_factory = uow_factory
        self._dispatcher = dispatcher

    async def create_job(self, *, job_type, meta: dict) -> UUID:
        async with self._uow_factory() as uow:
            assert uow.jobs is not None
            assert uow.outbox is not None

            job = await uow.jobs.create(job_type=job_type, meta=meta)
            queue_message = JobQueueMessage(
                job_id=job.job_id,
                job_type=job.job_type,
                meta=job.meta,
            )
            await uow.outbox.add(
                aggregate_id=job.job_id,
                topic=self._settings.redis_stream_name,
                payload=queue_message.to_stream_fields(),
            )
            await uow.commit()

        logger.info("job_created", extra={"job_id": str(job.job_id)})

        # Best-effort immediate flush keeps API responsive while preserving reliability via outbox retries.
        await self._dispatcher.run_once()

        return job.job_id

    async def get_job(self, job_id: UUID) -> JobModel | None:
        async with self._uow_factory() as uow:
            assert uow.jobs is not None
            return await uow.jobs.get(job_id)
