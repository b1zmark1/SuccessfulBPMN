import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from ml_backend.db.enums import JobStatus, JobType, OutboxStatus
from ml_backend.db.models import JobModel, OutboxMessageModel
from ml_backend.db.status_machine import is_transition_allowed


class JobsRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, *, job_type: JobType, meta: dict) -> JobModel:
        job = JobModel(job_type=job_type, status=JobStatus.PENDING, meta=meta)
        self._session.add(job)
        await self._session.flush()
        return job

    async def get(self, job_id: uuid.UUID) -> JobModel | None:
        return await self._session.get(JobModel, job_id)

    async def set_status(
        self,
        *,
        job: JobModel,
        status: JobStatus,
        result: dict | None = None,
        error: str | None = None,
    ) -> JobModel:
        if not is_transition_allowed(job.status, status):
            raise ValueError(f"Invalid status transition: {job.status} -> {status}")

        job.status = status

        if status == JobStatus.RUNNING and job.started_at is None:
            job.started_at = datetime.now(UTC)

        if status in (JobStatus.DONE, JobStatus.ERROR):
            if job.started_at is None:
                job.started_at = datetime.now(UTC)
            job.finished_at = datetime.now(UTC)

        if status == JobStatus.DONE:
            job.result = result or {}
            job.error = None
        elif status == JobStatus.ERROR:
            job.error = error or "Unknown worker error"

        await self._session.flush()
        return job


class OutboxRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(self, *, aggregate_id: uuid.UUID, topic: str, payload: dict) -> OutboxMessageModel:
        message = OutboxMessageModel(
            aggregate_id=aggregate_id,
            topic=topic,
            payload=payload,
            status=OutboxStatus.PENDING,
        )
        self._session.add(message)
        await self._session.flush()
        return message

    async def list_pending(self, *, limit: int) -> list[OutboxMessageModel]:
        stmt: Select[tuple[OutboxMessageModel]] = (
            select(OutboxMessageModel)
            .where(
                OutboxMessageModel.status == OutboxStatus.PENDING,
                OutboxMessageModel.available_at <= datetime.now(UTC),
            )
            .order_by(OutboxMessageModel.outbox_id.asc())
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def mark_published(self, message: OutboxMessageModel) -> None:
        message.status = OutboxStatus.PUBLISHED
        message.published_at = datetime.now(UTC)
        message.error = None
        await self._session.flush()

    async def mark_failed(self, message: OutboxMessageModel, reason: str, retry_after_seconds: int) -> None:
        message.status = OutboxStatus.PENDING
        message.attempts += 1
        message.error = reason
        message.available_at = datetime.now(UTC) + timedelta(seconds=retry_after_seconds)
        await self._session.flush()
