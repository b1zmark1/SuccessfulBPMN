import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ml_backend.db.enums import JobStatus, JobType
from ml_backend.db.repositories import JobsRepository


@pytest.mark.asyncio
async def test_job_status_transitions(postgres_dsn):
    engine = create_async_engine(postgres_dsn)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as session:
        repo = JobsRepository(session)
        job = await repo.create(job_type=JobType.IMAGE_TO_TEXT, meta={"input": "abc"})

        await repo.set_status(job=job, status=JobStatus.QUEUED)
        await repo.set_status(job=job, status=JobStatus.RUNNING)
        await repo.set_status(job=job, status=JobStatus.DONE, result={"text": "ok"})
        await session.commit()

        assert job.status == JobStatus.DONE
        assert job.started_at is not None
        assert job.finished_at is not None
        assert job.result == {"text": "ok"}

    async with session_factory() as session:
        repo = JobsRepository(session)
        job = await repo.create(job_type=JobType.IMAGE_TO_TEXT, meta={"input": "abc"})
        await session.flush()

        with pytest.raises(ValueError):
            await repo.set_status(job=job, status=JobStatus.DONE)

    await engine.dispose()
