from types import SimpleNamespace
from uuid import uuid4

import pytest

from ml_backend.config import Settings
from ml_backend.db.enums import JobType
from ml_backend.queue.schemas import JobQueueMessage
from ml_backend.services.job_service import JobService


class _FakeJobsRepo:
    def __init__(self, job):
        self._job = job
        self.create_calls = []

    async def create(self, *, job_type, meta):
        self.create_calls.append((job_type, meta))
        return self._job

    async def get(self, job_id):
        if job_id == self._job.job_id:
            return self._job
        return None


class _FakeOutboxRepo:
    def __init__(self):
        self.add_calls = []

    async def add(self, *, aggregate_id, topic, payload):
        self.add_calls.append(
            {
                "aggregate_id": aggregate_id,
                "topic": topic,
                "payload": payload,
            }
        )
        return None


class _FakeUoW:
    def __init__(self, jobs, outbox):
        self.jobs = jobs
        self.outbox = outbox
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    async def commit(self):
        self.committed = True


class _FakeDispatcher:
    def __init__(self):
        self.run_once_calls = 0

    async def run_once(self):
        self.run_once_calls += 1
        return 1


@pytest.mark.asyncio
async def test_job_service_create_job_writes_outbox_and_dispatches():
    job_id = uuid4()
    fake_job = SimpleNamespace(job_id=job_id, job_type=JobType.TEXT_TO_DIAGRAM, meta={"prompt": "x"})
    jobs_repo = _FakeJobsRepo(fake_job)
    outbox_repo = _FakeOutboxRepo()
    uow = _FakeUoW(jobs_repo, outbox_repo)
    dispatcher = _FakeDispatcher()

    service = JobService(
        settings=Settings(REDIS_STREAM_NAME="jobs:test:stream"),
        uow_factory=lambda: uow,
        dispatcher=dispatcher,
    )

    created = await service.create_job(job_type=JobType.TEXT_TO_DIAGRAM, meta={"prompt": "x"})

    assert created == job_id
    assert uow.committed is True
    assert dispatcher.run_once_calls == 1
    assert len(outbox_repo.add_calls) == 1
    assert outbox_repo.add_calls[0]["topic"] == "jobs:test:stream"
    assert outbox_repo.add_calls[0]["payload"]["JobID"] == str(job_id)
    assert outbox_repo.add_calls[0]["payload"]["job_type"] == "text_to_diagram"


def test_queue_message_contract_for_redis_stream():
    message = JobQueueMessage(job_id=uuid4(), job_type=JobType.IMAGE_TO_TABLE, meta={"k": "v"})

    payload = message.to_stream_fields()

    assert set(payload.keys()) == {"version", "JobID", "job_type", "Metadata"}
    assert payload["job_type"] == "image_to_table"
