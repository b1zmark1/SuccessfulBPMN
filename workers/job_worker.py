from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from redis.asyncio import Redis


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = REPO_ROOT / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_backend.config import Settings  # noqa: E402
from ml_backend.db.enums import JobStatus, JobType  # noqa: E402
from ml_backend.db.session import Database  # noqa: E402
from ml_backend.db.uow import UnitOfWorkFactory  # noqa: E402
from workers.image_to_text_pipeline import run_image_to_text_pipeline  # noqa: E402
from workers.text_to_image_pipeline import run_text_to_image_pipeline  # noqa: E402


class JobWorker:
    def __init__(
        self,
        *,
        redis_dsn: str,
        postgres_dsn: str,
        stream_name: str,
        start_id: str,
        block_ms: int,
    ) -> None:
        settings = Settings(POSTGRES_DSN=postgres_dsn, REDIS_DSN=redis_dsn, REDIS_STREAM_NAME=stream_name)
        self._db = Database(settings)
        self._uow_factory = UnitOfWorkFactory(self._db.session_factory)
        self._redis = Redis.from_url(redis_dsn, encoding="utf-8", decode_responses=True)
        self._stream_name = stream_name
        self._last_id = start_id
        self._block_ms = block_ms

    async def close(self) -> None:
        await self._redis.aclose()
        await self._db.dispose()

    async def _set_running(self, job_id: uuid.UUID) -> bool:
        async with self._uow_factory() as uow:
            assert uow.jobs is not None
            job = await uow.jobs.get(job_id)
            if job is None:
                return False
            if job.status in {JobStatus.DONE, JobStatus.ERROR}:
                return False
            if job.status == JobStatus.PENDING:
                await uow.jobs.set_status(job=job, status=JobStatus.QUEUED)
            if job.status in {JobStatus.QUEUED, JobStatus.PENDING}:
                await uow.jobs.set_status(job=job, status=JobStatus.RUNNING)
            await uow.commit()
            return True

    async def _set_done(self, job_id: uuid.UUID, result: dict[str, Any]) -> None:
        async with self._uow_factory() as uow:
            assert uow.jobs is not None
            job = await uow.jobs.get(job_id)
            if job is None:
                return
            if job.status != JobStatus.RUNNING:
                if job.status == JobStatus.QUEUED:
                    await uow.jobs.set_status(job=job, status=JobStatus.RUNNING)
                elif job.status in {JobStatus.DONE, JobStatus.ERROR}:
                    return
            await uow.jobs.set_status(job=job, status=JobStatus.DONE, result=result)
            await uow.commit()

    async def _set_error(self, job_id: uuid.UUID, message: str) -> None:
        async with self._uow_factory() as uow:
            assert uow.jobs is not None
            job = await uow.jobs.get(job_id)
            if job is None:
                return
            if job.status in {JobStatus.DONE, JobStatus.ERROR}:
                return
            if job.status == JobStatus.PENDING:
                await uow.jobs.set_status(job=job, status=JobStatus.QUEUED)
            await uow.jobs.set_status(job=job, status=JobStatus.ERROR, error=message[:4000])
            await uow.commit()

    async def _handle_job(self, job_type: str, meta: dict[str, Any]) -> dict[str, Any]:
        if job_type == JobType.IMAGE_TO_TEXT.value:
            image_url = meta.get("image_url")
            if not isinstance(image_url, str) or not image_url.strip():
                raise ValueError("meta.image_url is required for image_to_text")
            narrator_mode = meta.get("narrator_mode")
            if narrator_mode not in {"text", "table"}:
                narrator_mode = "text"
            return await run_image_to_text_pipeline(image_url, narrator_mode=narrator_mode)
        if job_type == JobType.TEXT_TO_IMAGE.value:
            prompt = meta.get("prompt") or meta.get("promt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("meta.prompt is required for text_to_image")
            return run_text_to_image_pipeline(prompt)
        raise ValueError(f"Unsupported job type for worker: {job_type}")

    async def run_forever(self) -> None:
        print(f"[worker] listening stream={self._stream_name} start_id={self._last_id}")
        while True:
            streams = await self._redis.xread(
                streams={self._stream_name: self._last_id},
                count=1,
                block=self._block_ms,
            )
            if not streams:
                continue

            for _, messages in streams:
                for stream_id, fields in messages:
                    self._last_id = stream_id
                    raw_job_id = fields.get("JobID")
                    raw_job_type = fields.get("job_type")
                    raw_meta = fields.get("Metadata")

                    if not raw_job_id or not raw_job_type or not raw_meta:
                        print(f"[worker] skip invalid payload stream_id={stream_id}")
                        continue

                    try:
                        job_id = uuid.UUID(raw_job_id)
                        meta = json.loads(raw_meta)
                    except Exception as exc:
                        print(f"[worker] invalid message stream_id={stream_id}: {exc}")
                        continue

                    should_process = await self._set_running(job_id)
                    if not should_process:
                        print(f"[worker] skip job_id={job_id} (terminal or not found)")
                        continue

                    print(f"[worker] processing job_id={job_id} type={raw_job_type}")
                    try:
                        result = await self._handle_job(raw_job_type, meta)
                        await self._set_done(job_id, result)
                        print(f"[worker] done job_id={job_id}")
                    except Exception as exc:
                        await self._set_error(job_id, str(exc))
                        print(f"[worker] error job_id={job_id}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Redis stream worker for ML jobs.")
    parser.add_argument("--redis-dsn", default=os.getenv("REDIS_DSN", "redis://localhost:6379/0"))
    parser.add_argument(
        "--postgres-dsn",
        default=os.getenv("POSTGRES_DSN", "postgresql+asyncpg://postgres:postgres@localhost:5432/ml_jobs"),
    )
    parser.add_argument("--stream-name", default=os.getenv("REDIS_STREAM_NAME", "jobs:stream"))
    parser.add_argument(
        "--start-id",
        default=os.getenv("WORKER_START_ID", "0-0"),
        help="Redis stream ID to start from. Use '$' for only new messages.",
    )
    parser.add_argument("--block-ms", type=int, default=5000)
    return parser.parse_args()


async def _main() -> None:
    args = parse_args()
    worker = JobWorker(
        redis_dsn=args.redis_dsn,
        postgres_dsn=args.postgres_dsn,
        stream_name=args.stream_name,
        start_id=args.start_id,
        block_ms=args.block_ms,
    )
    try:
        await worker.run_forever()
    finally:
        await worker.close()


if __name__ == "__main__":
    asyncio.run(_main())
