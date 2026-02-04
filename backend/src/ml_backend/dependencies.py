from __future__ import annotations

import asyncio
from dataclasses import dataclass

from fastapi import Request

from ml_backend.config import Settings, get_settings
from ml_backend.db.session import Database
from ml_backend.db.uow import UnitOfWorkFactory
from ml_backend.queue.client import build_redis_client
from ml_backend.queue.redis_stream_publisher import RedisStreamPublisher
from ml_backend.services import HealthService, JobService, OutboxDispatcher


@dataclass
class AppContainer:
    settings: Settings
    db: Database
    redis_client: object
    dispatcher: OutboxDispatcher
    dispatcher_task: asyncio.Task | None
    job_service: JobService
    health_service: HealthService


def build_container() -> AppContainer:
    settings = get_settings()
    db = Database(settings)
    redis_client = build_redis_client(settings)
    publisher = RedisStreamPublisher(redis_client)
    uow_factory = UnitOfWorkFactory(db.session_factory)
    dispatcher = OutboxDispatcher(
        uow_factory=uow_factory,
        publisher=publisher,
        batch_size=settings.outbox_batch_size,
    )
    job_service = JobService(settings=settings, uow_factory=uow_factory, dispatcher=dispatcher)
    health_service = HealthService(session_factory=db.session_factory, redis_client=redis_client)

    return AppContainer(
        settings=settings,
        db=db,
        redis_client=redis_client,
        dispatcher=dispatcher,
        dispatcher_task=None,
        job_service=job_service,
        health_service=health_service,
    )


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


def get_job_service(request: Request) -> JobService:
    return get_container(request).job_service


def get_health_service(request: Request) -> HealthService:
    return get_container(request).health_service
