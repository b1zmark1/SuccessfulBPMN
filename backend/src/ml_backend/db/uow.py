from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ml_backend.db.repositories.jobs import JobsRepository, OutboxRepository


class SqlAlchemyUnitOfWork:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory
        self.session: AsyncSession | None = None
        self.jobs: JobsRepository | None = None
        self.outbox: OutboxRepository | None = None

    async def __aenter__(self) -> "SqlAlchemyUnitOfWork":
        self.session = self._session_factory()
        self.jobs = JobsRepository(self.session)
        self.outbox = OutboxRepository(self.session)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        assert self.session is not None
        if exc_type:
            await self.session.rollback()
        await self.session.close()

    async def commit(self) -> None:
        assert self.session is not None
        await self.session.commit()

    async def rollback(self) -> None:
        assert self.session is not None
        await self.session.rollback()


class UnitOfWorkFactory:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory

    def __call__(self) -> AsyncIterator[SqlAlchemyUnitOfWork]:
        return SqlAlchemyUnitOfWork(self._session_factory)
