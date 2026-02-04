import asyncio
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from starlette.responses import Response

from ml_backend.api import api_router
from ml_backend.config import get_settings
from ml_backend.dependencies import AppContainer, build_container
from ml_backend.logging import configure_logging, request_id_ctx


@asynccontextmanager
async def lifespan(app: FastAPI):
    container: AppContainer = build_container()
    configure_logging(container.settings.log_level)

    app.state.container = container
    container.dispatcher_task = asyncio.create_task(container.dispatcher.start())

    try:
        yield
    finally:
        await container.dispatcher.stop()
        if container.dispatcher_task is not None:
            container.dispatcher_task.cancel()
            try:
                await container.dispatcher_task
            except asyncio.CancelledError:
                pass
        await container.redis_client.aclose()
        await container.db.dispose()


def create_app() -> FastAPI:
    app = FastAPI(title="ML Job Backend", version="1.0.0", lifespan=lifespan)
    settings = get_settings()

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token = request_id_ctx.set(request_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            request_id_ctx.reset(token)

    app.include_router(api_router, prefix=settings.api_prefix)
    return app


app = create_app()
