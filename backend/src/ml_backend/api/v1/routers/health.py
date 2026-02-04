from fastapi import APIRouter, Depends

from ml_backend.api.v1.schemas import HealthResponse
from ml_backend.dependencies import get_health_service
from ml_backend.services import HealthService

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health(service: HealthService = Depends(get_health_service)) -> HealthResponse:
    postgres_ok = await service.postgres_ok()
    redis_ok = await service.redis_ok()
    status = "ok" if postgres_ok and redis_ok else "degraded"
    return HealthResponse(
        status=status,
        postgres="ok" if postgres_ok else "error",
        redis="ok" if redis_ok else "error",
    )
