from fastapi import APIRouter

from ml_backend.api.v1.routers.health import router as health_router
from ml_backend.api.v1.routers.jobs import router as jobs_router

api_router = APIRouter()
api_router.include_router(jobs_router)
api_router.include_router(health_router)
