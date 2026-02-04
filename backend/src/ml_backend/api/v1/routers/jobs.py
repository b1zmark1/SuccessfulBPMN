from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from ml_backend.api.v1.schemas import CreateJobRequest, CreateJobResponse, JobResponse
from ml_backend.dependencies import get_job_service
from ml_backend.services import JobService

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("", response_model=CreateJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_job(payload: CreateJobRequest, service: JobService = Depends(get_job_service)) -> CreateJobResponse:
    job_id = await service.create_job(job_type=payload.job_type, meta=payload.meta)
    return CreateJobResponse(job_id=job_id)


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: UUID, service: JobService = Depends(get_job_service)) -> JobResponse:
    job = await service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return JobResponse.model_validate(job)
