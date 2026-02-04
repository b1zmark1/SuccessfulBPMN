from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ml_backend.db.enums import JobStatus, JobType


class CreateJobRequest(BaseModel):
    job_type: JobType
    meta: dict = Field(default_factory=dict)


class CreateJobResponse(BaseModel):
    job_id: UUID


class JobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    job_id: UUID
    job_type: JobType
    status: JobStatus
    result: dict | None = None
    error: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class HealthResponse(BaseModel):
    status: str
    postgres: str
    redis: str
