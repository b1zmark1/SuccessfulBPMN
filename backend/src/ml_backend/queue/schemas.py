from dataclasses import dataclass
from uuid import UUID

from ml_backend.db.enums import JobType

QUEUE_MESSAGE_VERSION = "v1"


@dataclass(frozen=True)
class JobQueueMessage:
    job_id: UUID
    job_type: JobType
    meta: dict

    def to_stream_fields(self) -> dict[str, str]:
        return {
            "version": QUEUE_MESSAGE_VERSION,
            "JobID": str(self.job_id),
            "job_type": self.job_type.value,
            "Metadata": _compact_json(self.meta),
        }


# Keep stream payload small and consistent for workers.
def _compact_json(payload: dict) -> str:
    import json

    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
