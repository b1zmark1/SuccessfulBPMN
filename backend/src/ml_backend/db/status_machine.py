from ml_backend.db.enums import JobStatus

_ALLOWED_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.PENDING: {JobStatus.QUEUED, JobStatus.ERROR},
    JobStatus.QUEUED: {JobStatus.RUNNING, JobStatus.ERROR},
    JobStatus.RUNNING: {JobStatus.DONE, JobStatus.ERROR},
    JobStatus.DONE: set(),
    JobStatus.ERROR: set(),
}


def is_transition_allowed(from_status: JobStatus, to_status: JobStatus) -> bool:
    return to_status in _ALLOWED_TRANSITIONS[from_status]
