from ml_backend.db.enums import JobStatus
from ml_backend.db.status_machine import is_transition_allowed


def test_allowed_status_transitions() -> None:
    assert is_transition_allowed(JobStatus.PENDING, JobStatus.QUEUED)
    assert is_transition_allowed(JobStatus.QUEUED, JobStatus.RUNNING)
    assert is_transition_allowed(JobStatus.RUNNING, JobStatus.DONE)


def test_forbidden_status_transitions() -> None:
    assert not is_transition_allowed(JobStatus.PENDING, JobStatus.DONE)
    assert not is_transition_allowed(JobStatus.DONE, JobStatus.RUNNING)
    assert not is_transition_allowed(JobStatus.ERROR, JobStatus.QUEUED)
