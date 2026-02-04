from enum import StrEnum


class JobType(StrEnum):
    IMAGE_TO_TEXT = "image_to_text"
    TEXT_TO_IMAGE = "text_to_image"
    TEXT_TO_DIAGRAM = "text_to_diagram"
    IMAGE_TO_TABLE = "image_to_table"


class JobStatus(StrEnum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class OutboxStatus(StrEnum):
    PENDING = "pending"
    PUBLISHED = "published"
    FAILED = "failed"
