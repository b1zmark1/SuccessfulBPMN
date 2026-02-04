"""create jobs and outbox tables

Revision ID: 0001_create_jobs_and_outbox
Revises:
Create Date: 2026-02-04 09:45:00

"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001_create_jobs_and_outbox"
down_revision: str | None = None
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


job_type_enum = postgresql.ENUM(
    "image_to_text",
    "text_to_image",
    "text_to_diagram",
    "image_to_table",
    name="job_type",
    create_type=False,
)
job_status_enum = postgresql.ENUM(
    "pending", "queued", "running", "done", "error", name="job_status", create_type=False
)
outbox_status_enum = postgresql.ENUM("pending", "published", "failed", name="outbox_status", create_type=False)


def upgrade() -> None:
    bind = op.get_bind()
    job_type_enum.create(bind, checkfirst=True)
    job_status_enum.create(bind, checkfirst=True)
    outbox_status_enum.create(bind, checkfirst=True)

    op.create_table(
        "jobs",
        sa.Column("jobId", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_type", job_type_enum, nullable=False),
        sa.Column("status", job_status_enum, nullable=False),
        sa.Column("meta", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("jobId"),
    )
    op.create_index("ix_jobs_status", "jobs", ["status"], unique=False)
    op.create_index("ix_jobs_created_at", "jobs", ["created_at"], unique=False)

    op.create_table(
        "outbox_messages",
        sa.Column("outbox_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("aggregate_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("topic", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", outbox_status_enum, nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("available_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("outbox_id"),
    )
    op.create_index("ix_outbox_status_available", "outbox_messages", ["status", "available_at"], unique=False)
    op.create_index("ix_outbox_aggregate_id", "outbox_messages", ["aggregate_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_outbox_aggregate_id", table_name="outbox_messages")
    op.drop_index("ix_outbox_status_available", table_name="outbox_messages")
    op.drop_table("outbox_messages")

    op.drop_index("ix_jobs_created_at", table_name="jobs")
    op.drop_index("ix_jobs_status", table_name="jobs")
    op.drop_table("jobs")

    bind = op.get_bind()
    outbox_status_enum.drop(bind, checkfirst=True)
    job_status_enum.drop(bind, checkfirst=True)
    job_type_enum.drop(bind, checkfirst=True)
