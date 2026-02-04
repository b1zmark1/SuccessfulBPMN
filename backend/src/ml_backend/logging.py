import logging
from contextvars import ContextVar
from logging.config import dictConfig

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = request_id_ctx.get()
        if not hasattr(record, "job_id"):
            record.job_id = "-"
        return True


def configure_logging(log_level: str) -> None:
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {
                "request_context": {
                    "()": "ml_backend.logging.RequestContextFilter",
                }
            },
            "formatters": {
                "json": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "fmt": "%(asctime)s %(levelname)s %(name)s %(message)s %(job_id)s %(request_id)s",
                }
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": "json",
                    "level": log_level,
                    "filters": ["request_context"],
                }
            },
            "root": {"handlers": ["default"], "level": log_level},
        }
    )
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
