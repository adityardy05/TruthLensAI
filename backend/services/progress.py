"""Request-scoped progress callbacks used by LangGraph nodes."""
from contextvars import ContextVar
from typing import Callable

_callback: ContextVar[Callable[[dict], None] | None] = ContextVar(
    "truthlens_progress", default=None
)


def set_progress_callback(callback):
    return _callback.set(callback)


def reset_progress_callback(token):
    _callback.reset(token)


def report_progress(stage: str, message: str, status: str = "completed") -> None:
    callback = _callback.get()
    if callback:
        callback({"stage": stage, "message": message, "status": status})
