"""Thread-safe in-memory progress jobs for the Flask verification UI.

The verification pipeline runs in a worker thread. The browser polls this
store while that work is in progress; callbacks are request/job scoped.
"""
from __future__ import annotations

from contextvars import ContextVar
from threading import RLock
from typing import Callable
from uuid import uuid4

_callback: ContextVar[Callable[[dict], None] | None] = ContextVar(
    "truthlens_progress", default=None
)
_jobs: dict[str, dict] = {}
_lock = RLock()


def set_progress_callback(callback):
    return _callback.set(callback)


def reset_progress_callback(token):
    _callback.reset(token)


def report_progress(stage: str, message: str, status: str = "completed") -> None:
    callback = _callback.get()
    if callback:
        callback({"stage": stage, "message": message, "status": status})


def create_job() -> str:
    job_id = uuid4().hex
    with _lock:
        _jobs[job_id] = {
            "status": "running",
            "events": [],
            "result": None,
            "error": None,
        }
    return job_id


def append_job_event(job_id: str, event: dict) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            job["events"].append(dict(event, sequence=len(job["events"]) + 1))


def finish_job(job_id: str, result: dict | None = None, error: str | None = None) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            job["status"] = "failed" if error else "completed"
            job["result"] = result
            job["error"] = error


def get_job(job_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        return {
            "status": job["status"],
            "events": [dict(event) for event in job["events"]],
            "result": job["result"],
            "error": job["error"],
        }
