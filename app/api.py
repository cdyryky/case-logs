from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .constants import DEFAULT_DB_PATH
from .models import connect, init_db
from .upload_queue import back, claim_next, get_current, update_upload_status

api = FastAPI(title="ACGME IR Case Log Local API")
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class FailurePayload(BaseModel):
    failure_reason: str | None = None


def _with_db(fn):
    conn = connect(DEFAULT_DB_PATH)
    try:
        init_db(conn)
        result = fn(conn)
        conn.commit()
        return result
    finally:
        conn.close()


@api.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@api.post("/queue/claim_next")
def api_claim_next() -> dict[str, Any]:
    item = _with_db(claim_next)
    return {"entry": item}


@api.get("/queue/current")
def api_current() -> dict[str, Any]:
    item = _with_db(get_current)
    return {"entry": item}


@api.post("/entries/{entry_id}/autofilled")
def api_autofilled(entry_id: int) -> dict[str, Any]:
    try:
        return {"entry": _with_db(lambda conn: update_upload_status(conn, entry_id, "autofilled", "autofilled"))}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.post("/entries/{entry_id}/submitted")
def api_submitted(entry_id: int) -> dict[str, Any]:
    try:
        return {"entry": _with_db(lambda conn: update_upload_status(conn, entry_id, "submitted", "submitted"))}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.post("/entries/{entry_id}/skip_upload")
def api_skip_upload(entry_id: int) -> dict[str, Any]:
    try:
        return {"entry": _with_db(lambda conn: update_upload_status(conn, entry_id, "skipped_upload_session", "skip_upload"))}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.post("/entries/{entry_id}/reset_upload")
def api_reset_upload(entry_id: int) -> dict[str, Any]:
    try:
        return {"entry": _with_db(lambda conn: update_upload_status(conn, entry_id, "reset", "reset"))}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.post("/entries/{entry_id}/failed")
def api_failed(entry_id: int, payload: FailurePayload) -> dict[str, Any]:
    try:
        return {
            "entry": _with_db(
                lambda conn: update_upload_status(conn, entry_id, "failed", "failed", payload.failure_reason)
            )
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.post("/session/back")
def api_back() -> dict[str, Any]:
    return {"entry": _with_db(back)}
