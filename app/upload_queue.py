from __future__ import annotations

import sqlite3
from typing import Any

from .models import log_event, row_to_dict, utc_now

EXPORTABLE_REVIEW = ("approved", "edited")
AVAILABLE_UPLOAD = ("not_uploaded", "reset")


def payload_from_entry(row: sqlite3.Row) -> dict[str, Any]:
    payload = {
        "local_entry_id": row["id"],
        "case_id": row["case_id"],
        "case_date": row["case_date"],
        "case_year": row["case_year"],
        "role": row["role"],
        "site": row["site"],
        "patient_type": row["patient_type"],
        "case_class": row["case_class"],
        "area": row["area"],
        "type": row["type"],
        "acgme_description": row["acgme_description"] or "",
        "acgme_def_category": row["acgme_def_category"] or "",
        "keyword": row["keyword"] or "",
        "comments": row["comments"] or "",
        "component_label": row["component_label"],
        "mapping_rule_name": row["mapping_rule_name"],
        "mapping_confidence": row["mapping_confidence"],
    }
    for key in ("exam_code", "procedure_text", "study_description", "attending_name"):
        if key in row.keys():
            payload[key] = row[key] or ""
    return payload


def entry_query(where: str) -> str:
    return f"""
        SELECT ge.*, sc.exam_code, sc.procedure_text, sc.study_description, sc.attending_name
        FROM generated_entries ge
        JOIN source_cases sc ON sc.id = ge.source_case_id
        WHERE {where}
    """


def _current_row(conn: sqlite3.Connection) -> sqlite3.Row | None:
    session = conn.execute("SELECT current_entry_id FROM upload_session WHERE id = 1").fetchone()
    if not session or not session["current_entry_id"]:
        return None
    return conn.execute(entry_query("ge.id = ?"), (session["current_entry_id"],)).fetchone()


def get_current(conn: sqlite3.Connection) -> dict[str, Any] | None:
    row = _current_row(conn)
    return payload_from_entry(row) if row else None


def claim_next(conn: sqlite3.Connection) -> dict[str, Any] | None:
    current = _current_row(conn)
    if current and current["upload_status"] in ("claimed", "autofilled"):
        return payload_from_entry(current)

    row = conn.execute(
        """
        SELECT ge.*, sc.exam_code, sc.procedure_text, sc.study_description, sc.attending_name
        FROM generated_entries ge
        JOIN source_cases sc ON sc.id = ge.source_case_id
        WHERE ge.review_status IN (?, ?)
          AND ge.upload_status IN (?, ?)
        ORDER BY id
        LIMIT 1
        """,
        (*EXPORTABLE_REVIEW, *AVAILABLE_UPLOAD),
    ).fetchone()
    if not row:
        conn.execute("UPDATE upload_session SET current_entry_id = NULL, updated_at = ? WHERE id = 1", (utc_now(),))
        return None
    old = row["upload_status"]
    conn.execute(
        "UPDATE generated_entries SET upload_status = 'claimed', updated_at = ? WHERE id = ?",
        (utc_now(), row["id"]),
    )
    session = conn.execute("SELECT current_entry_id FROM upload_session WHERE id = 1").fetchone()
    previous = session["current_entry_id"] if session else None
    conn.execute(
        "UPDATE upload_session SET previous_entry_id = ?, current_entry_id = ?, updated_at = ? WHERE id = 1",
        (previous, row["id"], utc_now()),
    )
    log_event(conn, row["id"], "claimed", old, "claimed", "api")
    return payload_from_entry(conn.execute(entry_query("ge.id = ?"), (row["id"],)).fetchone())


def update_upload_status(
    conn: sqlite3.Connection,
    entry_id: int,
    status: str,
    event_type: str,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM generated_entries WHERE id = ?", (entry_id,)).fetchone()
    if not row:
        raise KeyError(f"Entry not found: {entry_id}")
    old = row["upload_status"]
    timestamp = utc_now()
    if status == "failed":
        conn.execute(
            """
            UPDATE generated_entries
            SET upload_status = ?, failure_reason = ?, failure_timestamp = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, failure_reason or "Unknown extension failure", timestamp, timestamp, entry_id),
        )
    elif status == "submitted":
        conn.execute(
            "UPDATE generated_entries SET upload_status = ?, submitted_at = ?, updated_at = ? WHERE id = ?",
            (status, timestamp, timestamp, entry_id),
        )
    elif status == "reset":
        conn.execute(
            """
            UPDATE generated_entries
            SET upload_status = 'reset', failure_reason = NULL, failure_timestamp = NULL, updated_at = ?
            WHERE id = ?
            """,
            (timestamp, entry_id),
        )
    else:
        conn.execute(
            "UPDATE generated_entries SET upload_status = ?, updated_at = ? WHERE id = ?",
            (status, timestamp, entry_id),
        )
    if status in {"submitted", "skipped_upload_session", "failed", "reset"}:
        conn.execute(
            "UPDATE upload_session SET current_entry_id = NULL, updated_at = ? WHERE current_entry_id = ?",
            (timestamp, entry_id),
        )
    log_event(conn, entry_id, event_type, old, status, "api", failure_reason)
    return row_to_dict(conn.execute("SELECT * FROM generated_entries WHERE id = ?", (entry_id,)).fetchone())


def back(conn: sqlite3.Connection) -> dict[str, Any] | None:
    session = conn.execute("SELECT current_entry_id, previous_entry_id FROM upload_session WHERE id = 1").fetchone()
    if not session or not session["previous_entry_id"]:
        return None
    current_id = session["current_entry_id"]
    previous_id = session["previous_entry_id"]
    now = utc_now()
    if current_id:
        old = conn.execute("SELECT upload_status FROM generated_entries WHERE id = ?", (current_id,)).fetchone()
        if old and old["upload_status"] in ("claimed", "autofilled"):
            conn.execute("UPDATE generated_entries SET upload_status = 'reset', updated_at = ? WHERE id = ?", (now, current_id))
            log_event(conn, current_id, "back_reset", old["upload_status"], "reset", "api")
    conn.execute(
        "UPDATE generated_entries SET upload_status = 'claimed', updated_at = ? WHERE id = ?",
        (now, previous_id),
    )
    conn.execute(
        "UPDATE upload_session SET current_entry_id = ?, previous_entry_id = NULL, updated_at = ? WHERE id = 1",
        (previous_id, now),
    )
    log_event(conn, previous_id, "back_claimed", None, "claimed", "api")
    row = conn.execute(entry_query("ge.id = ?"), (previous_id,)).fetchone()
    return payload_from_entry(row)
