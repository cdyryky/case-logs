from __future__ import annotations

import json
import sqlite3
from typing import Any

from .candidates import load_acgme_targets
from .config_io import load_resident_profile
from .constants import DEFAULT_CASE_CLASS, ROOT
from .models import log_event, row_to_dict, utc_now
from .utils import case_year_from_date

EXPORTABLE_REVIEW = ("approved", "edited", "accepted_auto", "accepted_manual", "edited_manual")
EXPORTABLE_REVIEW_SQL = ",".join("?" for _ in EXPORTABLE_REVIEW)
AVAILABLE_UPLOAD = ("not_uploaded", "reset")
ACTIVE_UPLOAD = ("claimed", "autofilled")


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
        "acgme_code": row["acgme_code"] or "" if "acgme_code" in row.keys() else "",
        "area": row["area"],
        "type": row["type"],
        "acgme_description": row["acgme_description"] or "",
        "acgme_def_category": row["acgme_def_category"] or "",
        "keyword": row["keyword"] or "",
        "comments": row["comments"] or "",
        "component_label": row["component_label"],
        "mapping_rule_name": row["mapping_rule_name"],
        "mapping_confidence": row["mapping_confidence"],
        "mapping_pathway": row["mapping_pathway"] if "mapping_pathway" in row.keys() else "",
    }
    for key in ("exam_code", "procedure_text", "study_description", "attending_name"):
        if key in row.keys():
            payload[key] = row[key] or ""
    return payload


def payload_from_group(rows: list[sqlite3.Row]) -> dict[str, Any] | None:
    if not rows:
        return None
    first = rows[0]
    codes = [payload_from_entry(row) for row in rows]
    confidence = sorted({code["mapping_confidence"] for code in codes if code.get("mapping_confidence")})
    payload = {
        "source_case_id": first["source_case_id"],
        "case_id": first["case_id"],
        "case_date": first["case_date"],
        "case_year": first["case_year"],
        "role": first["role"],
        "site": first["site"],
        "patient_type": first["patient_type"],
        "mapping_confidence": "/".join(confidence),
        "codes": codes,
    }
    for key in ("exam_code", "procedure_text", "study_description", "attending_name"):
        if key in first.keys():
            payload[key] = first[key] or ""
    return payload


def entry_query(where: str) -> str:
    return f"""
        SELECT ge.*, sc.exam_code, sc.procedure_text, sc.study_description, sc.attending_name
        FROM generated_entries ge
        JOIN source_cases sc ON sc.id = ge.source_case_id
        WHERE {where}
    """


def group_query(where: str, order_by: str = "ge.id") -> str:
    return f"""
        SELECT ge.*, sc.exam_code, sc.procedure_text, sc.study_description, sc.attending_name
        FROM generated_entries ge
        JOIN source_cases sc ON sc.id = ge.source_case_id
        WHERE {where}
        ORDER BY {order_by}
    """


def _current_row(conn: sqlite3.Connection) -> sqlite3.Row | None:
    session = conn.execute("SELECT current_entry_id FROM upload_session WHERE id = 1").fetchone()
    if not session or not session["current_entry_id"]:
        return None
    return conn.execute(entry_query("ge.id = ?"), (session["current_entry_id"],)).fetchone()


def get_current(conn: sqlite3.Connection) -> dict[str, Any] | None:
    row = _current_row(conn)
    return payload_from_entry(row) if row else None


def _current_group_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    current = _current_row(conn)
    if not current or current["upload_status"] not in ACTIVE_UPLOAD:
        return []
    return conn.execute(
        group_query(
            f"""
            ge.source_case_id = ?
              AND ge.review_status IN ({EXPORTABLE_REVIEW_SQL})
              AND ge.upload_status IN (?, ?)
            """
        ),
        (current["source_case_id"], *EXPORTABLE_REVIEW, *ACTIVE_UPLOAD),
    ).fetchall()


def get_current_group(conn: sqlite3.Connection) -> dict[str, Any] | None:
    return payload_from_group(_current_group_rows(conn))


def claim_next(conn: sqlite3.Connection) -> dict[str, Any] | None:
    current = _current_row(conn)
    if current and current["upload_status"] in ("claimed", "autofilled"):
        return payload_from_entry(current)

    row = conn.execute(
        f"""
        SELECT ge.*, sc.exam_code, sc.procedure_text, sc.study_description, sc.attending_name
        FROM generated_entries ge
        JOIN source_cases sc ON sc.id = ge.source_case_id
        WHERE ge.review_status IN ({EXPORTABLE_REVIEW_SQL})
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


def claim_next_group(conn: sqlite3.Connection) -> dict[str, Any] | None:
    current = _current_group_rows(conn)
    if current:
        return payload_from_group(current)

    row = conn.execute(
        f"""
        SELECT ge.source_case_id, MIN(ge.id) AS first_entry_id
        FROM generated_entries ge
        WHERE ge.review_status IN ({EXPORTABLE_REVIEW_SQL})
          AND ge.upload_status IN (?, ?)
        GROUP BY ge.source_case_id
        ORDER BY first_entry_id
        LIMIT 1
        """,
        (*EXPORTABLE_REVIEW, *AVAILABLE_UPLOAD),
    ).fetchone()
    if not row:
        conn.execute("UPDATE upload_session SET current_entry_id = NULL, updated_at = ? WHERE id = 1", (utc_now(),))
        return None

    rows = conn.execute(
        group_query(
            f"""
            ge.source_case_id = ?
              AND ge.review_status IN ({EXPORTABLE_REVIEW_SQL})
              AND ge.upload_status IN (?, ?)
            """
        ),
        (row["source_case_id"], *EXPORTABLE_REVIEW, *AVAILABLE_UPLOAD),
    ).fetchall()
    if not rows:
        return None

    now = utc_now()
    for entry in rows:
        conn.execute(
            "UPDATE generated_entries SET upload_status = 'claimed', updated_at = ? WHERE id = ?",
            (now, entry["id"]),
        )
        log_event(conn, entry["id"], "claimed", entry["upload_status"], "claimed", "api")
    session = conn.execute("SELECT current_entry_id FROM upload_session WHERE id = 1").fetchone()
    previous = session["current_entry_id"] if session else None
    conn.execute(
        "UPDATE upload_session SET previous_entry_id = ?, current_entry_id = ?, updated_at = ? WHERE id = 1",
        (previous, rows[0]["id"], now),
    )
    return get_current_group(conn)


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


def update_group_upload_status(
    conn: sqlite3.Connection,
    source_case_id: int,
    status: str,
    event_type: str,
    failure_reason: str | None = None,
) -> dict[str, Any] | None:
    rows = conn.execute(
        group_query(
            f"""
            ge.source_case_id = ?
              AND ge.review_status IN ({EXPORTABLE_REVIEW_SQL})
              AND ge.upload_status IN (?, ?)
            """
        ),
        (source_case_id, *EXPORTABLE_REVIEW, *ACTIVE_UPLOAD),
    ).fetchall()
    if not rows:
        raise KeyError(f"Active queue case not found: {source_case_id}")

    now = utc_now()
    for row in rows:
        old = row["upload_status"]
        if status == "failed":
            conn.execute(
                """
                UPDATE generated_entries
                SET upload_status = ?, failure_reason = ?, failure_timestamp = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, failure_reason or "Unknown extension failure", now, now, row["id"]),
            )
        elif status == "submitted":
            conn.execute(
                "UPDATE generated_entries SET upload_status = ?, submitted_at = ?, updated_at = ? WHERE id = ?",
                (status, now, now, row["id"]),
            )
        else:
            conn.execute(
                "UPDATE generated_entries SET upload_status = ?, updated_at = ? WHERE id = ?",
                (status, now, row["id"]),
            )
        log_event(conn, row["id"], event_type, old, status, "api", failure_reason)

    if status in {"submitted", "skipped_upload_session", "failed", "reset"}:
        conn.execute(
            """
            UPDATE upload_session
            SET current_entry_id = NULL, updated_at = ?
            WHERE current_entry_id IN (
              SELECT id FROM generated_entries WHERE source_case_id = ?
            )
            """,
            (now, source_case_id),
        )
        return None
    return get_current_group(conn)


def _manual_dedupe_key(source: sqlite3.Row, case_date: str, code: dict[str, Any], component_label: str) -> str:
    return "|".join(
        [
            source["accession_number"],
            case_date,
            str(code.get("case_class") or DEFAULT_CASE_CLASS),
            str(code.get("acgme_code") or ""),
            str(code["area"]),
            str(code["type"]),
            str(code.get("acgme_description") or ""),
            component_label,
        ]
    )


def save_group_edit(conn: sqlite3.Connection, source_case_id: int, codes: list[dict[str, Any]]) -> dict[str, Any]:
    if not codes:
        raise ValueError("At least one code must be selected.")

    source = conn.execute("SELECT * FROM source_cases WHERE id = ?", (source_case_id,)).fetchone()
    if not source:
        raise KeyError(f"Source case not found: {source_case_id}")
    active_rows = conn.execute(
        group_query(
            f"""
            ge.source_case_id = ?
              AND ge.review_status IN ({EXPORTABLE_REVIEW_SQL})
              AND ge.upload_status IN (?, ?)
            """
        ),
        (source_case_id, *EXPORTABLE_REVIEW, *ACTIVE_UPLOAD),
    ).fetchall()
    base = active_rows[0] if active_rows else conn.execute(
        group_query("ge.source_case_id = ?", "ge.id LIMIT 1"), (source_case_id,)
    ).fetchone()
    if not base:
        raise KeyError(f"No generated entries found for source case: {source_case_id}")

    now = utc_now()
    selected_existing = {int(code["local_entry_id"]) for code in codes if code.get("local_entry_id")}
    for row in active_rows:
        if row["id"] not in selected_existing:
            conn.execute(
                "UPDATE generated_entries SET upload_status = 'skipped_upload_session', updated_at = ? WHERE id = ?",
                (now, row["id"]),
            )
            log_event(conn, row["id"], "edit_deselected", row["upload_status"], "skipped_upload_session", "api")

    for index, code in enumerate(codes, start=1):
        component_label = str(code.get("component_label") or f"manual_{index}")
        values = (
            str(code.get("case_class") or DEFAULT_CASE_CLASS),
            str(code.get("acgme_code") or ""),
            str(code["area"]),
            str(code["type"]),
            str(code.get("acgme_description") or ""),
            str(code.get("acgme_def_category") or ""),
            str(code.get("keyword") or ""),
            str(code.get("comments") or base["comments"] or ""),
            component_label,
            now,
        )
        if code.get("local_entry_id"):
            entry_id = int(code["local_entry_id"])
            conn.execute(
                """
                UPDATE generated_entries
            SET case_class = ?, acgme_code = ?, area = ?, type = ?, acgme_description = ?, acgme_def_category = ?,
                    keyword = ?, comments = ?, component_label = ?, review_status = 'edited',
                    upload_status = CASE
                      WHEN upload_status IN ('claimed', 'autofilled') THEN upload_status
                      ELSE 'claimed'
                    END,
                    updated_at = ?
                WHERE id = ? AND source_case_id = ?
                """,
                (*values, entry_id, source_case_id),
            )
            log_event(conn, entry_id, "edited", base["review_status"], "edited", "api")
            continue

        case_date = base["case_date"]
        dedupe_key = _manual_dedupe_key(source, case_date, code, component_label)
        profile = load_resident_profile()["resident"]
        conn.execute(
            """
            INSERT OR IGNORE INTO generated_entries(
              source_case_id, dedupe_key, component_label, case_id, case_date, case_year, role, site,
              patient_type, case_class, acgme_code, area, type, acgme_description, acgme_def_category, keyword, comments,
              mapping_rule_id, mapping_rule_version, mapping_rules_file_hash, mapping_rule_name,
              mapping_confidence, role_confidence, compound_flag, review_status, upload_status, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'manual', '1', 'manual',
                    'Popup edit', 'low', ?, 1, 'edited', 'claimed', ?, ?)
            """,
            (
                source_case_id,
                dedupe_key,
                component_label,
                base["case_id"],
                case_date,
                case_year_from_date(
                    source["study_date"],
                    int(profile["expected_graduation_year"]),
                    int(profile.get("pgy_max", 5)),
                ),
                base["role"],
                base["site"],
                base["patient_type"],
                str(code.get("case_class") or DEFAULT_CASE_CLASS),
                str(code.get("acgme_code") or ""),
                str(code["area"]),
                str(code["type"]),
                str(code.get("acgme_description") or ""),
                str(code.get("acgme_def_category") or ""),
                str(code.get("keyword") or ""),
                str(code.get("comments") or base["comments"] or ""),
                base["role_confidence"],
                now,
                now,
            ),
        )
        entry = conn.execute("SELECT id, upload_status FROM generated_entries WHERE dedupe_key = ?", (dedupe_key,)).fetchone()
        if entry and entry["upload_status"] not in ACTIVE_UPLOAD:
            conn.execute(
                "UPDATE generated_entries SET upload_status = 'claimed', review_status = 'edited', updated_at = ? WHERE id = ?",
                (now, entry["id"]),
            )
        if entry:
            log_event(conn, entry["id"], "edited_added", None, "edited", "api")

    selected = conn.execute(
        group_query(
            f"""
            ge.source_case_id = ?
              AND ge.review_status IN ({EXPORTABLE_REVIEW_SQL})
              AND ge.upload_status IN (?, ?)
            """
        ),
        (source_case_id, *EXPORTABLE_REVIEW, *ACTIVE_UPLOAD),
    ).fetchall()
    conn.execute(
        "UPDATE upload_session SET current_entry_id = ?, updated_at = ? WHERE id = 1",
        (selected[0]["id"], now),
    )
    return payload_from_group(selected)


def parse_acgme_options() -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    for target in load_acgme_targets():
        options.append(
            {
                "case_class": DEFAULT_CASE_CLASS,
                "acgme_code": target.acgme_code,
                "area": target.area,
                "type": target.type,
                "acgme_description": target.acgme_description,
                "acgme_def_category": target.acgme_def_category,
                "label": " | ".join(
                    part
                    for part in [
                        target.acgme_code,
                        target.acgme_description or target.type,
                        f"{target.area} / {target.type}",
                        f"Def Cat: {target.acgme_def_category}" if target.acgme_def_category else "",
                    ]
                    if part
                ),
            }
        )
    return sorted(options, key=lambda item: (item["area"], item["type"], item["acgme_description"], item["acgme_code"]))


def search_acgme_options(query: str = "", limit: int = 80) -> list[dict[str, str]]:
    terms = [part.casefold() for part in query.split() if part.strip()]
    matches = []
    for option in parse_acgme_options():
        haystack = " ".join(option.values()).casefold()
        if all(term in haystack for term in terms):
            matches.append(option)
        if len(matches) >= limit:
            break
    return matches


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
