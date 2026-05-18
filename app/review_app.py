from __future__ import annotations

import json
import sqlite3
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.config_io import load_dropdowns
from app.config_io import load_resident_profile
from app.constants import DEFAULT_CASE_CLASS, DEFAULT_DB_PATH, DEFAULT_SITE
from app.candidates import add_manual_candidates, approve_candidate_review, search_acgme_targets
from app.export_payload import export_approved_json
from app.importer import import_mpower_csv
from app.llm_mapping import llm_health
from app.learning import append_learned_rule, apply_mapping_to_matching_unsubmitted, learned_rule_count
from app.models import connect, init_db, log_event, utc_now
from app.parser import parse_mpower_report
from app.review_queue import (
    best_report_context_for_source,
    clear_deterministic_review_state,
    import_scope_summary,
    latest_import,
    load_next_candidate_group,
    load_next_generated_group,
    load_next_unmapped,
    remap_unresolved_cases,
    review_counts,
    source_row_to_mapping_source,
)
from app.utils import case_year_from_date, canonical_key, format_acgme_date, patient_type

DEFAULT_MPOWER_CSV_PATH = "data/exports/mpower-download-260526-clean.csv"


def _new_import_job(path: str) -> dict[str, Any]:
    return {
        "path": path,
        "use_llm": True,
        "llm_timeout_seconds": 30.0,
        "running": True,
        "import_id": None,
        "phase": "Queued",
        "row": 0,
        "total": 0,
        "accession": "",
        "summary": None,
        "error": "",
        "traceback": "",
        "lock": threading.Lock(),
        "thread": None,
    }


def _run_background_import(job: dict[str, Any]) -> None:
    def update(values: dict[str, Any]) -> None:
        with job["lock"]:
            job.update(values)

    def on_progress(event: dict[str, Any]) -> None:
        update(
            {
                "row": int(event.get("row") or 0),
                "total": int(event.get("total") or 0),
                "accession": str(event.get("accession") or ""),
                "phase": str(event.get("phase") or "Working"),
                "import_id": event.get("import_id") or job.get("import_id"),
            }
        )

    conn = connect(DEFAULT_DB_PATH)
    try:
        init_db(conn)
        use_llm = bool(job.get("use_llm", True))
        llm_timeout_seconds = float(job.get("llm_timeout_seconds") or 30.0)
        summary = import_mpower_csv(
            conn,
            job["path"],
            progress_callback=on_progress,
            commit_per_row=True,
            mapping_mode="llm" if use_llm else "legacy",
            llm_timeout_seconds=llm_timeout_seconds,
        )
        conn.commit()
        update({"summary": summary, "phase": "Complete", "running": False})
    except Exception as exc:
        conn.rollback()
        update({"error": str(exc), "traceback": traceback.format_exc(), "phase": "Failed", "running": False})
    finally:
        conn.close()


def start_background_import(path: str, use_llm: bool = True, llm_timeout_seconds: float = 30.0) -> dict[str, Any]:
    job = _new_import_job(path)
    job["use_llm"] = use_llm
    job["llm_timeout_seconds"] = llm_timeout_seconds
    thread = threading.Thread(target=_run_background_import, args=(job,), daemon=True)
    job["thread"] = thread
    thread.start()
    return job


def import_job_snapshot(job: dict[str, Any] | None) -> dict[str, Any] | None:
    if not job:
        return None
    with job["lock"]:
        return {key: value for key, value in job.items() if key not in {"lock", "thread"}}


def get_conn() -> sqlite3.Connection:
    conn = connect(DEFAULT_DB_PATH)
    if st.session_state.get("_db_initialized"):
        return conn
    try:
        init_db(conn)
        st.session_state["_db_initialized"] = True
    except sqlite3.OperationalError as exc:
        if "database is locked" not in str(exc).lower():
            conn.close()
            raise
        conn.rollback()
        required = conn.execute(
            """
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type = 'table'
              AND name IN ('imports', 'source_cases', 'generated_entries', 'entry_events')
            """
        ).fetchone()[0]
        if required < 4:
            conn.close()
            raise
        st.session_state["_db_initialized"] = True
    return conn


def area_options() -> list[str]:
    dropdowns = load_dropdowns()
    return sorted({item["area"]["visible_label"] for item in dropdowns["case_options"]})


def type_options_for_area(area: str) -> list[str]:
    dropdowns = load_dropdowns()
    values = [
        item["type"]["visible_label"]
        for item in dropdowns["case_options"]
        if item["area"]["visible_label"] == area
    ]
    return sorted(set(values))


def index_or_zero(options: list[str], value: str | None) -> int:
    try:
        return options.index(value or "")
    except ValueError:
        return 0


def load_entries(conn: sqlite3.Connection, filter_name: str) -> pd.DataFrame:
    where = {
        "New / needs review": "review_status IN ('new_high_confidence', 'needs_review')",
        "Batch approvable": "review_status = 'new_high_confidence' AND mapping_confidence = 'high' AND role_confidence = 'high' AND compound_flag = 0",
        "Approved not uploaded": "review_status IN ('approved', 'edited') AND upload_status IN ('not_uploaded', 'reset')",
        "Autofilled not submitted": "upload_status = 'autofilled'",
        "Upload failures": "upload_status = 'failed'",
        "Submitted": "upload_status = 'submitted'",
    }[filter_name]
    df = pd.read_sql_query(
        f"""
        SELECT ge.id, ge.case_date, ge.case_id, ge.role, ge.patient_type, ge.area, ge.type,
               ge.acgme_description, ge.acgme_def_category,
               ge.component_label, ge.mapping_confidence, ge.role_confidence, ge.compound_flag,
               ge.review_status, ge.upload_status, ge.mapping_rule_name, ge.failure_reason,
               ge.evidence_excerpt, ge.llm_model, ge.llm_prompt_version,
               sc.exam_code, sc.procedure_text, sc.study_description, sc.attending_name, sc.parsed_report_json,
               sc.resident_found_in_report, sc.resident_position, sc.needs_review_reason
        FROM generated_entries ge
        JOIN source_cases sc ON sc.id = ge.source_case_id
        WHERE {where}
        ORDER BY ge.id
        """,
        conn,
    )
    return add_parsed_report_preview(df)


def parse_report_json(value: object) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def parsed_report_for_source(source: sqlite3.Row) -> dict[str, Any]:
    parsed = parse_report_json(source["parsed_report_json"] if "parsed_report_json" in source.keys() else None)
    if parsed:
        return parsed
    raw_text = source["report_snippet"] if "report_snippet" in source.keys() else ""
    if not raw_text:
        return {}
    profile = load_resident_profile()
    return parse_mpower_report(raw_text, profile["resident"].get("aliases", []))


def compact_lines(value: object, limit: int = 4) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        lines = [str(item).strip() for item in value if str(item).strip()]
    else:
        lines = [line.strip() for line in str(value).splitlines() if line.strip()]
    return " | ".join(lines[:limit])


def summary_preview(parsed: dict[str, Any], limit: int = 6) -> str:
    values: list[str] = []
    for section in parsed.get("procedure_summary_sections") or []:
        if not isinstance(section, dict):
            continue
        for line in section.get("bullets") or section.get("lines") or []:
            text = str(line).strip()
            if text and text.lower() != "additional procedure(s): none":
                values.append(text)
        for line in section.get("additional_procedures") or []:
            text = str(line).strip()
            if text:
                values.append(f"Additional: {text}")
    return " | ".join(list(dict.fromkeys(values))[:limit])


def candidate_preview(parsed: dict[str, Any], limit: int = 8) -> str:
    values = [str(item).strip() for item in parsed.get("candidate_procedure_phrases") or [] if str(item).strip()]
    return " | ".join(values[:limit])


def add_parsed_report_preview(df: pd.DataFrame) -> pd.DataFrame:
    if "parsed_report_json" not in df.columns or df.empty:
        return df
    out = df.copy()
    parsed_values = out["parsed_report_json"].map(parse_report_json)
    out["parsed_procedure_title"] = parsed_values.map(lambda p: p.get("procedure_title", ""))
    out["impression"] = parsed_values.map(lambda p: compact_lines(p.get("impression"), 4))
    out["procedure_summary"] = parsed_values.map(summary_preview)
    out["candidate_procedure_phrases"] = parsed_values.map(candidate_preview)
    out["parse_warnings"] = parsed_values.map(lambda p: ", ".join(p.get("parse_warnings") or []))
    out = out.drop(columns=["parsed_report_json"])
    return out


def update_review_status(conn: sqlite3.Connection, ids: list[int], status: str, source: str = "review_app") -> None:
    now = utc_now()
    for entry_id in ids:
        row = conn.execute("SELECT review_status FROM generated_entries WHERE id = ?", (entry_id,)).fetchone()
        if not row:
            continue
        old = row["review_status"]
        conn.execute(
            "UPDATE generated_entries SET review_status = ?, updated_at = ? WHERE id = ?",
            (status, now, entry_id),
        )
        log_event(conn, entry_id, status, old, status, source)
    conn.commit()


def reset_upload(conn: sqlite3.Connection, ids: list[int]) -> None:
    now = utc_now()
    for entry_id in ids:
        row = conn.execute("SELECT upload_status FROM generated_entries WHERE id = ?", (entry_id,)).fetchone()
        if not row:
            continue
        old = row["upload_status"]
        conn.execute(
            "UPDATE generated_entries SET upload_status = 'reset', failure_reason = NULL, failure_timestamp = NULL, updated_at = ? WHERE id = ?",
            (now, entry_id),
        )
        log_event(conn, entry_id, "reset", old, "reset", "review_app")
    conn.commit()


def mark_upload_submitted(conn: sqlite3.Connection, ids: list[int]) -> None:
    now = utc_now()
    for entry_id in ids:
        row = conn.execute("SELECT upload_status FROM generated_entries WHERE id = ?", (entry_id,)).fetchone()
        if not row:
            continue
        old = row["upload_status"]
        conn.execute(
            "UPDATE generated_entries SET upload_status = 'submitted', submitted_at = ?, updated_at = ? WHERE id = ?",
            (now, now, entry_id),
        )
        log_event(conn, entry_id, "submitted", old, "submitted", "review_app")
    conn.commit()


def save_entry_edit(conn: sqlite3.Connection, entry_id: int, values: dict[str, object]) -> None:
    row = conn.execute("SELECT review_status FROM generated_entries WHERE id = ?", (entry_id,)).fetchone()
    if not row:
        raise ValueError(f"Entry not found: {entry_id}")
    now = utc_now()
    conn.execute(
        """
        UPDATE generated_entries
        SET role = ?, site = ?, patient_type = ?, case_class = ?, area = ?, type = ?,
            acgme_description = ?, acgme_def_category = ?, keyword = ?, comments = ?,
            component_label = ?, review_status = 'edited', updated_at = ?
        WHERE id = ?
        """,
        (
            values["role"],
            values["site"],
            values["patient_type"],
            values["case_class"],
            values["area"],
            values["type"],
            values.get("acgme_description", ""),
            values.get("acgme_def_category", ""),
            values["keyword"],
            values["comments"],
            values["component_label"],
            now,
            entry_id,
        ),
    )
    log_event(conn, entry_id, "edited", row["review_status"], "edited", "review_app")
    conn.commit()


def learn_from_generated_entry(
    conn: sqlite3.Connection,
    entry_id: int,
    values: dict[str, object],
    apply_now: bool,
) -> tuple[str, int]:
    source = conn.execute(
        """
        SELECT sc.*
        FROM source_cases sc
        JOIN generated_entries ge ON ge.source_case_id = sc.id
        WHERE ge.id = ?
        """,
        (entry_id,),
    ).fetchone()
    if not source:
        raise ValueError(f"Source case not found for entry: {entry_id}")
    learned = append_learned_rule(source, values)
    applied = apply_mapping_to_matching_unsubmitted(conn, source, values, learned) if apply_now else 0
    return learned["rule_id"], applied


def create_manual_entry(conn: sqlite3.Connection, source_case_id: int, values: dict[str, object]) -> int:
    source = conn.execute("SELECT * FROM source_cases WHERE id = ?", (source_case_id,)).fetchone()
    if not source:
        raise ValueError(f"Source case not found: {source_case_id}")
    case_date = format_acgme_date(source["study_date"])
    component_label = str(values["component_label"] or "manual")
    dedupe_key = "|".join(
        [
            source["accession_number"],
            case_date,
            str(values["case_class"]),
            str(values["area"]),
            str(values["type"]),
            str(values.get("acgme_description") or ""),
            component_label,
        ]
    )
    now = utc_now()
    resident = load_resident_profile()["resident"]
    cur = conn.execute(
        """
        INSERT INTO generated_entries(
          source_case_id, dedupe_key, component_label, case_id, case_date, case_year, role, site,
          patient_type, case_class, acgme_code, area, type, acgme_description, acgme_def_category, keyword, comments, mapping_rule_id,
          mapping_rule_version, mapping_rules_file_hash, mapping_rule_name, mapping_confidence,
          role_confidence, compound_flag, review_status, upload_status, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'manual', '1', 'manual', 'Manual entry',
                'low', 'low', 0, 'edited', 'not_uploaded', ?, ?)
        """,
        (
            source_case_id,
            dedupe_key,
            component_label,
            source["accession_number"],
            case_date,
            case_year_from_date(
                source["study_date"],
                int(resident["expected_graduation_year"]),
                int(resident.get("pgy_max", 5)),
            ),
            values["role"],
            values["site"],
            values["patient_type"],
            values["case_class"],
            values.get("acgme_code", ""),
            values["area"],
            values["type"],
            values.get("acgme_description", ""),
            values.get("acgme_def_category", ""),
            values["keyword"],
            values["comments"],
            now,
            now,
        ),
    )
    entry_id = int(cur.lastrowid)
    conn.execute(
        "UPDATE source_cases SET source_mapping_status = 'manual_entry_created' WHERE id = ?",
        (source_case_id,),
    )
    log_event(conn, entry_id, "manual_created", None, "edited", "review_app")
    conn.commit()
    return entry_id


def learn_from_source_case(
    conn: sqlite3.Connection,
    source_case_id: int,
    values: dict[str, object],
    apply_now: bool,
) -> tuple[str, int]:
    source = conn.execute("SELECT * FROM source_cases WHERE id = ?", (source_case_id,)).fetchone()
    if not source:
        raise ValueError(f"Source case not found: {source_case_id}")
    learned = append_learned_rule(source, values)
    applied = apply_mapping_to_matching_unsubmitted(conn, source, values, learned) if apply_now else 0
    return learned["rule_id"], applied


def default_values_for_source(source: sqlite3.Row) -> dict[str, object]:
    mapping_source = source_row_to_mapping_source(source)
    derived = mapping_source["derived"]
    return {
        "role": derived["role"],
        "site": derived["site"],
        "patient_type": derived["patient_type"],
        "case_class": DEFAULT_CASE_CLASS,
        "area": "",
        "type": "",
        "acgme_description": "",
        "acgme_def_category": "",
        "keyword": "",
        "comments": source["needs_review_reason"] or "",
        "component_label": canonical_key(source["procedure_text"]) or "dominant_procedure",
    }


def source_context(conn: sqlite3.Connection, source: sqlite3.Row) -> None:
    mapping_source = source_row_to_mapping_source(source)
    derived = mapping_source["derived"]
    report_context = best_report_context_for_source(conn, source)
    parsed = report_context["parsed"] or parsed_report_for_source(source)
    st.subheader(f"{source['procedure_text'] or source['study_description'] or 'Unlabeled case'}")
    c1, c2, c3, c4 = st.columns(4)
    c1.caption("Date")
    c1.write(format_acgme_date(source["study_date"]))
    c2.caption("Accession")
    c2.code(source["accession_number"])
    c3.caption("Exam code")
    c3.code(source["exam_code"] or "-")
    c4.caption("Role")
    c4.write(f"{derived['role']} ({derived['role_confidence']})")
    st.caption(
        f"Study: {source['study_description'] or '-'} | Attending: {source['attending_name'] or '-'}"
    )
    if source["needs_review_reason"]:
        st.warning(source["needs_review_reason"])
    st.markdown("**Report context**")
    if not parsed:
        st.caption("Parsed report missing; warnings: parsed_report_missing")
    else:
        visible_warnings = [
            str(item)
            for item in parsed.get("parse_warnings") or []
            if item not in {"missing_impression", "missing_procedure_summary", "missing_personnel_section", "personnel_role_low_confidence"}
        ]
        if visible_warnings:
            st.caption(f"Parsed report warnings: {', '.join(visible_warnings)}")
    if parsed.get("procedure_title"):
        st.write(f"Procedure title: {parsed['procedure_title']}")
    if parsed.get("impression"):
        st.caption("Impression")
        st.text(parsed["impression"])
    summaries = parsed.get("procedure_summary_sections") or []
    if summaries:
        st.caption("Procedure summary")
        for section in summaries:
            heading = section.get("heading") or "PROCEDURE SUMMARY"
            lines = section.get("bullets") or section.get("lines") or []
            visible = [line for line in lines if str(line).strip()]
            if visible:
                st.write(f"{heading}:")
                st.markdown("\n".join(f"- {line}" for line in visible))
            additional = [line for line in section.get("additional_procedures") or [] if str(line).strip()]
            if additional:
                st.write("Additional procedures:")
                st.markdown("\n".join(f"- {line}" for line in additional))
    candidates = parsed.get("candidate_procedure_phrases") or []
    if candidates:
        st.caption("Candidate procedure phrases")
        st.markdown("\n".join(f"- {phrase}" for phrase in candidates[:20]))
    raw_text = report_context["raw_text"] or (source["report_snippet"] if "report_snippet" in source.keys() else "")
    if raw_text:
        with st.expander("Raw report text", expanded=False):
            st.text(raw_text)


def candidate_label(candidate: sqlite3.Row) -> str:
    return candidate["acgme_description"] or candidate["type"]


def candidate_metadata(candidate: sqlite3.Row) -> str:
    parts = []
    if "acgme_code" in candidate.keys() and candidate["acgme_code"]:
        parts.append(str(candidate["acgme_code"]))
    parts.append(f"{candidate['area']} / {candidate['type']}")
    if candidate["acgme_def_category"]:
        parts.append(f"Def Cat: {candidate['acgme_def_category']}")
    return " | ".join(parts)


def target_primary_label(target: dict[str, object]) -> str:
    return str(target.get("acgme_description") or target.get("type") or "ACGME target")


def target_metadata(target: dict[str, object]) -> str:
    parts = []
    if target.get("acgme_code"):
        parts.append(str(target["acgme_code"]))
    parts.append(f"{target['area']} / {target['type']}")
    if target.get("acgme_def_category"):
        parts.append(f"Def Cat: {target['acgme_def_category']}")
    return " | ".join(parts)


def _candidate_event_label(candidate: sqlite3.Row) -> str:
    if "event_label" in candidate.keys() and candidate["event_label"]:
        return str(candidate["event_label"])
    return "Other suggestions"


def _render_candidate_checks(rows: list[sqlite3.Row], default_checked: bool) -> set[int]:
    selected_ids: set[int] = set()
    grouped: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(_candidate_event_label(row), []).append(row)
    for event_label, event_rows in grouped.items():
        if event_label != "Other suggestions":
            st.caption(f"Procedure event: {event_label}")
        for row in event_rows:
            checked = st.checkbox(
                candidate_label(row),
                value=default_checked,
                key=f"candidate_checked_{row['id']}",
                help=row["evidence_snippet"] or row["match_reason"] or None,
            )
            st.caption(f"{candidate_metadata(row)} · {row['confidence']} {row['score']:.2f} - {row['match_reason'] or ''}")
            if checked:
                selected_ids.add(int(row["id"]))
    return selected_ids


def candidate_review_card(conn: sqlite3.Connection, source: sqlite3.Row, candidates: list[sqlite3.Row]) -> None:
    selected_ids: set[int] = set()
    initially_selected_ids = {
        int(row["id"])
        for row in candidates
        if int(row["user_checked"] if row["user_checked"] is not None else row["default_checked"])
    }
    selected = [
        row
        for row in candidates
        if int(row["id"]) in initially_selected_ids
    ]
    possible = [row for row in candidates if int(row["id"]) not in initially_selected_ids]

    st.markdown("**Selected mappings**")
    if not selected:
        st.caption("No mappings are currently selected.")
    selected_ids.update(_render_candidate_checks(selected, True))

    with st.expander("Possible matches", expanded=bool(possible)):
        if not possible:
            st.caption("No additional possible matches.")
        selected_ids.update(_render_candidate_checks(possible, False))

    st.markdown("**Search ACGME procedures**")
    query = st.text_input("Search by description, type, area, or def cat", key=f"acgme_search_{source['id']}")
    results = search_acgme_targets(query) if query else []
    checked_results: list[dict[str, object]] = []
    if results:
        for idx, target in enumerate(results[:10]):
            checked = st.checkbox(
                target_primary_label(target),
                value=False,
                key=f"acgme_search_result_{source['id']}_{idx}_{target.get('acgme_code', '')}_{target['area']}_{target['type']}_{target.get('acgme_description', '')}",
            )
            st.caption(target_metadata(target))
            if checked:
                checked_results.append(target)
    elif query:
        st.caption("No official ACGME targets matched that search.")

    approve_col, skip_col = st.columns(2)
    with approve_col:
        if st.button("Approve checked", type="primary", key=f"approve_candidates_{source['id']}"):
            if checked_results:
                selected_ids.update(add_manual_candidates(conn, int(source["id"]), checked_results))
            inserted = approve_candidate_review(conn, int(source["id"]), selected_ids)
            st.success(f"Approved {len(selected_ids)} mappings; created {inserted} new entries.")
            st.rerun()
    with skip_col:
        if st.button("Skip this case", key=f"skip_candidates_{source['id']}"):
            approve_candidate_review(conn, int(source["id"]), set())
            st.success("Skipped this case.")
            st.rerun()


def mapping_form(
    form_key: str,
    defaults: dict[str, object],
    submit_label: str,
    allow_learning: bool = True,
) -> tuple[bool, dict[str, object], bool, bool]:
    with st.form(form_key):
        role = st.selectbox("Role", ["Primary", "Secondary"], index=0 if defaults.get("role") == "Primary" else 1)
        patient = st.selectbox(
            "Patient Type",
            ["Adult", "Pediatric"],
            index=0 if defaults.get("patient_type") == "Adult" else 1,
        )
        site = st.text_input("Site", str(defaults.get("site") or DEFAULT_SITE))
        areas = area_options()
        default_area = str(defaults.get("area") or areas[0])
        area = st.selectbox("Area", areas, index=index_or_zero(areas, default_area))
        types = type_options_for_area(area)
        default_type = str(defaults.get("type") or (types[0] if types else ""))
        typ = st.selectbox("Type", types, index=index_or_zero(types, default_type))
        acgme_description = st.text_input("ACGME Description", str(defaults.get("acgme_description") or ""))
        acgme_def_category = st.text_input("Def Cat", str(defaults.get("acgme_def_category") or ""))
        component = st.text_input("Component label", str(defaults.get("component_label") or "dominant_procedure"))
        keyword = st.text_input("Keyword", str(defaults.get("keyword") or ""))
        comments = st.text_area("Comments", str(defaults.get("comments") or ""))
        learn = st.checkbox("Learn for future imports", value=allow_learning, disabled=not allow_learning)
        apply_now = st.checkbox("Apply to matching unsubmitted entries now", value=allow_learning, disabled=not allow_learning)
        submitted = st.form_submit_button(submit_label, type="primary")
    values = {
        "role": role,
        "site": site,
        "patient_type": patient,
        "case_class": DEFAULT_CASE_CLASS,
        "area": area,
        "type": typ,
        "acgme_description": acgme_description,
        "acgme_def_category": acgme_def_category,
        "component_label": component,
        "keyword": keyword,
        "comments": comments,
    }
    return submitted, values, learn, apply_now


def mark_before_date(conn: sqlite3.Connection, cutoff: str, mode: str) -> int:
    cutoff_parts = [int(part) for part in cutoff.split("-")]
    cutoff_key = tuple(cutoff_parts)
    rows = conn.execute("SELECT id, case_date, review_status, upload_status FROM generated_entries").fetchall()
    changed = 0
    now = utc_now()
    for row in rows:
        month, day, year = [int(part) for part in row["case_date"].split("/")]
        if (year, month, day) >= cutoff_key:
            continue
        if mode == "submitted":
            conn.execute(
                "UPDATE generated_entries SET upload_status = 'submitted', submitted_at = ?, updated_at = ? WHERE id = ?",
                (now, now, row["id"]),
            )
            log_event(conn, row["id"], "baseline_submitted", row["upload_status"], "submitted", "review_app", cutoff)
        else:
            conn.execute(
                "UPDATE generated_entries SET review_status = 'skipped', updated_at = ? WHERE id = ?",
                (now, row["id"]),
            )
            log_event(conn, row["id"], "baseline_skipped", row["review_status"], "skipped", "review_app", cutoff)
        changed += 1
    conn.commit()
    return changed


def load_unmapped(conn: sqlite3.Connection, import_id: int | None = None) -> pd.DataFrame:
    scope = (
        "AND EXISTS (SELECT 1 FROM import_source_cases isc WHERE isc.source_case_id = source_cases.id AND isc.import_id = ?)"
        if import_id is not None
        else ""
    )
    params = (import_id,) if import_id is not None else ()
    df = pd.read_sql_query(
        f"""
        SELECT id, accession_number, study_date, exam_code, procedure_text, study_description,
               source_mapping_status, needs_review_reason, attending_name, parsed_report_json
        FROM source_cases
        WHERE source_mapping_status IN ('unmapped', 'flag_only', 'llm_failed_unmapped')
          {scope}
        ORDER BY study_date DESC
        LIMIT 1000
        """,
        conn,
        params=params,
    )
    return add_parsed_report_preview(df)


st.set_page_config(page_title="ACGME IR Case Logs", layout="wide")
st.title("ACGME IR Case Log Review")
st.caption(f"Learned mapping correction rules: {learned_rule_count()}")

conn = get_conn()

with st.sidebar:
    st.header("Local LLM")
    health = llm_health()
    st.caption(f"Model: {health['model']}")
    st.caption(f"Ollama: {health['base_url']} · ctx {health['num_ctx']}")
    if health["reachable"] and health["model_available"]:
        st.success("LLM reachable")
    elif health["reachable"]:
        st.warning("Ollama reachable; configured model not installed")
    else:
        st.warning("Ollama not reachable; imports will use legacy fallback")

    st.header("Import")
    import_path = st.text_input("Default mPower CSV", DEFAULT_MPOWER_CSV_PATH)
    uploaded = st.file_uploader("Optional alternate import file", type=["csv"])
    use_llm_for_import = st.toggle("Use local LLM for import mapping", value=True)
    llm_timeout_seconds = st.number_input(
        "LLM timeout per case (seconds)",
        min_value=5,
        max_value=600,
        value=int(float(health.get("timeout_seconds") or 30)),
        step=5,
        disabled=not use_llm_for_import,
    )
    current_import = import_job_snapshot(st.session_state.get("import_job"))
    import_running = bool(current_import and current_import["running"])
    if st.button("Import mPower CSV", type="primary", disabled=import_running):
        path: str | Path
        if uploaded:
            tmp = Path("data/imports") / uploaded.name
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(uploaded.getbuffer())
            path = tmp
        else:
            path = import_path
        st.session_state["import_job"] = start_background_import(
            str(path),
            use_llm=use_llm_for_import,
            llm_timeout_seconds=float(llm_timeout_seconds),
        )
        st.rerun()

    current_import = import_job_snapshot(st.session_state.get("import_job"))
    if current_import:
        total = max(int(current_import.get("total") or 1), 1)
        row = min(int(current_import.get("row") or 0), total)
        phase = str(current_import.get("phase") or "Working")
        accession = str(current_import.get("accession") or "-")
        progress_label = f"{row}/{total} · {accession} · {phase}"
        st.progress(row / total, text=progress_label)
        if current_import.get("running"):
            st.info("Import is running in the background. Newly mapped cases are available in the review queue as they finish.")
            if st.button("Refresh import status"):
                st.rerun()
        elif current_import.get("error"):
            st.error(f"Import failed: {current_import['error']}")
            with st.expander("Import error details"):
                st.code(str(current_import.get("traceback") or ""))
            if st.button("Clear failed import"):
                st.session_state.pop("import_job", None)
                st.rerun()
        else:
            summary = current_import.get("summary") or {}
            st.success(
                "Imported {row_count} rows; generated {generated_entries_count} entries.".format(
                    row_count=summary.get("row_count", row),
                    generated_entries_count=summary.get("generated_entries_count", 0),
                )
            )
            if st.button("Clear completed import"):
                st.session_state.pop("import_job", None)
                st.rerun()

    latest = latest_import(conn)
    running_import_id = int(current_import["import_id"]) if current_import and current_import.get("import_id") else None
    latest_import_id = running_import_id or (int(latest["id"]) if latest else None)
    st.header("Review scope")
    scope_choice = st.radio(
        "Scope",
        ["Latest import", "All backlog"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )
    active_import_id = latest_import_id if scope_choice == "Latest import" else None
    scope = import_scope_summary(conn, active_import_id)
    if active_import_id is None:
        st.caption("Reviewing all historical backlog")
    else:
        st.caption(
            "Reviewing import #{import_id} · {mapped}/{scoped} cases mapped · {rows} rows processed".format(
                import_id=active_import_id,
                mapped=scope.get("mapped_source_cases", 0),
                scoped=scope.get("scoped_source_cases", 0),
                rows=scope.get("row_count", 0),
            )
        )

    if import_running:
        @st.fragment(run_every="3s")
        def auto_refresh_import() -> None:
            now = time.monotonic()
            last = float(st.session_state.get("last_import_auto_refresh", now))
            st.session_state["last_import_auto_refresh"] = now
            if now - last >= 2.5:
                st.rerun(scope="app")

        auto_refresh_import()

    st.header("Export")
    if st.button("Export approved JSON"):
        out = export_approved_json(conn)
        st.success(f"Wrote {out}")

    st.header("Remap")
    if st.button("Remap unresolved cases"):
        with st.spinner("Applying improved matching to unresolved cases..."):
            summary = remap_unresolved_cases(conn)
        st.success(
            "Checked {checked}; generated {generated_entries} entries across {generated_cases} cases; "
            "{suggested_only} have suggestions only.".format(**summary)
        )
        st.rerun()

    st.header("Reset")
    reset_label = "this import" if active_import_id is not None else "all backlog"
    confirm_reset = st.checkbox(f"Confirm reset for {reset_label}", key="confirm_clear_non_llm_review")
    if st.button("Clear non-LLM review cases", disabled=not confirm_reset):
        with st.spinner("Clearing deterministic review state..."):
            summary = clear_deterministic_review_state(conn, active_import_id)
        st.success(
            "Cleared {deleted_candidates} candidate rows; skipped {skipped_entries} generated non-LLM entries; "
            "reset {reset_sources} cases.".format(**summary)
        )
        st.rerun()

tab_review, tab_diagnostics, tab_imports = st.tabs(["Review Queue", "Diagnostics", "Imports"])

with tab_review:
    counts = review_counts(conn, active_import_id)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Batch approvable", counts["batch_approvable"])
    m2.metric("Needs review", counts["needs_review"])
    m3.metric("Unmapped", counts["unmapped_total"])
    m4.metric("Upload failures", counts["upload_failures"])

    qc1, qc2 = st.columns([2, 1])
    with qc1:
        queue = st.radio(
            "Queue",
            ["Needs review", "Unmapped with suggestions", "Unmapped without suggestions", "Upload failures"],
            horizontal=True,
        )
    with qc2:
        if st.button("Approve all safe high-confidence", disabled=counts["batch_approvable"] == 0):
            ids = pd.read_sql_query(
                """
                SELECT id FROM generated_entries
                WHERE review_status = 'new_high_confidence'
                  AND mapping_confidence = 'high'
                  AND role_confidence = 'high'
                  AND compound_flag = 0
                  AND upload_status IN ('not_uploaded', 'reset')
                """,
                conn,
            )["id"].astype(int).tolist()
            update_review_status(conn, ids, "approved")
            st.success(f"Approved {len(ids)} entries.")
            st.rerun()

    if queue == "Needs review":
        source, candidates = load_next_candidate_group(conn, active_import_id)
        if not source:
            if active_import_id is not None and import_running:
                st.info("Waiting for LLM-mapped cases from this import.")
            elif active_import_id is not None:
                st.info("No reviewable cases in this import yet. Use All backlog to review historical cases.")
            else:
                st.info("No candidate mappings need review.")
        else:
            source_context(conn, source)
            candidate_review_card(conn, source, candidates)

            legacy_source, entries = load_next_generated_group(conn, active_import_id)
            if legacy_source and int(legacy_source["id"]) == int(source["id"]) and entries:
                generated_rows_are_llm = any((row["mapping_rule_id"] or "").startswith("llm:") for row in entries)
                with st.expander("Generated entries", expanded=generated_rows_are_llm):
                    rows = [
                        {
                            "id": row["id"],
                            "source": row["mapping_rule_name"],
                            "role": row["role"],
                            "area": row["area"],
                            "type": row["type"],
                            "description": row["acgme_description"],
                            "def_cat": row["acgme_def_category"],
                            "confidence": row["mapping_confidence"],
                            "reason": row["comments"] or row["mapping_rule_name"],
                            "evidence": row["evidence_excerpt"] if "evidence_excerpt" in row.keys() else "",
                            "llm_model": row["llm_model"] if "llm_model" in row.keys() else "",
                        }
                        for row in entries
                    ]
                    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                    generated_ids = [int(row["id"]) for row in entries]
                    approve_col, skip_col = st.columns(2)
                    with approve_col:
                        if st.button("Approve generated entries", type="primary", key=f"approve_generated_{source['id']}"):
                            update_review_status(conn, generated_ids, "approved")
                            st.rerun()
                    with skip_col:
                        if st.button("Skip generated entries", key=f"skip_generated_{source['id']}"):
                            update_review_status(conn, generated_ids, "skipped")
                            st.rerun()

    elif queue in {"Unmapped with suggestions", "Unmapped without suggestions"}:
        source, suggestions = load_next_unmapped(
            conn,
            require_suggestion=queue == "Unmapped with suggestions",
            import_id=active_import_id,
        )
        if not source:
            st.info("No source cases in this queue.")
        else:
            source_context(conn, source)
            defaults = default_values_for_source(source)
            if suggestions:
                st.caption("Suggested ACGME targets")
                suggestion_labels = [
                    f"{idx + 1}. {item['area']} / {item['type']} ({item['confidence']}, {item['score']:.3f})"
                    for idx, item in enumerate(suggestions)
                ]
                selected_suggestion = st.radio("Suggestion", suggestion_labels, label_visibility="collapsed")
                suggestion = suggestions[suggestion_labels.index(selected_suggestion)]
                defaults.update(
                    {
                        "area": suggestion["area"],
                        "type": suggestion["type"],
                        "acgme_description": suggestion["acgme_description"],
                        "acgme_def_category": suggestion["acgme_def_category"],
                        "keyword": suggestion["keyword"],
                        "component_label": suggestion["component_label"],
                        "comments": suggestion["reason"],
                    }
                )
                if st.button("Use suggestion", type="primary"):
                    entry_id = create_manual_entry(conn, int(source["id"]), defaults)
                    rule_id, applied = learn_from_source_case(conn, int(source["id"]), defaults, True)
                    st.success(f"Created entry {entry_id}. Learned `{rule_id}`; applied to {applied} matching entries.")
                    st.rerun()

            with st.expander("Edit"):
                submitted, values, learn, apply_now = mapping_form("unmapped_edit_form", defaults, "Create entry")
                if submitted:
                    entry_id = create_manual_entry(conn, int(source["id"]), values)
                    message = f"Created entry {entry_id}."
                    if learn:
                        rule_id, applied = learn_from_source_case(conn, int(source["id"]), values, apply_now)
                        message += f" Learned `{rule_id}`; applied to {applied} matching entries."
                    st.success(message)
                    st.rerun()
            if st.button("Skip"):
                conn.execute(
                    "UPDATE source_cases SET source_mapping_status = 'excluded', needs_review_reason = COALESCE(needs_review_reason, 'Skipped in review') WHERE id = ?",
                    (source["id"],),
                )
                conn.commit()
                st.rerun()

    else:
        failures = load_entries(conn, "Upload failures")
        st.dataframe(failures, width="stretch", hide_index=True)
        selected_raw = st.text_input("Failed entry IDs, comma-separated")
        selected_ids = [int(x.strip()) for x in selected_raw.split(",") if x.strip().isdigit()]
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Reset selected", disabled=not selected_ids):
                reset_upload(conn, selected_ids)
                st.rerun()
        with c2:
            if st.button("Mark selected submitted", disabled=not selected_ids):
                mark_upload_submitted(conn, selected_ids)
                st.rerun()

with tab_diagnostics:
    filter_name = st.selectbox(
        "Generated entries",
        [
            "New / needs review",
            "Batch approvable",
            "Approved not uploaded",
            "Autofilled not submitted",
            "Upload failures",
            "Submitted",
        ],
    )
    df = load_entries(conn, filter_name)
    st.caption(f"{len(df)} generated entries")
    st.dataframe(df, width="stretch", hide_index=True)
    unmapped = load_unmapped(conn, active_import_id)
    st.caption("Unmapped and flag-only source cases")
    st.dataframe(unmapped, width="stretch", hide_index=True)

with tab_imports:
    imports = pd.read_sql_query("SELECT * FROM imports ORDER BY imported_at DESC", conn)
    st.dataframe(imports, width="stretch", hide_index=True)
    with st.expander("Baseline reconciliation"):
        cutoff = st.date_input("Apply to generated entries before date")
        mode = st.selectbox("Baseline action", ["submitted", "skipped"])
        if st.button("Apply baseline action"):
            changed = mark_before_date(conn, cutoff.isoformat(), mode)
            st.success(f"Updated {changed} entries.")

conn.close()
