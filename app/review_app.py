from __future__ import annotations

import json
import html
import sqlite3
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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
    load_next_failed_upload_group,
    load_next_generated_group,
    load_next_high_confidence_group,
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


def install_review_css() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] {background: transparent; border: 0; padding: 0;}
        div[data-testid="stMetricValue"] {font-size: 1.05rem;}
        .review-meta {
            border: 1px solid #e6e8eb;
            border-radius: 8px;
            padding: 0.65rem 0.75rem;
            margin: 0.35rem 0 0.75rem 0;
            background: #fbfbfc;
        }
        .review-meta-title {
            font-weight: 700;
            font-size: 1.18rem;
            line-height: 1.25;
            margin-bottom: 0.35rem;
            color: #252735;
        }
        .review-meta-grid {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 0.35rem 0.7rem;
            font-size: 0.84rem;
        }
        .review-meta-label {
            color: #7a7f8c;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0;
        }
        .review-chip {
            display: inline-block;
            border: 1px solid #d9dde3;
            border-radius: 8px;
            padding: 0.16rem 0.42rem;
            margin: 0.08rem 0.18rem 0.08rem 0;
            background: #fff;
            font-size: 0.78rem;
            color: #303342;
        }
        .review-warning {
            border-color: #f1d58a;
            background: #fff9e6;
            color: #7b5a00;
        }
        .procedure-row {
            border-top: 1px solid #edf0f2;
            padding: 0.35rem 0 0.3rem 0;
        }
        .procedure-meta {
            color: #737985;
            font-size: 0.78rem;
            line-height: 1.3;
        }
        .pane-label {
            color: #6f7480;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0;
            margin-top: 0.2rem;
        }
        .review-actions {
            border-top: 1px solid #e6e8eb;
            padding-top: 0.65rem;
            margin-top: 0.75rem;
        }
        section[data-testid="stSidebar"] .stButton button {width: 100%;}
        @media (max-width: 1100px) {
            .review-meta-grid {grid-template-columns: repeat(2, minmax(0, 1fr));}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def scroll_to_top_on_next_render() -> None:
    st.session_state["_scroll_to_top"] = True


def maybe_scroll_to_top() -> None:
    if st.session_state.pop("_scroll_to_top", False):
        components.html(
            """
            <script>
            const root = window.parent || window;
            root.scrollTo({top: 0, left: 0, behavior: "auto"});
            </script>
            """,
            height=0,
        )


def source_title(source: sqlite3.Row) -> str:
    return str(source["procedure_text"] or source["study_description"] or "Unlabeled case")


def compact_case_header(conn: sqlite3.Connection, source: sqlite3.Row, *, status: str = "") -> dict[str, Any]:
    mapping_source = source_row_to_mapping_source(source)
    derived = mapping_source["derived"]
    report_context = best_report_context_for_source(conn, source)
    parsed = report_context["parsed"] or parsed_report_for_source(source)
    visible_warnings = []
    if parsed:
        visible_warnings = [
            str(item)
            for item in parsed.get("parse_warnings") or []
            if item not in {"missing_impression", "missing_procedure_summary", "missing_personnel_section", "personnel_role_low_confidence"}
        ]
    if source["needs_review_reason"]:
        visible_warnings.append(str(source["needs_review_reason"]))
    chips = []
    if status:
        chips.append(f"<span class='review-chip'>{html.escape(status)}</span>")
    chips.extend(f"<span class='review-chip review-warning'>{html.escape(warning)}</span>" for warning in visible_warnings[:3])
    st.markdown(
        """
        <div class="review-meta">
          <div class="review-meta-title">{title}</div>
          <div class="review-meta-grid">
            <div><div class="review-meta-label">Date</div><div>{date}</div></div>
            <div><div class="review-meta-label">Accession</div><div>{accession}</div></div>
            <div><div class="review-meta-label">Exam</div><div>{exam}</div></div>
            <div><div class="review-meta-label">Role</div><div>{role} ({role_conf})</div></div>
            <div><div class="review-meta-label">Study</div><div>{study}</div></div>
            <div><div class="review-meta-label">Attending</div><div>{attending}</div></div>
          </div>
          <div>{chips}</div>
        </div>
        """.format(
            title=html.escape(source_title(source)),
            date=html.escape(format_acgme_date(source["study_date"])),
            accession=html.escape(str(source["accession_number"])),
            exam=html.escape(str(source["exam_code"] or "-")),
            role=html.escape(str(derived["role"])),
            role_conf=html.escape(str(derived["role_confidence"])),
            study=html.escape(str(source["study_description"] or "-")),
            attending=html.escape(str(source["attending_name"] or "-")),
            chips="".join(chips),
        ),
        unsafe_allow_html=True,
    )
    return {"derived": derived, "report_context": report_context, "parsed": parsed}


def render_report_evidence(report_context: dict[str, Any], source: sqlite3.Row) -> None:
    parsed = report_context["parsed"] or parsed_report_for_source(source)
    st.markdown("<div class='pane-label'>Report evidence</div>", unsafe_allow_html=True)
    if not parsed:
        st.caption("Parsed report missing.")
    elif parsed.get("procedure_title"):
        st.write(f"**Procedure title:** {parsed['procedure_title']}")
    if parsed.get("impression"):
        st.markdown("**Impression**")
        st.text(str(parsed["impression"]))
    summaries = parsed.get("procedure_summary_sections") or []
    if summaries:
        st.markdown("**Procedure summary**")
        for section in summaries:
            heading = section.get("heading") or "PROCEDURE SUMMARY"
            lines = [str(line).strip() for line in (section.get("bullets") or section.get("lines") or []) if str(line).strip()]
            additional = [str(line).strip() for line in section.get("additional_procedures") or [] if str(line).strip()]
            if lines:
                st.caption(str(heading))
                st.markdown("\n".join(f"- {line}" for line in lines))
            if additional:
                st.caption("Additional procedures")
                st.markdown("\n".join(f"- {line}" for line in additional))
    phrases = [str(item).strip() for item in parsed.get("candidate_procedure_phrases") or [] if str(item).strip()]
    if phrases:
        with st.expander("Parsed candidate phrases", expanded=False):
            st.markdown("\n".join(f"- {phrase}" for phrase in phrases[:20]))
    raw_text = report_context["raw_text"] or (source["report_snippet"] if "report_snippet" in source.keys() else "")
    if raw_text:
        with st.expander("Raw report", expanded=False):
            st.text(raw_text)


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


def render_candidate_selector(
    source: sqlite3.Row,
    candidates: list[sqlite3.Row],
    *,
    possible_limit: int = 4,
) -> tuple[set[int], list[dict[str, object]]]:
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

    st.markdown("<div class='pane-label'>Checked procedures</div>", unsafe_allow_html=True)
    if not selected:
        st.caption("No mappings are currently selected.")
    selected_ids.update(_render_candidate_checks(selected, True))

    st.markdown("<div class='pane-label'>Suggestions</div>", unsafe_allow_html=True)
    visible_possible = possible[:possible_limit]
    overflow_possible = possible[possible_limit:]
    if not possible:
        st.caption("No additional possible matches.")
    else:
        selected_ids.update(_render_candidate_checks(visible_possible, False))
        if overflow_possible:
            with st.expander(f"{len(overflow_possible)} more suggestions", expanded=False):
                selected_ids.update(_render_candidate_checks(overflow_possible, False))

    st.markdown("<div class='pane-label'>Search ACGME procedures</div>", unsafe_allow_html=True)
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
    return selected_ids, checked_results


def candidate_review_card(conn: sqlite3.Connection, source: sqlite3.Row, candidates: list[sqlite3.Row]) -> None:
    selected_ids, checked_results = render_candidate_selector(source, candidates)
    approve_col, skip_col = st.columns(2)
    with approve_col:
        if st.button("Approve checked", type="primary", key=f"approve_candidates_{source['id']}"):
            if checked_results:
                selected_ids.update(add_manual_candidates(conn, int(source["id"]), checked_results))
            inserted = approve_candidate_review(conn, int(source["id"]), selected_ids)
            st.success(f"Approved {len(selected_ids)} mappings; created {inserted} new entries.")
            scroll_to_top_on_next_render()
            st.rerun()
    with skip_col:
        if st.button("Skip this case", key=f"skip_candidates_{source['id']}"):
            approve_candidate_review(conn, int(source["id"]), set())
            st.success("Skipped this case.")
            scroll_to_top_on_next_render()
            st.rerun()


def entry_label(entry: sqlite3.Row) -> str:
    return str(entry["acgme_description"] or entry["type"] or "Generated procedure")


def entry_metadata(entry: sqlite3.Row) -> str:
    parts = []
    if "acgme_code" in entry.keys() and entry["acgme_code"]:
        parts.append(str(entry["acgme_code"]))
    parts.append(f"{entry['area']} / {entry['type']}")
    if entry["acgme_def_category"]:
        parts.append(f"Def Cat: {entry['acgme_def_category']}")
    confidence = entry["mapping_confidence"] if "mapping_confidence" in entry.keys() else ""
    if confidence:
        parts.append(f"Confidence: {confidence}")
    return " | ".join(parts)


def render_generated_entry_checks(entries: list[sqlite3.Row], *, key_prefix: str, default_checked: bool = True) -> set[int]:
    selected_ids: set[int] = set()
    if not entries:
        st.caption("No generated procedures.")
        return selected_ids
    for row in entries:
        checked = st.checkbox(
            entry_label(row),
            value=default_checked,
            key=f"{key_prefix}_entry_{row['id']}",
            help=(row["evidence_excerpt"] if "evidence_excerpt" in row.keys() else "") or row["comments"] or row["mapping_rule_name"] or None,
        )
        st.markdown(
            f"<div class='procedure-meta'>{html.escape(entry_metadata(row))}</div>",
            unsafe_allow_html=True,
        )
        if "failure_reason" in row.keys() and row["failure_reason"]:
            st.caption(f"Failure: {row['failure_reason']}")
        if checked:
            selected_ids.add(int(row["id"]))
    return selected_ids


def target_to_manual_values(source: sqlite3.Row, target: dict[str, object], *, comments: str = "") -> dict[str, object]:
    values = default_values_for_source(source)
    values.update(
        {
            "acgme_code": target.get("acgme_code", ""),
            "area": target.get("area", ""),
            "type": target.get("type", ""),
            "acgme_description": target.get("acgme_description", ""),
            "acgme_def_category": target.get("acgme_def_category", ""),
            "keyword": target.get("keyword", ""),
            "component_label": target.get("component_label", "") or values["component_label"],
            "comments": comments or target.get("reason", "") or target.get("match_reason", "") or "Added from review UI.",
        }
    )
    return values


def create_manual_entries_from_targets(conn: sqlite3.Connection, source: sqlite3.Row, targets: list[dict[str, object]]) -> int:
    created = 0
    for target in targets:
        create_manual_entry(conn, int(source["id"]), target_to_manual_values(source, target))
        created += 1
    return created


def render_target_suggestions(
    source: sqlite3.Row,
    suggestions: list[dict[str, object]],
    *,
    key_prefix: str,
    possible_limit: int = 4,
) -> list[dict[str, object]]:
    checked: list[dict[str, object]] = []
    visible = suggestions[:possible_limit]
    overflow = suggestions[possible_limit:]
    if not suggestions:
        st.caption("No suggested procedures.")
    for idx, target in enumerate(visible):
        selected = st.checkbox(
            target_primary_label(target),
            value=False,
            key=f"{key_prefix}_suggestion_{source['id']}_{idx}_{target.get('area', '')}_{target.get('type', '')}",
        )
        meta = target_metadata(target)
        if target.get("confidence") or target.get("score"):
            meta = f"{meta} | {target.get('confidence', '')} {target.get('score', '')}".strip()
        st.caption(meta)
        if selected:
            checked.append(target)
    if overflow:
        with st.expander(f"{len(overflow)} more suggestions", expanded=False):
            for idx, target in enumerate(overflow, start=len(visible)):
                selected = st.checkbox(
                    target_primary_label(target),
                    value=False,
                    key=f"{key_prefix}_suggestion_{source['id']}_{idx}_{target.get('area', '')}_{target.get('type', '')}",
                )
                st.caption(target_metadata(target))
                if selected:
                    checked.append(target)
    return checked


def render_target_search(source: sqlite3.Row, *, key_prefix: str) -> list[dict[str, object]]:
    query = st.text_input("Search by description, type, area, or def cat", key=f"{key_prefix}_search_{source['id']}")
    results = search_acgme_targets(query) if query else []
    checked: list[dict[str, object]] = []
    if results:
        for idx, target in enumerate(results[:10]):
            selected = st.checkbox(
                target_primary_label(target),
                value=False,
                key=f"{key_prefix}_search_result_{source['id']}_{idx}_{target.get('acgme_code', '')}_{target['area']}_{target['type']}_{target.get('acgme_description', '')}",
            )
            st.caption(target_metadata(target))
            if selected:
                checked.append(target)
    elif query:
        st.caption("No official ACGME targets matched that search.")
    return checked


def generated_entry_review_card(
    conn: sqlite3.Connection,
    source: sqlite3.Row,
    entries: list[sqlite3.Row],
    *,
    key_prefix: str,
    approve_label: str = "Approve checked",
) -> None:
    st.markdown("<div class='pane-label'>Checked procedures</div>", unsafe_allow_html=True)
    selected_ids = render_generated_entry_checks(entries, key_prefix=key_prefix, default_checked=True)
    st.markdown("<div class='pane-label'>Suggestions</div>", unsafe_allow_html=True)
    st.caption("No additional deterministic suggestions for this generated entry.")
    st.markdown("<div class='pane-label'>Search ACGME procedures</div>", unsafe_allow_html=True)
    checked_results = render_target_search(source, key_prefix=key_prefix)
    approve_col, skip_col = st.columns(2)
    with approve_col:
        if st.button(approve_label, type="primary", key=f"{key_prefix}_approve_{source['id']}"):
            skipped = [int(row["id"]) for row in entries if int(row["id"]) not in selected_ids]
            if selected_ids:
                update_review_status(conn, sorted(selected_ids), "approved")
            if skipped:
                update_review_status(conn, skipped, "skipped")
            created = create_manual_entries_from_targets(conn, source, checked_results) if checked_results else 0
            st.success(f"Approved {len(selected_ids)} entries; added {created} manual entries.")
            scroll_to_top_on_next_render()
            st.rerun()
    with skip_col:
        if st.button("Skip this case", key=f"{key_prefix}_skip_{source['id']}"):
            update_review_status(conn, [int(row["id"]) for row in entries], "skipped")
            scroll_to_top_on_next_render()
            st.rerun()


def no_match_review_card(conn: sqlite3.Connection, source: sqlite3.Row, suggestions: list[dict[str, object]]) -> None:
    st.markdown("<div class='pane-label'>Checked procedures</div>", unsafe_allow_html=True)
    st.caption("No procedures are currently checked.")
    st.markdown("<div class='pane-label'>Suggestions</div>", unsafe_allow_html=True)
    checked_suggestions = render_target_suggestions(source, suggestions, key_prefix="no_match")
    st.markdown("<div class='pane-label'>Search ACGME procedures</div>", unsafe_allow_html=True)
    checked_results = render_target_search(source, key_prefix="no_match")
    approve_col, skip_col = st.columns(2)
    with approve_col:
        if st.button("Approve checked", type="primary", key=f"no_match_approve_{source['id']}"):
            selected = checked_suggestions + checked_results
            created = create_manual_entries_from_targets(conn, source, selected) if selected else 0
            if created == 0:
                conn.execute(
                    "UPDATE source_cases SET source_mapping_status = 'candidate_reviewed_empty' WHERE id = ?",
                    (source["id"],),
                )
                conn.commit()
            st.success(f"Created {created} entries.")
            scroll_to_top_on_next_render()
            st.rerun()
    with skip_col:
        if st.button("Skip this case", key=f"no_match_skip_{source['id']}"):
            conn.execute(
                "UPDATE source_cases SET source_mapping_status = 'excluded', needs_review_reason = COALESCE(needs_review_reason, 'Skipped in review') WHERE id = ?",
                (source["id"],),
            )
            conn.commit()
            scroll_to_top_on_next_render()
            st.rerun()


def failed_upload_review_card(conn: sqlite3.Connection, source: sqlite3.Row, entries: list[sqlite3.Row]) -> None:
    st.markdown("<div class='pane-label'>Checked failed procedures</div>", unsafe_allow_html=True)
    selected_ids = render_generated_entry_checks(entries, key_prefix="failed_upload", default_checked=True)
    st.markdown("<div class='pane-label'>Suggestions</div>", unsafe_allow_html=True)
    st.caption("Failed uploads keep their original mapped procedures. Use search only if the target needs correction.")
    st.markdown("<div class='pane-label'>Search ACGME procedures</div>", unsafe_allow_html=True)
    checked_results = render_target_search(source, key_prefix="failed_upload")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Reset checked for upload", type="primary", disabled=not selected_ids, key=f"failed_reset_{source['id']}"):
            reset_upload(conn, sorted(selected_ids))
            created = create_manual_entries_from_targets(conn, source, checked_results) if checked_results else 0
            st.success(f"Reset {len(selected_ids)} entries; added {created} manual entries.")
            scroll_to_top_on_next_render()
            st.rerun()
    with c2:
        if st.button("Mark checked submitted", disabled=not selected_ids, key=f"failed_submitted_{source['id']}"):
            mark_upload_submitted(conn, sorted(selected_ids))
            scroll_to_top_on_next_render()
            st.rerun()


def render_review_workspace(
    conn: sqlite3.Connection,
    source: sqlite3.Row,
    render_procedure_pane: Any,
    *,
    status: str = "",
) -> None:
    context = compact_case_header(conn, source, status=status)
    report_col, procedure_col = st.columns([1.45, 1], gap="medium")
    with report_col:
        render_report_evidence(context["report_context"], source)
    with procedure_col:
        render_procedure_pane()


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
install_review_css()
maybe_scroll_to_top()
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
    queue_labels = {
        "High Confidence": f"High Confidence ({counts['high_confidence']})",
        "Low Confidence": f"Low Confidence ({counts['low_confidence']})",
        "No Match": f"No Match ({counts['no_match']})",
        "Failed Uploads": f"Failed Uploads ({counts['failed_uploads']})",
    }
    active_queue = st.segmented_control(
        "Review queue",
        list(queue_labels),
        format_func=lambda key: queue_labels[str(key)],
        default=st.session_state.get("active_review_queue", "High Confidence"),
        key="active_review_queue",
        label_visibility="collapsed",
    )
    active_queue = str(active_queue or "High Confidence")

    if active_queue == "High Confidence":
        action_col, note_col = st.columns([1, 3])
        with action_col:
            if st.button("Accept all", type="primary", disabled=counts["high_confidence"] == 0, key="accept_all_high_confidence"):
                scope_filter = (
                    "AND EXISTS (SELECT 1 FROM import_source_cases isc WHERE isc.source_case_id = generated_entries.source_case_id AND isc.import_id = ?)"
                    if active_import_id is not None
                    else ""
                )
                ids = pd.read_sql_query(
                    f"""
                    SELECT id FROM generated_entries
                    WHERE review_status = 'new_high_confidence'
                      AND mapping_confidence = 'high'
                      AND role_confidence = 'high'
                      AND compound_flag = 0
                      AND upload_status IN ('not_uploaded', 'reset')
                      {scope_filter}
                    """,
                    conn,
                    params=(active_import_id,) if active_import_id is not None else (),
                )["id"].astype(int).tolist()
                update_review_status(conn, ids, "approved")
                st.success(f"Approved {len(ids)} high-confidence entries.")
                scroll_to_top_on_next_render()
                st.rerun()
        with note_col:
            st.caption("Safe single-procedure, high-confidence entries. Review one case below or accept the full current scope.")
        source, entries = load_next_high_confidence_group(conn, active_import_id)
        if not source:
            st.info("No high-confidence cases in this scope.")
        else:
            render_review_workspace(
                conn,
                source,
                lambda: generated_entry_review_card(
                    conn,
                    source,
                    entries,
                    key_prefix="high_confidence",
                    approve_label="Approve checked",
                ),
                status="High confidence",
            )

    elif active_queue == "Low Confidence":
        source, candidates = load_next_candidate_group(conn, active_import_id, include_unmapped=False)
        if not source:
            if active_import_id is not None and import_running:
                st.info("Waiting for mapped cases from this import.")
            elif active_import_id is not None:
                st.info("No low-confidence cases in this import. Use All backlog to review historical cases.")
            else:
                st.info("No low-confidence cases need review.")
        else:
            render_review_workspace(
                conn,
                source,
                lambda: candidate_review_card(conn, source, candidates),
                status="Low confidence",
            )
            legacy_source, entries = load_next_generated_group(conn, active_import_id)
            if legacy_source and int(legacy_source["id"]) == int(source["id"]) and entries:
                generated_rows_are_llm = any((row["mapping_rule_id"] or "").startswith("llm:") for row in entries)
                with st.expander("Generated entries for this case", expanded=generated_rows_are_llm):
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

    elif active_queue == "No Match":
        source, suggestions = load_next_unmapped(conn, require_suggestion=None, import_id=active_import_id)
        if not source:
            st.info("No unmatched source cases in this scope.")
        else:
            render_review_workspace(
                conn,
                source,
                lambda: no_match_review_card(conn, source, suggestions),
                status="No match",
            )

    else:
        source, entries = load_next_failed_upload_group(conn, active_import_id)
        if not source:
            st.info("No failed uploads in this scope.")
        else:
            render_review_workspace(
                conn,
                source,
                lambda: failed_upload_review_card(conn, source, entries),
                status="Failed upload",
            )

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
