from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from app.config_io import load_dropdowns
from app.config_io import load_resident_profile
from app.constants import DEFAULT_CASE_CLASS, DEFAULT_DB_PATH, DEFAULT_SITE
from app.export_payload import export_approved_json
from app.importer import import_mpower_csv, import_xlsx
from app.learning import append_learned_rule, apply_mapping_to_matching_unsubmitted, learned_rule_count
from app.models import connect, init_db, log_event, utc_now
from app.review_queue import (
    load_next_generated_group,
    load_next_unmapped,
    remap_unresolved_cases,
    review_counts,
    source_row_to_mapping_source,
)
from app.utils import case_year_from_date, canonical_key, format_acgme_date, patient_type


def get_conn() -> sqlite3.Connection:
    conn = connect(DEFAULT_DB_PATH)
    init_db(conn)
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
    return pd.read_sql_query(
        f"""
        SELECT ge.id, ge.case_date, ge.case_id, ge.role, ge.patient_type, ge.area, ge.type,
               ge.acgme_description, ge.acgme_def_category,
               ge.component_label, ge.mapping_confidence, ge.role_confidence, ge.compound_flag,
               ge.review_status, ge.upload_status, ge.mapping_rule_name, ge.failure_reason,
               sc.exam_code, sc.procedure_text, sc.study_description, sc.attending_name,
               sc.resident_found_in_report, sc.resident_position, sc.needs_review_reason
        FROM generated_entries ge
        JOIN source_cases sc ON sc.id = ge.source_case_id
        WHERE {where}
        ORDER BY ge.id
        """,
        conn,
    )


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
          patient_type, case_class, area, type, acgme_description, acgme_def_category, keyword, comments, mapping_rule_id,
          mapping_rule_version, mapping_rules_file_hash, mapping_rule_name, mapping_confidence,
          role_confidence, compound_flag, review_status, upload_status, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'manual', '1', 'manual', 'Manual entry',
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


def source_context(source: sqlite3.Row) -> None:
    mapping_source = source_row_to_mapping_source(source)
    derived = mapping_source["derived"]
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


def load_unmapped(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT id, accession_number, study_date, exam_code, procedure_text, study_description,
               source_mapping_status, needs_review_reason, attending_name
        FROM source_cases
        WHERE source_mapping_status IN ('unmapped', 'flag_only')
        ORDER BY study_date DESC
        LIMIT 1000
        """,
        conn,
    )


st.set_page_config(page_title="ACGME IR Case Logs", layout="wide")
st.title("ACGME IR Case Log Review")
st.caption(f"Learned mapping correction rules: {learned_rule_count()}")

conn = get_conn()

with st.sidebar:
    st.header("Import")
    uploaded = st.file_uploader("Visage XLSX or mPower CSV", type=["xlsx", "csv"])
    import_path = st.text_input("Or local import path", "data/exports/mpower-download-260526-clean.csv")
    if st.button("Import file", type="primary"):
        path: str | Path
        if uploaded:
            tmp = Path("data/imports") / uploaded.name
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(uploaded.getbuffer())
            path = tmp
        else:
            path = import_path
        with st.spinner("Importing and mapping cases..."):
            suffix = Path(path).suffix.lower()
            summary = import_mpower_csv(conn, path) if suffix == ".csv" else import_xlsx(conn, path)
        st.success(f"Imported {summary['row_count']} rows; generated {summary['generated_entries_count']} entries.")

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

tab_review, tab_diagnostics, tab_imports = st.tabs(["Review Queue", "Diagnostics", "Imports"])

with tab_review:
    counts = review_counts(conn)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Batch approvable", counts["batch_approvable"])
    m2.metric("Needs review", counts["needs_review"])
    m3.metric("Unmapped suggestions", counts["unmapped_with_suggestions"])
    m4.metric("Unmapped no suggestion", counts["unmapped_without_suggestions"])
    m5.metric("Upload failures", counts["upload_failures"])

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
        source, entries = load_next_generated_group(conn)
        if not source:
            st.info("No generated entries need review.")
        else:
            source_context(source)
            rows = [
                {
                    "id": row["id"],
                    "role": row["role"],
                    "area": row["area"],
                    "type": row["type"],
                    "description": row["acgme_description"],
                    "def_cat": row["acgme_def_category"],
                    "confidence": row["mapping_confidence"],
                    "reason": row["comments"] or row["mapping_rule_name"],
                }
                for row in entries
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            ids = [int(row["id"]) for row in entries]
            a1, a2 = st.columns(2)
            with a1:
                if st.button("Approve", type="primary"):
                    update_review_status(conn, ids, "approved")
                    st.rerun()
            with a2:
                if st.button("Skip"):
                    update_review_status(conn, ids, "skipped")
                    st.rerun()

            with st.expander("Edit"):
                labels = [f"{row['id']}: {row['area']} / {row['type']}" for row in entries]
                selected_label = st.selectbox("Entry", labels)
                selected_id = int(selected_label.split(":", 1)[0])
                selected = next(row for row in entries if int(row["id"]) == selected_id)
                defaults = dict(selected)
                submitted, values, learn, apply_now = mapping_form("review_edit_form", defaults, "Save")
                if submitted:
                    save_entry_edit(conn, selected_id, values)
                    message = "Entry saved."
                    if learn:
                        rule_id, applied = learn_from_generated_entry(conn, selected_id, values, apply_now)
                        message += f" Learned `{rule_id}`; applied to {applied} matching entries."
                    st.success(message)
                    st.rerun()

    elif queue in {"Unmapped with suggestions", "Unmapped without suggestions"}:
        source, suggestions = load_next_unmapped(conn, require_suggestion=queue == "Unmapped with suggestions")
        if not source:
            st.info("No source cases in this queue.")
        else:
            source_context(source)
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
    unmapped = load_unmapped(conn)
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
