from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from app.config_io import load_dropdowns
from app.config_io import load_resident_profile
from app.constants import DEFAULT_CASE_CLASS, DEFAULT_DB_PATH, DEFAULT_SITE
from app.export_payload import export_approved_json
from app.importer import import_xlsx
from app.learning import append_learned_rule, apply_mapping_to_matching_unsubmitted, learned_rule_count
from app.models import connect, init_db, log_event, utc_now
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
    uploaded = st.file_uploader("Visage/mPower XLSX", type=["xlsx"])
    import_path = st.text_input("Or local XLSX path", "visage-data-export.xlsx")
    if st.button("Import XLSX", type="primary"):
        path: str | Path
        if uploaded:
            tmp = Path("data/imports") / uploaded.name
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(uploaded.getbuffer())
            path = tmp
        else:
            path = import_path
        with st.spinner("Importing and mapping cases..."):
            summary = import_xlsx(conn, path)
        st.success(f"Imported {summary['row_count']} rows; generated {summary['generated_entries_count']} entries.")

    st.header("Export")
    if st.button("Export approved JSON"):
        out = export_approved_json(conn)
        st.success(f"Wrote {out}")

tab_entries, tab_unmapped, tab_imports = st.tabs(["Generated entries", "Unmapped", "Imports"])

with tab_entries:
    filter_name = st.selectbox(
        "Filter",
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
    st.caption(f"{len(df)} entries")
    st.dataframe(df, width="stretch", hide_index=True)
    selected_raw = st.text_input("Entry IDs for action, comma-separated")
    selected_ids = [int(x.strip()) for x in selected_raw.split(",") if x.strip().isdigit()]
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        if st.button("Batch approve visible high-confidence"):
            visible = df[
                (df["mapping_confidence"] == "high")
                & (df["role_confidence"] == "high")
                & (df["compound_flag"] == 0)
            ]["id"].astype(int).tolist()
            update_review_status(conn, visible, "approved")
            st.success(f"Approved {len(visible)} entries.")
            st.rerun()
    with c2:
        if st.button("Approve selected", disabled=not selected_ids):
            update_review_status(conn, selected_ids, "approved")
            st.rerun()
    with c3:
        if st.button("Skip selected permanently", disabled=not selected_ids):
            update_review_status(conn, selected_ids, "skipped")
            st.rerun()
    with c4:
        if st.button("Reset upload selected", disabled=not selected_ids):
            reset_upload(conn, selected_ids)
            st.rerun()
    with c5:
        if st.button("Mark selected submitted", disabled=not selected_ids):
            mark_upload_submitted(conn, selected_ids)
            st.rerun()

    with st.expander("Edit one generated entry"):
        edit_id = st.number_input("Entry ID", min_value=0, step=1, key="edit_id")
        if edit_id:
            row = conn.execute("SELECT * FROM generated_entries WHERE id = ?", (int(edit_id),)).fetchone()
            if row:
                with st.form("edit_entry_form"):
                    role = st.selectbox("Role", ["Primary", "Secondary"], index=0 if row["role"] == "Primary" else 1)
                    site = st.text_input("Site", row["site"])
                    patient = st.selectbox("Patient Type", ["Adult", "Pediatric"], index=0 if row["patient_type"] == "Adult" else 1)
                    case_class = DEFAULT_CASE_CLASS
                    areas = area_options()
                    area = st.selectbox("Area", areas, index=index_or_zero(areas, row["area"]))
                    types = type_options_for_area(area)
                    typ = st.selectbox("Type", types, index=index_or_zero(types, row["type"]))
                    acgme_description = st.text_input("ACGME Description", row["acgme_description"] or "")
                    acgme_def_category = st.text_input("Def Cat", row["acgme_def_category"] or "")
                    component = st.text_input("Component label", row["component_label"])
                    keyword = st.text_input("Keyword", row["keyword"] or "")
                    comments = st.text_area("Comments", row["comments"] or "")
                    learn = st.checkbox("Learn this mapping correction for future imports", value=True)
                    apply_now = st.checkbox("Apply learned mapping to matching unsubmitted entries now", value=True)
                    if st.form_submit_button("Save as edited"):
                        values = {
                            "role": role,
                            "site": site,
                            "patient_type": patient,
                            "case_class": case_class,
                            "area": area,
                            "type": typ,
                            "acgme_description": acgme_description,
                            "acgme_def_category": acgme_def_category,
                            "component_label": component,
                            "keyword": keyword,
                            "comments": comments,
                        }
                        save_entry_edit(
                            conn,
                            int(edit_id),
                            values,
                        )
                        message = "Entry saved as edited."
                        if learn:
                            rule_id, applied = learn_from_generated_entry(conn, int(edit_id), values, apply_now)
                            message += f" Learned `{rule_id}`; applied to {applied} matching unsubmitted entries."
                        st.success(message)
                        st.rerun()

with tab_unmapped:
    unmapped = load_unmapped(conn)
    st.caption("Unmapped and flag-only source cases do not create exportable ACGME entries.")
    st.dataframe(unmapped, width="stretch", hide_index=True)
    with st.expander("Create manual entry from source case"):
        source_id = st.number_input("Source case ID", min_value=0, step=1, key="manual_source_id")
        if source_id:
            source = conn.execute("SELECT * FROM source_cases WHERE id = ?", (int(source_id),)).fetchone()
            if source:
                default_patient = patient_type(source["study_date"], source["patient_birth_date"])
                with st.form("manual_entry_form"):
                    st.write(f"Accession `{source['accession_number']}` | {source['procedure_text']}")
                    role = st.selectbox("Role", ["Primary", "Secondary"])
                    site = st.text_input("Site", DEFAULT_SITE)
                    patient = st.selectbox(
                        "Patient Type",
                        ["Adult", "Pediatric"],
                        index=0 if default_patient == "Adult" else 1,
                    )
                    case_class = DEFAULT_CASE_CLASS
                    areas = area_options()
                    area = st.selectbox("Area", areas)
                    types = type_options_for_area(area)
                    typ = st.selectbox("Type", types)
                    acgme_description = st.text_input("ACGME Description")
                    acgme_def_category = st.text_input("Def Cat")
                    component = st.text_input("Component label", canonical_key(source["procedure_text"]) or "manual")
                    keyword = st.text_input("Keyword")
                    comments = st.text_area("Comments", source["needs_review_reason"] or "")
                    learn = st.checkbox("Learn this manual mapping for future imports", value=True)
                    apply_now = st.checkbox("Apply learned mapping to matching unsubmitted entries now", value=True)
                    if st.form_submit_button("Create edited entry"):
                        values = {
                            "role": role,
                            "site": site,
                            "patient_type": patient,
                            "case_class": case_class,
                            "area": area,
                            "type": typ,
                            "acgme_description": acgme_description,
                            "acgme_def_category": acgme_def_category,
                            "component_label": component,
                            "keyword": keyword,
                            "comments": comments,
                        }
                        entry_id = create_manual_entry(
                            conn,
                            int(source_id),
                            values,
                        )
                        message = f"Created entry {entry_id}."
                        if learn:
                            rule_id, applied = learn_from_source_case(conn, int(source_id), values, apply_now)
                            message += f" Learned `{rule_id}`; applied to {applied} matching unsubmitted entries."
                        st.success(message)
                        st.rerun()

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
