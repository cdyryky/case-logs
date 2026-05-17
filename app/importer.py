from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .config_io import load_resident_profile
from .mapper import load_mapping_rules, map_source_case, validate_rules
from .models import init_db, log_event, utc_now
from .parser import iter_mpower_raw_rows, iter_raw_rows, transform_mpower_row, transform_source_row
from .utils import file_sha256


def insert_source_case(
    conn: sqlite3.Connection,
    source: dict[str, Any],
    source_file_name: str,
    import_id: int,
) -> tuple[int, bool]:
    now = utc_now()
    if source.get("source_format") == "mpower_csv":
        existing_hash = conn.execute(
            """
            SELECT id FROM source_cases
            WHERE source_format = 'mpower_csv' AND source_row_hash = ?
            """,
            (source["source_row_hash"],),
        ).fetchone()
        if existing_hash:
            return int(existing_hash["id"]), False

    params = {
        **{k: source.get(k) for k in [
            "accession_number",
            "study_date",
            "patient_birth_date",
            "patient_age_years",
            "exam_code",
            "study_description",
            "report_snippet",
            "procedure_text",
            "institution_name",
            "source_format",
            "source_row_number",
            "modality",
            "cpt_code",
            "duplicate_accession_flag",
            "parsed_report_json",
            "principal_result_interpreter_raw",
            "attending_name",
            "source_row_hash",
            "resident_found_in_report",
            "resident_position",
            "role_parse_source",
            "aborted_flag",
            "unsuccessful_flag",
            "no_procedure_flag",
            "needs_review_reason",
        ]},
        "source_file_name": source_file_name,
        "import_id": import_id,
        "imported_at": now,
    }
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO source_cases(
          accession_number, study_date, patient_birth_date, patient_age_years, exam_code, study_description, report_snippet,
          procedure_text, institution_name, source_format, source_row_number, modality, cpt_code,
          duplicate_accession_flag, parsed_report_json, principal_result_interpreter_raw, attending_name,
          source_file_name, import_id, source_row_hash, resident_found_in_report, resident_position,
          role_parse_source, aborted_flag, unsuccessful_flag, no_procedure_flag, needs_review_reason, imported_at
        )
        VALUES (
          :accession_number, :study_date, :patient_birth_date, :patient_age_years, :exam_code, :study_description, :report_snippet,
          :procedure_text, :institution_name, :source_format, :source_row_number, :modality, :cpt_code,
          :duplicate_accession_flag, :parsed_report_json, :principal_result_interpreter_raw, :attending_name,
          :source_file_name, :import_id, :source_row_hash, :resident_found_in_report, :resident_position,
          :role_parse_source, :aborted_flag, :unsuccessful_flag, :no_procedure_flag, :needs_review_reason, :imported_at
        )
        """,
        params,
    )
    inserted = cur.rowcount == 1
    if inserted:
        return int(cur.lastrowid), True
    source_format = source.get("source_format") or "visage_xlsx"
    row = conn.execute(
        """
        SELECT id FROM source_cases
        WHERE source_format = ? AND source_row_hash = ?
        """,
        (source_format, source["source_row_hash"]),
    ).fetchone()
    if row:
        return int(row["id"]), False
    row = conn.execute(
        """
        SELECT id FROM source_cases
        WHERE accession_number = ? AND study_date = ? AND exam_code = ?
        ORDER BY id
        LIMIT 1
        """,
        (source["accession_number"], source["study_date"], source["exam_code"]),
    ).fetchone()
    if not row:
        raise ValueError(f"Failed to insert or locate source case: {source['accession_number']}")
    return int(row["id"]), inserted


def update_source_mapping_status(conn: sqlite3.Connection, source_case_id: int, update: dict[str, Any]) -> None:
    conn.execute(
        """
        UPDATE source_cases
        SET source_mapping_status = ?, needs_review_reason = COALESCE(NULLIF(?, ''), needs_review_reason)
        WHERE id = ?
        """,
        (update.get("source_mapping_status", "unmapped"), update.get("needs_review_reason", ""), source_case_id),
    )


def insert_generated_entries(
    conn: sqlite3.Connection,
    source_case_id: int,
    entries: list[dict[str, Any]],
) -> int:
    count = 0
    now = utc_now()
    for entry in entries:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO generated_entries(
              source_case_id, dedupe_key, component_label, case_id, case_date, case_year, role, site,
              patient_type, case_class, acgme_code, area, type, acgme_description, acgme_def_category, keyword, comments, mapping_rule_id,
              mapping_rule_version, mapping_rules_file_hash, mapping_rule_name, mapping_confidence,
              role_confidence, compound_flag, review_status, upload_status, created_at, updated_at
            )
            VALUES (
              :source_case_id, :dedupe_key, :component_label, :case_id, :case_date, :case_year, :role, :site,
              :patient_type, :case_class, :acgme_code, :area, :type, :acgme_description, :acgme_def_category, :keyword, :comments, :mapping_rule_id,
              :mapping_rule_version, :mapping_rules_file_hash, :mapping_rule_name, :mapping_confidence,
              :role_confidence, :compound_flag, :review_status, 'not_uploaded', :created_at, :updated_at
            )
            """,
            {**entry, "acgme_code": entry.get("acgme_code", ""), "source_case_id": source_case_id, "created_at": now, "updated_at": now},
        )
        if cur.rowcount == 1:
            count += 1
            entry_id = conn.execute("SELECT id FROM generated_entries WHERE dedupe_key = ?", (entry["dedupe_key"],)).fetchone()["id"]
            log_event(conn, entry_id, "generated", None, entry["review_status"], "importer", entry["mapping_rule_name"])
    return count


def import_xlsx(conn: sqlite3.Connection, xlsx_path: str | Path) -> dict[str, int | str]:
    init_db(conn)
    path = Path(xlsx_path)
    profile = load_resident_profile()
    rules, rules_hash = load_mapping_rules()
    validate_rules(rules)
    file_hash = file_sha256(path)
    import_cur = conn.execute(
        """
        INSERT INTO imports(filename, file_hash, imported_at, row_count, new_source_cases, duplicate_source_cases, generated_entries_count)
        VALUES (?, ?, ?, 0, 0, 0, 0)
        """,
        (path.name, file_hash, utc_now()),
    )
    import_id = int(import_cur.lastrowid)
    row_count = new_count = duplicate_count = generated_count = 0

    for raw in iter_raw_rows(path):
        row_count += 1
        source = transform_source_row(raw, profile)
        source_case_id, inserted = insert_source_case(conn, source, path.name, import_id)
        new_count += int(inserted)
        duplicate_count += int(not inserted)
        entries, source_update = map_source_case(source, rules, rules_hash, validate=False)
        update_source_mapping_status(conn, source_case_id, source_update)
        generated_count += insert_generated_entries(conn, source_case_id, entries)

    conn.execute(
        """
        UPDATE imports
        SET row_count = ?, new_source_cases = ?, duplicate_source_cases = ?, generated_entries_count = ?
        WHERE id = ?
        """,
        (row_count, new_count, duplicate_count, generated_count, import_id),
    )
    log_event(conn, None, "imported", None, None, "importer", f"{path.name}: {row_count} rows")
    return {
        "import_id": import_id,
        "row_count": row_count,
        "new_source_cases": new_count,
        "duplicate_source_cases": duplicate_count,
        "generated_entries_count": generated_count,
    }


def import_mpower_csv(conn: sqlite3.Connection, csv_path: str | Path) -> dict[str, int | str]:
    init_db(conn)
    path = Path(csv_path)
    profile = load_resident_profile()
    rules, rules_hash = load_mapping_rules()
    validate_rules(rules)
    file_hash = file_sha256(path)
    import_cur = conn.execute(
        """
        INSERT INTO imports(filename, file_hash, imported_at, row_count, new_source_cases, duplicate_source_cases, generated_entries_count)
        VALUES (?, ?, ?, 0, 0, 0, 0)
        """,
        (path.name, file_hash, utc_now()),
    )
    import_id = int(import_cur.lastrowid)
    raw_rows = list(iter_mpower_raw_rows(path))
    accession_counts: dict[str, int] = {}
    for raw in raw_rows:
        accession = str(raw.get("Accession Number") or "").strip()
        accession_counts[accession] = accession_counts.get(accession, 0) + 1

    row_count = new_count = duplicate_count = generated_count = 0
    for raw in raw_rows:
        row_count += 1
        accession = str(raw.get("Accession Number") or "").strip()
        source = transform_mpower_row(raw, duplicate_accession=accession_counts.get(accession, 0) > 1, resident_profile=profile)
        source_case_id, inserted = insert_source_case(conn, source, path.name, import_id)
        new_count += int(inserted)
        duplicate_count += int(not inserted)
        entries, source_update = map_source_case(source, rules, rules_hash, validate=False)
        update_source_mapping_status(conn, source_case_id, source_update)
        generated_count += insert_generated_entries(conn, source_case_id, entries)

    conn.execute(
        """
        UPDATE imports
        SET row_count = ?, new_source_cases = ?, duplicate_source_cases = ?, generated_entries_count = ?
        WHERE id = ?
        """,
        (row_count, new_count, duplicate_count, generated_count, import_id),
    )
    log_event(conn, None, "imported", None, None, "importer", f"{path.name}: {row_count} mPower rows")
    return {
        "import_id": import_id,
        "row_count": row_count,
        "new_source_cases": new_count,
        "duplicate_source_cases": duplicate_count,
        "generated_entries_count": generated_count,
    }
