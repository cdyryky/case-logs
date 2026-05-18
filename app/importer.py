from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Callable

from .config_io import load_resident_profile
from .llm_client import LLMSettings, OllamaClient
from .llm_mapping import LLMExtraction, LLMMappingError, llm_entries_for_source
from .mapper import load_mapping_rules, map_source_case, validate_rules
from .models import init_db, log_event, utc_now
from .parser import iter_mpower_raw_rows, transform_mpower_row
from .utils import file_sha256


ProgressCallback = Callable[[dict[str, Any]], None]


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
    source_format = source.get("source_format") or "mpower_csv"
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


def record_import_source_case(
    conn: sqlite3.Connection,
    import_id: int,
    source_case_id: int,
    source_row_number: int | None,
    accession_number: str,
    inserted_source_case: bool,
    generated_entries_count: int = 0,
    mapped: bool = False,
) -> None:
    now = utc_now()
    conn.execute(
        """
        INSERT INTO import_source_cases(
          import_id, source_case_id, source_row_number, accession_number, inserted_source_case,
          generated_entries_count, mapped_at, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(import_id, source_row_number, source_case_id) DO UPDATE SET
          inserted_source_case = excluded.inserted_source_case,
          generated_entries_count = excluded.generated_entries_count,
          mapped_at = COALESCE(excluded.mapped_at, import_source_cases.mapped_at),
          updated_at = excluded.updated_at
        """,
        (
            import_id,
            source_case_id,
            source_row_number,
            accession_number,
            int(inserted_source_case),
            generated_entries_count,
            now if mapped else None,
            now,
            now,
        ),
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
              role_confidence, compound_flag, review_status, upload_status, evidence_excerpt, llm_model,
              llm_prompt_version, llm_raw_response_json, created_at, updated_at
            )
            VALUES (
              :source_case_id, :dedupe_key, :component_label, :case_id, :case_date, :case_year, :role, :site,
              :patient_type, :case_class, :acgme_code, :area, :type, :acgme_description, :acgme_def_category, :keyword, :comments, :mapping_rule_id,
              :mapping_rule_version, :mapping_rules_file_hash, :mapping_rule_name, :mapping_confidence,
              :role_confidence, :compound_flag, :review_status, 'not_uploaded', :evidence_excerpt, :llm_model,
              :llm_prompt_version, :llm_raw_response_json, :created_at, :updated_at
            )
            """,
            {
                **entry,
                "acgme_code": entry.get("acgme_code", ""),
                "evidence_excerpt": entry.get("evidence_excerpt", ""),
                "llm_model": entry.get("llm_model", ""),
                "llm_prompt_version": entry.get("llm_prompt_version", ""),
                "llm_raw_response_json": entry.get("llm_raw_response_json", ""),
                "source_case_id": source_case_id,
                "created_at": now,
                "updated_at": now,
            },
        )
        if cur.rowcount == 1:
            count += 1
            entry_id = conn.execute("SELECT id FROM generated_entries WHERE dedupe_key = ?", (entry["dedupe_key"],)).fetchone()["id"]
            log_event(conn, entry_id, "generated", None, entry["review_status"], "importer", entry["mapping_rule_name"])
    return count


def _emit_progress(
    callback: ProgressCallback | None,
    row_number: int,
    total_rows: int,
    accession: str,
    phase: str,
    **extra: Any,
) -> None:
    if callback:
        callback({"row": row_number, "total": total_rows, "accession": accession, "phase": phase, **extra})


def _map_with_llm_or_fallback(
    source: dict[str, Any],
    rules: list[Any],
    rules_hash: str,
    llm_client: Any | None = None,
    mapping_mode: str | None = None,
    llm_timeout_seconds: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mode = (mapping_mode or os.environ.get("ACGME_MAPPING_MODE", "llm")).strip().lower()
    if mode == "legacy":
        return map_source_case(source, rules, rules_hash, validate=False)

    if llm_client is None and llm_timeout_seconds is not None:
        settings = LLMSettings.from_env()
        llm_client = OllamaClient(
            LLMSettings(
                base_url=settings.base_url,
                model=settings.model,
                timeout_seconds=llm_timeout_seconds,
                num_ctx=settings.num_ctx,
            )
        )

    try:
        extraction: LLMExtraction = llm_entries_for_source(source, client=llm_client)
    except LLMMappingError as exc:
        entries, update = map_source_case(source, rules, rules_hash, validate=False)
        if entries:
            update["source_mapping_status"] = "llm_failed_fallback_generated"
            update["needs_review_reason"] = "; ".join(
                reason for reason in [source.get("needs_review_reason", ""), f"LLM failed; legacy fallback used: {exc}"] if reason
            )
        else:
            update["source_mapping_status"] = "llm_failed_unmapped"
            update["needs_review_reason"] = "; ".join(
                reason for reason in [source.get("needs_review_reason", ""), f"LLM failed; no fallback mapping: {exc}"] if reason
            )
        return entries, update

    if extraction.entries:
        reason = "; ".join(item for item in [source.get("needs_review_reason", ""), *extraction.warnings] if item)
        return extraction.entries, {"source_mapping_status": "generated_llm", "needs_review_reason": reason}

    entries, update = map_source_case(source, rules, rules_hash, validate=False)
    if entries:
        update["source_mapping_status"] = "llm_failed_fallback_generated"
        update["needs_review_reason"] = "; ".join(
            reason
            for reason in [source.get("needs_review_reason", ""), "LLM returned no procedures; legacy fallback used."]
            if reason
        )
    else:
        update["source_mapping_status"] = "llm_failed_unmapped"
        update["needs_review_reason"] = "; ".join(
            reason for reason in [source.get("needs_review_reason", ""), "LLM returned no loggable procedures."] if reason
        )
    return entries, update


def import_mpower_csv(
    conn: sqlite3.Connection,
    csv_path: str | Path,
    progress_callback: ProgressCallback | None = None,
    llm_client: Any | None = None,
    commit_per_row: bool = False,
    mapping_mode: str | None = None,
    llm_timeout_seconds: float | None = None,
) -> dict[str, int | str]:
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
    _emit_progress(progress_callback, 0, 0, "", "Import started", import_id=import_id)
    if commit_per_row:
        conn.commit()
    raw_rows = list(iter_mpower_raw_rows(path))
    accession_counts: dict[str, int] = {}
    for raw in raw_rows:
        accession = str(raw.get("Accession Number") or "").strip()
        accession_counts[accession] = accession_counts.get(accession, 0) + 1

    row_count = new_count = duplicate_count = generated_count = 0
    total_rows = len(raw_rows)
    for raw in raw_rows:
        row_count += 1
        accession = str(raw.get("Accession Number") or "").strip()
        _emit_progress(progress_callback, row_count, total_rows, accession, "Parsing source row", import_id=import_id)
        source = transform_mpower_row(raw, duplicate_accession=accession_counts.get(accession, 0) > 1, resident_profile=profile)
        source_case_id, inserted = insert_source_case(conn, source, path.name, import_id)
        source_row_number = source.get("source_row_number")
        source_row_number = int(source_row_number) if source_row_number not in (None, "") else row_count
        record_import_source_case(
            conn,
            import_id,
            source_case_id,
            source_row_number,
            accession,
            inserted,
        )
        new_count += int(inserted)
        duplicate_count += int(not inserted)
        if commit_per_row:
            conn.execute(
                """
                UPDATE imports
                SET row_count = ?, new_source_cases = ?, duplicate_source_cases = ?, generated_entries_count = ?
                WHERE id = ?
                """,
                (row_count, new_count, duplicate_count, generated_count, import_id),
            )
            conn.commit()
            _emit_progress(progress_callback, row_count, total_rows, accession, "Queued for mapping", import_id=import_id)
        mode = (mapping_mode or os.environ.get("ACGME_MAPPING_MODE", "llm")).strip().lower()
        phase = "Mapping with legacy rules" if mode == "legacy" else "Mapping with local LLM"
        _emit_progress(progress_callback, row_count, total_rows, accession, phase, import_id=import_id)
        entries, source_update = _map_with_llm_or_fallback(
            source,
            rules,
            rules_hash,
            llm_client=llm_client,
            mapping_mode=mapping_mode,
            llm_timeout_seconds=llm_timeout_seconds,
        )
        _emit_progress(progress_callback, row_count, total_rows, accession, "Writing generated entries", import_id=import_id)
        update_source_mapping_status(conn, source_case_id, source_update)
        inserted_entries = insert_generated_entries(conn, source_case_id, entries)
        generated_count += inserted_entries
        record_import_source_case(
            conn,
            import_id,
            source_case_id,
            source_row_number,
            accession,
            inserted,
            inserted_entries,
            mapped=True,
        )
        if commit_per_row:
            conn.execute(
                """
                UPDATE imports
                SET row_count = ?, new_source_cases = ?, duplicate_source_cases = ?, generated_entries_count = ?
                WHERE id = ?
                """,
                (row_count, new_count, duplicate_count, generated_count, import_id),
            )
            conn.commit()
            _emit_progress(progress_callback, row_count, total_rows, accession, "Committed row", import_id=import_id)

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
