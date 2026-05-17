from __future__ import annotations

import sqlite3
from typing import Any

from .config_io import load_resident_profile
from .constants import DEFAULT_CASE_CLASS, DEFAULT_SITE
from .importer import insert_generated_entries, update_source_mapping_status
from .mapper import load_mapping_rules, map_source_case
from .matching import suggest_mappings
from .utils import case_year_from_date, format_acgme_date, patient_type, patient_type_from_age


def source_row_to_mapping_source(row: sqlite3.Row, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    profile = profile or load_resident_profile()
    resident = profile["resident"]
    defaults = profile["defaults"]
    resident_position = row["resident_position"]
    found_resident = bool(row["resident_found_in_report"])
    return {
        "accession_number": row["accession_number"],
        "study_date": row["study_date"],
        "patient_birth_date": row["patient_birth_date"],
        "patient_age_years": row["patient_age_years"] if "patient_age_years" in row.keys() else None,
        "exam_code": row["exam_code"] or "",
        "study_description": row["study_description"] or "",
        "report_snippet": row["report_snippet"] or "",
        "procedure_text": row["procedure_text"] or "",
        "institution_name": row["institution_name"] or "",
        "source_format": row["source_format"] if "source_format" in row.keys() else "visage_xlsx",
        "source_row_number": row["source_row_number"] if "source_row_number" in row.keys() else None,
        "modality": row["modality"] if "modality" in row.keys() else "",
        "cpt_code": row["cpt_code"] if "cpt_code" in row.keys() else "",
        "duplicate_accession_flag": row["duplicate_accession_flag"] if "duplicate_accession_flag" in row.keys() else 0,
        "parsed_report_json": row["parsed_report_json"] if "parsed_report_json" in row.keys() else None,
        "principal_result_interpreter_raw": row["principal_result_interpreter_raw"] or "",
        "attending_name": row["attending_name"] or "",
        "source_row_hash": row["source_row_hash"],
        "resident_found_in_report": int(found_resident),
        "resident_position": resident_position,
        "role_parse_source": row["role_parse_source"],
        "aborted_flag": row["aborted_flag"],
        "unsuccessful_flag": row["unsuccessful_flag"],
        "no_procedure_flag": row["no_procedure_flag"],
        "needs_review_reason": row["needs_review_reason"] or "",
        "derived": {
            "case_id": row["accession_number"],
            "case_date": format_acgme_date(row["study_date"]),
            "case_year": case_year_from_date(
                row["study_date"],
                int(resident["expected_graduation_year"]),
                int(resident.get("pgy_max", 5)),
            ),
            "role": "Secondary" if found_resident and resident_position and resident_position > 1 else "Primary",
            "role_confidence": "high" if found_resident else "low",
            "site": defaults.get("site", DEFAULT_SITE),
            "patient_type": (
                patient_type_from_age(row["patient_age_years"], int(defaults.get("pediatric_age_cutoff", 18)))
                if ("patient_age_years" in row.keys() and row["patient_age_years"] not in (None, ""))
                else patient_type(
                    row["study_date"],
                    row["patient_birth_date"],
                    int(defaults.get("pediatric_age_cutoff", 18)),
                )
            ),
            "case_class": defaults.get("case_class", DEFAULT_CASE_CLASS),
        },
    }


def review_counts(conn: sqlite3.Connection) -> dict[str, int]:
    counts = {
        "batch_approvable": conn.execute(
            """
            SELECT COUNT(*) FROM generated_entries
            WHERE review_status = 'new_high_confidence'
              AND mapping_confidence = 'high'
              AND role_confidence = 'high'
              AND compound_flag = 0
              AND upload_status IN ('not_uploaded', 'reset')
            """
        ).fetchone()[0],
        "needs_review": conn.execute(
            """
            SELECT COUNT(DISTINCT source_case_id) FROM generated_entries
            WHERE review_status IN ('new_high_confidence', 'needs_review')
              AND upload_status IN ('not_uploaded', 'reset')
            """
        ).fetchone()[0],
        "upload_failures": conn.execute(
            "SELECT COUNT(*) FROM generated_entries WHERE upload_status = 'failed'"
        ).fetchone()[0],
        "unmapped_total": conn.execute(
            "SELECT COUNT(*) FROM source_cases WHERE source_mapping_status IN ('unmapped', 'flag_only')"
        ).fetchone()[0],
    }
    with_suggestions = 0
    rules, _ = load_mapping_rules()
    rows = conn.execute(
        """
        SELECT * FROM source_cases
        WHERE source_mapping_status IN ('unmapped', 'flag_only')
        ORDER BY study_date DESC
        LIMIT 250
        """
    ).fetchall()
    for row in rows:
        if suggest_mappings(source_row_to_mapping_source(row), rules, limit=1):
            with_suggestions += 1
    counts["unmapped_with_suggestions"] = with_suggestions
    counts["unmapped_without_suggestions"] = max(counts["unmapped_total"] - with_suggestions, 0)
    return counts


def load_next_generated_group(conn: sqlite3.Connection) -> tuple[sqlite3.Row | None, list[sqlite3.Row]]:
    source = conn.execute(
        """
        SELECT sc.*
        FROM source_cases sc
        JOIN generated_entries ge ON ge.source_case_id = sc.id
        WHERE ge.review_status IN ('new_high_confidence', 'needs_review')
          AND ge.upload_status IN ('not_uploaded', 'reset')
        ORDER BY
          CASE ge.review_status WHEN 'needs_review' THEN 0 ELSE 1 END,
          ge.id
        LIMIT 1
        """
    ).fetchone()
    if not source:
        return None, []
    entries = conn.execute(
        """
        SELECT * FROM generated_entries
        WHERE source_case_id = ?
          AND review_status IN ('new_high_confidence', 'needs_review')
          AND upload_status IN ('not_uploaded', 'reset')
        ORDER BY id
        """,
        (source["id"],),
    ).fetchall()
    return source, entries


def load_next_unmapped(conn: sqlite3.Connection, require_suggestion: bool | None = None) -> tuple[sqlite3.Row | None, list[dict[str, Any]]]:
    rules, _ = load_mapping_rules()
    rows = conn.execute(
        """
        SELECT * FROM source_cases
        WHERE source_mapping_status IN ('unmapped', 'flag_only')
        ORDER BY study_date DESC
        LIMIT 500
        """
    ).fetchall()
    fallback: tuple[sqlite3.Row | None, list[dict[str, Any]]] = (None, [])
    for row in rows:
        suggestions = [suggestion.as_dict() for suggestion in suggest_mappings(source_row_to_mapping_source(row), rules)]
        if require_suggestion is True and suggestions:
            return row, suggestions
        if require_suggestion is False and not suggestions:
            return row, []
        if require_suggestion is None:
            return row, suggestions
        if fallback[0] is None:
            fallback = (row, suggestions)
    return fallback


def remap_unresolved_cases(conn: sqlite3.Connection, limit: int | None = None) -> dict[str, int]:
    rules, rules_hash = load_mapping_rules()
    sql = """
        SELECT * FROM source_cases
        WHERE source_mapping_status IN ('unmapped', 'flag_only')
        ORDER BY study_date DESC
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql).fetchall()
    summary = {"checked": 0, "generated_entries": 0, "generated_cases": 0, "suggested_only": 0, "unchanged": 0}
    for row in rows:
        summary["checked"] += 1
        source = source_row_to_mapping_source(row)
        entries, update = map_source_case(source, rules, rules_hash)
        if entries:
            inserted = insert_generated_entries(conn, row["id"], entries)
            update_source_mapping_status(conn, row["id"], update)
            summary["generated_entries"] += inserted
            summary["generated_cases"] += int(inserted > 0)
        elif update.get("mapping_suggestions"):
            summary["suggested_only"] += 1
        else:
            summary["unchanged"] += 1
    conn.commit()
    return summary
