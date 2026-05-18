from __future__ import annotations

import json
import sqlite3
from typing import Any

from .config_io import load_resident_profile
from .constants import DEFAULT_CASE_CLASS, DEFAULT_SITE
from .candidates import ALGORITHM_VERSION, ensure_candidates_for_source, load_candidates
from .importer import insert_generated_entries, update_source_mapping_status
from .mapper import load_mapping_rules, map_source_case, validate_rules
from .matching import suggest_mappings
from .models import utc_now
from .utils import case_year_from_date, format_acgme_date, patient_type, patient_type_from_age


def latest_import(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM imports ORDER BY id DESC LIMIT 1").fetchone()


def import_scope_summary(conn: sqlite3.Connection, import_id: int | None) -> dict[str, Any]:
    if import_id is None:
        return {"import_id": None, "filename": "All backlog", "row_count": 0, "scoped_source_cases": 0, "mapped_source_cases": 0}
    row = conn.execute("SELECT * FROM imports WHERE id = ?", (import_id,)).fetchone()
    if not row:
        return {"import_id": import_id, "filename": "", "row_count": 0, "scoped_source_cases": 0, "mapped_source_cases": 0}
    scoped = conn.execute(
        """
        SELECT
          COUNT(DISTINCT source_case_id) AS source_cases,
          COUNT(DISTINCT CASE WHEN mapped_at IS NOT NULL THEN source_case_id END) AS mapped_cases
        FROM import_source_cases
        WHERE import_id = ?
        """,
        (import_id,),
    ).fetchone()
    return {
        "import_id": import_id,
        "filename": row["filename"],
        "row_count": row["row_count"],
        "new_source_cases": row["new_source_cases"],
        "duplicate_source_cases": row["duplicate_source_cases"],
        "generated_entries_count": row["generated_entries_count"],
        "imported_at": row["imported_at"],
        "scoped_source_cases": scoped["source_cases"] if scoped else 0,
        "mapped_source_cases": scoped["mapped_cases"] if scoped else 0,
    }


def _scope_exists_sql(alias: str = "sc") -> str:
    return f"EXISTS (SELECT 1 FROM import_source_cases isc WHERE isc.source_case_id = {alias}.id AND isc.import_id = ?)"


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
        "source_format": row["source_format"] if "source_format" in row.keys() else "mpower_csv",
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


def parse_report_context_json(value: object) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def report_context_score(parsed: dict[str, Any], raw_text: object = "") -> int:
    if not parsed:
        return 0
    score = min(len(str(raw_text or "")), 5000)
    if parsed.get("procedure_title"):
        score += 100
    if parsed.get("impression"):
        score += 5000
    score += 3000 * len(parsed.get("procedure_summary_sections") or [])
    score += 50 * len(parsed.get("candidate_procedure_phrases") or [])
    return score


def best_report_context_for_source(conn: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    parsed = parse_report_context_json(row["parsed_report_json"] if "parsed_report_json" in row.keys() else None)
    raw_text = row["report_snippet"] if "report_snippet" in row.keys() else ""
    best = {
        "parsed": parsed,
        "raw_text": raw_text or "",
        "source_case_id": row["id"] if "id" in row.keys() else None,
        "source_format": row["source_format"] if "source_format" in row.keys() else "",
    }
    best_score = report_context_score(parsed, raw_text)
    accession = row["accession_number"] if "accession_number" in row.keys() else ""
    if not accession:
        return best
    siblings = conn.execute(
        """
        SELECT id, source_format, report_snippet, parsed_report_json
        FROM source_cases
        WHERE accession_number = ?
        """,
        (accession,),
    ).fetchall()
    for sibling in siblings:
        sibling_parsed = parse_report_context_json(sibling["parsed_report_json"])
        sibling_raw = sibling["report_snippet"] or ""
        sibling_score = report_context_score(sibling_parsed, sibling_raw)
        if sibling_score > best_score:
            best = {
                "parsed": sibling_parsed,
                "raw_text": sibling_raw,
                "source_case_id": sibling["id"],
                "source_format": sibling["source_format"] or "",
            }
            best_score = sibling_score
    return best


def review_counts(conn: sqlite3.Connection, import_id: int | None = None) -> dict[str, int]:
    if import_id is None:
        scope_params: tuple[Any, ...] = ()
        generated_scope = ""
        candidate_scope = ""
        source_scope = ""
    else:
        scope_params = (import_id,)
        generated_scope = "AND EXISTS (SELECT 1 FROM import_source_cases isc WHERE isc.source_case_id = generated_entries.source_case_id AND isc.import_id = ?)"
        candidate_scope = "AND EXISTS (SELECT 1 FROM import_source_cases isc WHERE isc.source_case_id = source_match_candidates.source_case_id AND isc.import_id = ?)"
        source_scope = "AND EXISTS (SELECT 1 FROM import_source_cases isc WHERE isc.source_case_id = source_cases.id AND isc.import_id = ?)"
    return {
        "batch_approvable": conn.execute(
            f"""
            SELECT COUNT(*) FROM generated_entries
            WHERE review_status = 'new_high_confidence'
              AND mapping_confidence = 'high'
              AND role_confidence = 'high'
              AND compound_flag = 0
              AND upload_status IN ('not_uploaded', 'reset')
              {generated_scope}
            """,
            scope_params,
        ).fetchone()[0],
        "needs_review": conn.execute(
            f"""
            SELECT COUNT(DISTINCT source_case_id)
            FROM (
              SELECT source_case_id
              FROM source_match_candidates
              WHERE user_status = 'pending'
                AND algorithm_version = ?
                {candidate_scope}
              UNION
              SELECT source_case_id
              FROM generated_entries
              WHERE review_status IN ('new_high_confidence', 'needs_review')
                AND upload_status IN ('not_uploaded', 'reset')
                {generated_scope}
            )
            """,
            (ALGORITHM_VERSION, *scope_params, *scope_params),
        ).fetchone()[0],
        "upload_failures": conn.execute(
            f"""
            SELECT COUNT(*) FROM generated_entries
            WHERE upload_status = 'failed'
              {generated_scope}
            """,
            scope_params,
        ).fetchone()[0],
        "unmapped_total": conn.execute(
            f"""
            SELECT COUNT(*) FROM source_cases
            WHERE source_mapping_status IN ('unmapped', 'flag_only', 'llm_failed_unmapped')
              {source_scope}
            """,
            scope_params,
        ).fetchone()[0],
    }


def load_next_generated_group(conn: sqlite3.Connection, import_id: int | None = None) -> tuple[sqlite3.Row | None, list[sqlite3.Row]]:
    scope = f"AND {_scope_exists_sql('sc')}" if import_id is not None else ""
    params = (import_id,) if import_id is not None else ()
    source = conn.execute(
        f"""
        SELECT sc.*
        FROM source_cases sc
        JOIN generated_entries ge ON ge.source_case_id = sc.id
        WHERE ge.review_status IN ('new_high_confidence', 'needs_review')
          AND ge.upload_status IN ('not_uploaded', 'reset')
          {scope}
        ORDER BY
          CASE WHEN ge.mapping_rule_id LIKE 'llm:%' THEN 0 ELSE 1 END,
          CASE ge.review_status WHEN 'needs_review' THEN 0 ELSE 1 END,
          ge.id
        LIMIT 1
        """,
        params,
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


def load_next_candidate_group(conn: sqlite3.Connection, import_id: int | None = None) -> tuple[sqlite3.Row | None, list[sqlite3.Row]]:
    scope = f"AND {_scope_exists_sql('sc')}" if import_id is not None else ""
    params = (import_id,) if import_id is not None else ()
    source = conn.execute(
        f"""
        SELECT sc.*
        FROM source_cases sc
        JOIN generated_entries ge ON ge.source_case_id = sc.id
        WHERE ge.review_status IN ('new_high_confidence', 'needs_review')
          AND ge.upload_status IN ('not_uploaded', 'reset')
          AND ge.mapping_rule_id LIKE 'llm:%'
          AND sc.source_mapping_status NOT IN ('candidate_reviewed', 'candidate_reviewed_empty', 'excluded')
          {scope}
        ORDER BY ge.id
        LIMIT 1
        """,
        params,
    ).fetchone()
    if not source:
        source = conn.execute(
            f"""
        SELECT sc.*
        FROM source_cases sc
        WHERE EXISTS (
          SELECT 1
          FROM source_match_candidates smc
          WHERE smc.source_case_id = sc.id
            AND smc.user_status = 'pending'
            AND smc.algorithm_version = ?
        )
          AND sc.source_mapping_status NOT IN ('candidate_reviewed', 'candidate_reviewed_empty', 'excluded')
          {scope}
        ORDER BY sc.study_date DESC, sc.id
        LIMIT 1
        """,
            (ALGORITHM_VERSION, *params),
        ).fetchone()
    if not source:
        source = conn.execute(
            f"""
            SELECT sc.*
            FROM source_cases sc
            JOIN generated_entries ge ON ge.source_case_id = sc.id
            WHERE ge.review_status IN ('new_high_confidence', 'needs_review')
              AND ge.upload_status IN ('not_uploaded', 'reset')
              AND sc.source_mapping_status NOT IN ('candidate_reviewed', 'candidate_reviewed_empty', 'excluded')
              {scope}
            ORDER BY
              CASE WHEN ge.mapping_rule_id LIKE 'llm:%' THEN 0 ELSE 1 END,
              CASE ge.review_status WHEN 'needs_review' THEN 0 ELSE 1 END,
              ge.id
            LIMIT 1
            """,
            params,
        ).fetchone()
    if not source:
        source = conn.execute(
            f"""
            SELECT sc.*
            FROM source_cases sc
            WHERE sc.source_mapping_status IN ('unmapped', 'flag_only', 'llm_failed_unmapped')
              {scope}
            ORDER BY sc.study_date DESC, sc.id
            LIMIT 1
            """,
            params,
        ).fetchone()
    if not source:
        return None, []
    ensure_candidates_for_source(conn, source)
    return source, load_candidates(conn, int(source["id"]))


def clear_deterministic_review_state(conn: sqlite3.Connection, import_id: int | None = None) -> dict[str, int]:
    candidate_scope = (
        "AND EXISTS (SELECT 1 FROM import_source_cases isc WHERE isc.source_case_id = source_match_candidates.source_case_id AND isc.import_id = ?)"
        if import_id is not None
        else ""
    )
    generated_scope = (
        "AND EXISTS (SELECT 1 FROM import_source_cases isc WHERE isc.source_case_id = generated_entries.source_case_id AND isc.import_id = ?)"
        if import_id is not None
        else ""
    )
    params = (import_id,) if import_id is not None else ()
    affected_ids = {
        int(row["source_case_id"])
        for row in conn.execute(
            f"""
            SELECT source_case_id
            FROM source_match_candidates
            WHERE user_status = 'pending'
              {candidate_scope}
            """,
            params,
        ).fetchall()
    }
    affected_ids.update(
        int(row["source_case_id"])
        for row in conn.execute(
            f"""
            SELECT source_case_id
            FROM generated_entries
            WHERE review_status IN ('new_high_confidence', 'needs_review')
              AND upload_status IN ('not_uploaded', 'reset')
              AND mapping_rule_id NOT LIKE 'llm:%'
              {generated_scope}
            """,
            params,
        ).fetchall()
    )
    deleted_candidates = conn.execute(
        f"""
        DELETE FROM source_match_candidates
        WHERE user_status = 'pending'
          {candidate_scope}
        """,
        params,
    ).rowcount
    skipped_entries = conn.execute(
        f"""
        UPDATE generated_entries
        SET review_status = 'skipped', updated_at = ?
        WHERE review_status IN ('new_high_confidence', 'needs_review')
          AND upload_status IN ('not_uploaded', 'reset')
          AND mapping_rule_id NOT LIKE 'llm:%'
          {generated_scope}
        """,
        (utc_now(), *params),
    ).rowcount
    reset_sources = 0
    if affected_ids:
        placeholders = ",".join("?" for _ in affected_ids)
        reset_sources = conn.execute(
            f"""
            UPDATE source_cases
            SET source_mapping_status = 'unmapped'
            WHERE id IN ({placeholders})
              AND source_mapping_status NOT IN ('candidate_reviewed', 'candidate_reviewed_empty', 'excluded')
            """,
            tuple(sorted(affected_ids)),
        ).rowcount
    conn.commit()
    return {
        "deleted_candidates": int(deleted_candidates),
        "skipped_entries": int(skipped_entries),
        "reset_sources": int(reset_sources),
    }


def load_next_unmapped(
    conn: sqlite3.Connection,
    require_suggestion: bool | None = None,
    import_id: int | None = None,
) -> tuple[sqlite3.Row | None, list[dict[str, Any]]]:
    rules, _ = load_mapping_rules()
    scope = f"AND {_scope_exists_sql('source_cases')}" if import_id is not None else ""
    params = (import_id,) if import_id is not None else ()
    rows = conn.execute(
        f"""
        SELECT * FROM source_cases
        WHERE source_mapping_status IN ('unmapped', 'flag_only', 'llm_failed_unmapped')
          {scope}
        ORDER BY study_date DESC
        LIMIT 500
        """,
        params,
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
    validate_rules(rules)
    sql = """
        SELECT * FROM source_cases
        WHERE source_mapping_status IN ('unmapped', 'flag_only', 'llm_failed_unmapped')
        ORDER BY study_date DESC
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql).fetchall()
    summary = {"checked": 0, "generated_entries": 0, "generated_cases": 0, "suggested_only": 0, "unchanged": 0}
    for row in rows:
        summary["checked"] += 1
        source = source_row_to_mapping_source(row)
        entries, update = map_source_case(source, rules, rules_hash, validate=False)
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
