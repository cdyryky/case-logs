from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .constants import DATA_DIR, DEFAULT_DB_PATH, EXPORTS_DIR, IMPORTS_DIR


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dirs() -> None:
    for path in (DATA_DIR, IMPORTS_DIR, EXPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


SOURCE_CASE_COLUMNS = [
    "id",
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
    "source_file_name",
    "import_id",
    "source_row_hash",
    "resident_found_in_report",
    "resident_position",
    "role_parse_source",
    "aborted_flag",
    "unsuccessful_flag",
    "no_procedure_flag",
    "source_mapping_status",
    "needs_review_reason",
    "imported_at",
]


def _create_source_cases_sql(table_name: str = "source_cases") -> str:
    return f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
          id INTEGER PRIMARY KEY,
          accession_number TEXT NOT NULL,
          study_date TEXT NOT NULL,
          patient_birth_date TEXT,
          patient_age_years TEXT,
          exam_code TEXT,
          study_description TEXT,
          report_snippet TEXT,
          procedure_text TEXT,
          institution_name TEXT,
          source_format TEXT NOT NULL DEFAULT 'visage_xlsx',
          source_row_number INTEGER,
          modality TEXT,
          cpt_code TEXT,
          duplicate_accession_flag INTEGER NOT NULL DEFAULT 0,
          parsed_report_json TEXT,
          principal_result_interpreter_raw TEXT,
          attending_name TEXT,
          source_file_name TEXT,
          import_id INTEGER,
          source_row_hash TEXT NOT NULL,
          resident_found_in_report INTEGER,
          resident_position INTEGER,
          role_parse_source TEXT,
          aborted_flag INTEGER NOT NULL DEFAULT 0,
          unsuccessful_flag INTEGER NOT NULL DEFAULT 0,
          no_procedure_flag INTEGER NOT NULL DEFAULT 0,
          source_mapping_status TEXT NOT NULL DEFAULT 'unmapped',
          needs_review_reason TEXT,
          imported_at TEXT NOT NULL,
          UNIQUE(source_format, accession_number, study_date, exam_code, source_row_hash),
          FOREIGN KEY(import_id) REFERENCES imports(id)
        );
    """


def migrate_source_cases_unique_constraint(conn: sqlite3.Connection) -> None:
    old_unique_exists = False
    for index in conn.execute("PRAGMA index_list(source_cases)"):
        if not index["unique"]:
            continue
        cols = [row["name"] for row in conn.execute(f"PRAGMA index_info({index['name']})")]
        if cols == ["accession_number", "study_date", "exam_code"]:
            old_unique_exists = True
            break
    if not old_unique_exists:
        return

    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("DROP TABLE IF EXISTS source_cases_new")
    conn.executescript(_create_source_cases_sql("source_cases_new").replace("IF NOT EXISTS ", ""))
    cols = ", ".join(SOURCE_CASE_COLUMNS)
    conn.execute(
        f"""
        INSERT OR IGNORE INTO source_cases_new({cols})
        SELECT {cols}
        FROM source_cases
        """
    )
    conn.execute("DROP TABLE source_cases")
    conn.execute("ALTER TABLE source_cases_new RENAME TO source_cases")
    conn.execute("PRAGMA foreign_keys = ON")


@contextmanager
def db(db_path: str | Path = DEFAULT_DB_PATH) -> Iterable[sqlite3.Connection]:
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS imports (
          id INTEGER PRIMARY KEY,
          filename TEXT NOT NULL,
          file_hash TEXT NOT NULL,
          imported_at TEXT NOT NULL,
          row_count INTEGER NOT NULL,
          new_source_cases INTEGER NOT NULL,
          duplicate_source_cases INTEGER NOT NULL,
          generated_entries_count INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS generated_entries (
          id INTEGER PRIMARY KEY,
          source_case_id INTEGER NOT NULL,
          dedupe_key TEXT NOT NULL UNIQUE,
          component_label TEXT NOT NULL,
          case_id TEXT NOT NULL,
          case_date TEXT NOT NULL,
          case_year INTEGER NOT NULL,
          role TEXT NOT NULL,
          site TEXT NOT NULL,
          patient_type TEXT NOT NULL,
          case_class TEXT NOT NULL,
          area TEXT NOT NULL,
          type TEXT NOT NULL,
          acgme_description TEXT,
          acgme_def_category TEXT,
          keyword TEXT,
          comments TEXT,
          mapping_rule_id TEXT,
          mapping_rule_version TEXT,
          mapping_rules_file_hash TEXT,
          mapping_rule_name TEXT,
          mapping_confidence TEXT NOT NULL,
          role_confidence TEXT NOT NULL,
          compound_flag INTEGER NOT NULL DEFAULT 0,
          review_status TEXT NOT NULL,
          upload_status TEXT NOT NULL DEFAULT 'not_uploaded',
          failure_reason TEXT,
          failure_timestamp TEXT,
          submitted_at TEXT,
          acgme_confirmation TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          FOREIGN KEY(source_case_id) REFERENCES source_cases(id)
        );

        CREATE INDEX IF NOT EXISTS idx_generated_review_upload
          ON generated_entries(review_status, upload_status);

        CREATE TABLE IF NOT EXISTS baseline_submissions (
          id INTEGER PRIMARY KEY,
          accession_number TEXT,
          case_date TEXT,
          case_class TEXT,
          area TEXT,
          type TEXT,
          component_label TEXT,
          baseline_source TEXT NOT NULL,
          marked_at TEXT NOT NULL,
          notes TEXT
        );

        CREATE TABLE IF NOT EXISTS entry_events (
          id INTEGER PRIMARY KEY,
          entry_id INTEGER,
          event_type TEXT NOT NULL,
          old_status TEXT,
          new_status TEXT,
          timestamp TEXT NOT NULL,
          source TEXT NOT NULL,
          details TEXT,
          FOREIGN KEY(entry_id) REFERENCES generated_entries(id)
        );

        CREATE TABLE IF NOT EXISTS upload_session (
          id INTEGER PRIMARY KEY CHECK (id = 1),
          current_entry_id INTEGER,
          previous_entry_id INTEGER,
          updated_at TEXT NOT NULL
        );

        INSERT OR IGNORE INTO upload_session(id, updated_at) VALUES (1, datetime('now'));
        """
    )
    conn.executescript(_create_source_cases_sql())
    ensure_columns(
        conn,
        "source_cases",
        {
            "patient_age_years": "TEXT",
            "source_format": "TEXT NOT NULL DEFAULT 'visage_xlsx'",
            "source_row_number": "INTEGER",
            "modality": "TEXT",
            "cpt_code": "TEXT",
            "duplicate_accession_flag": "INTEGER NOT NULL DEFAULT 0",
            "parsed_report_json": "TEXT",
        },
    )
    migrate_source_cases_unique_constraint(conn)
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_source_accession
          ON source_cases(accession_number);
        """
    )
    ensure_columns(
        conn,
        "generated_entries",
        {
            "acgme_description": "TEXT",
            "acgme_def_category": "TEXT",
        },
    )


def log_event(
    conn: sqlite3.Connection,
    entry_id: int | None,
    event_type: str,
    old_status: str | None = None,
    new_status: str | None = None,
    source: str = "app",
    details: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO entry_events(entry_id, event_type, old_status, new_status, timestamp, source, details)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (entry_id, event_type, old_status, new_status, utc_now(), source, details),
    )


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None
