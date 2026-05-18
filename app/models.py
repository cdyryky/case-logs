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
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    if str(db_path) != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 30000")
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
    "llm_case_id",
    "mapping_pathway",
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
          source_format TEXT NOT NULL DEFAULT 'mpower_csv',
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
          llm_case_id TEXT,
          mapping_pathway TEXT NOT NULL DEFAULT 'ollama',
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
          UNIQUE(llm_case_id),
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


def migrate_source_cases_default(conn: sqlite3.Connection) -> None:
    legacy_default = "DEFAULT " + repr("vis" + "age_xlsx")
    row = conn.execute(
        """
        SELECT sql
        FROM sqlite_master
        WHERE type = 'table' AND name = 'source_cases'
        """
    ).fetchone()
    if not row or legacy_default not in (row["sql"] or ""):
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
          generated_entries_count INTEGER NOT NULL,
          mapping_pathway TEXT NOT NULL DEFAULT 'ollama'
        );

        CREATE TABLE IF NOT EXISTS import_source_cases (
          id INTEGER PRIMARY KEY,
          import_id INTEGER NOT NULL,
          source_case_id INTEGER NOT NULL,
          source_row_number INTEGER,
          accession_number TEXT,
          inserted_source_case INTEGER NOT NULL DEFAULT 0,
          generated_entries_count INTEGER NOT NULL DEFAULT 0,
          mapped_at TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          UNIQUE(import_id, source_row_number, source_case_id),
          FOREIGN KEY(import_id) REFERENCES imports(id),
          FOREIGN KEY(source_case_id) REFERENCES source_cases(id)
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
          acgme_code TEXT,
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
          evidence_excerpt TEXT,
          llm_model TEXT,
          llm_prompt_version TEXT,
          llm_raw_response_json TEXT,
          mapping_pathway TEXT NOT NULL DEFAULT 'ollama',
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          FOREIGN KEY(source_case_id) REFERENCES source_cases(id)
        );

        CREATE INDEX IF NOT EXISTS idx_generated_review_upload
          ON generated_entries(review_status, upload_status);

        CREATE INDEX IF NOT EXISTS idx_import_source_cases_import
          ON import_source_cases(import_id, source_case_id);

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

        CREATE TABLE IF NOT EXISTS source_match_candidates (
          id INTEGER PRIMARY KEY,
          source_case_id INTEGER NOT NULL,
          candidate_key TEXT NOT NULL UNIQUE,
          case_class TEXT NOT NULL,
          acgme_code TEXT,
          area TEXT NOT NULL,
          type TEXT NOT NULL,
          acgme_description TEXT,
          acgme_def_category TEXT,
          keyword TEXT,
          component_label TEXT NOT NULL,
          score REAL NOT NULL,
          confidence TEXT NOT NULL,
          default_checked INTEGER NOT NULL DEFAULT 0,
          user_checked INTEGER,
          user_status TEXT NOT NULL DEFAULT 'pending',
          match_reason TEXT,
          matched_phrases_json TEXT,
          evidence_snippet TEXT,
          event_key TEXT,
          event_label TEXT,
          source_kind TEXT NOT NULL,
          algorithm_version TEXT NOT NULL,
          learning_weight REAL NOT NULL DEFAULT 0,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          FOREIGN KEY(source_case_id) REFERENCES source_cases(id)
        );

        CREATE TABLE IF NOT EXISTS mapping_learning_signals (
          id INTEGER PRIMARY KEY,
          source_case_id INTEGER NOT NULL,
          candidate_id INTEGER,
          signal_type TEXT NOT NULL,
          learning_weight REAL NOT NULL,
          exam_code TEXT,
          procedure_text TEXT,
          case_class TEXT NOT NULL,
          area TEXT NOT NULL,
          type TEXT NOT NULL,
          acgme_description TEXT,
          acgme_def_category TEXT,
          evidence_json TEXT,
          created_at TEXT NOT NULL,
          FOREIGN KEY(source_case_id) REFERENCES source_cases(id),
          FOREIGN KEY(candidate_id) REFERENCES source_match_candidates(id)
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
            "source_format": "TEXT NOT NULL DEFAULT 'mpower_csv'",
            "source_row_number": "INTEGER",
            "modality": "TEXT",
            "cpt_code": "TEXT",
            "duplicate_accession_flag": "INTEGER NOT NULL DEFAULT 0",
            "parsed_report_json": "TEXT",
            "llm_case_id": "TEXT",
            "mapping_pathway": "TEXT NOT NULL DEFAULT 'ollama'",
        },
    )
    migrate_source_cases_unique_constraint(conn)
    migrate_source_cases_default(conn)
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_source_accession
          ON source_cases(accession_number);
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_candidates_source_status
          ON source_match_candidates(source_case_id, user_status);
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_learning_signals_target
          ON mapping_learning_signals(exam_code, area, type);
        """
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO import_source_cases(
          import_id, source_case_id, source_row_number, accession_number, inserted_source_case,
          generated_entries_count, mapped_at, created_at, updated_at
        )
        SELECT
          sc.import_id,
          sc.id,
          sc.source_row_number,
          sc.accession_number,
          1,
          (SELECT COUNT(*) FROM generated_entries ge WHERE ge.source_case_id = sc.id),
          sc.imported_at,
          sc.imported_at,
          sc.imported_at
        FROM source_cases sc
        WHERE sc.import_id IS NOT NULL
        """
    )
    ensure_columns(
        conn,
        "source_match_candidates",
        {
            "acgme_code": "TEXT",
            "event_key": "TEXT",
            "event_label": "TEXT",
        },
    )
    ensure_columns(
        conn,
        "generated_entries",
        {
            "acgme_code": "TEXT",
            "acgme_description": "TEXT",
            "acgme_def_category": "TEXT",
            "evidence_excerpt": "TEXT",
            "llm_model": "TEXT",
            "llm_prompt_version": "TEXT",
            "llm_raw_response_json": "TEXT",
            "mapping_pathway": "TEXT NOT NULL DEFAULT 'ollama'",
        },
    )
    ensure_columns(
        conn,
        "imports",
        {
            "mapping_pathway": "TEXT NOT NULL DEFAULT 'ollama'",
        },
    )
    conn.execute(
        """
        UPDATE source_cases
        SET llm_case_id = 'llm_' || substr(source_row_hash, 1, 16)
        WHERE llm_case_id IS NULL OR llm_case_id = ''
        """
    )
    conn.execute(
        """
        UPDATE generated_entries
        SET mapping_pathway = CASE
          WHEN mapping_rule_id = 'api-llm' THEN 'api'
          WHEN mapping_rule_id LIKE 'llm:%' THEN 'ollama'
          ELSE 'fuzzy'
        END
        WHERE mapping_pathway IS NULL OR mapping_pathway = ''
        """
    )
    conn.commit()


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
