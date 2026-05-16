from __future__ import annotations

import csv
import re
import sqlite3
from pathlib import Path
from typing import Any

from .constants import CONFIG_DIR
from .mapper import load_mapping_rules
from .models import log_event, utc_now
from .utils import canonical_key, file_sha256

MAPPING_COLUMNS = [
    "priority",
    "active",
    "rule_id",
    "rule_version",
    "rule_name",
    "action",
    "match_exam_code",
    "match_study_description_regex",
    "match_procedure_regex",
    "match_report_regex",
    "exclude_procedure_regex",
    "exclude_report_regex",
    "case_class",
    "area",
    "type",
    "acgme_description",
    "acgme_def_category",
    "keyword",
    "component_label",
    "mapping_confidence",
    "needs_review_reason",
    "notes",
]


def _next_learned_version(path: Path) -> str:
    if not path.exists():
        return "1"
    learned = 0
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("rule_id") or "").startswith("learned_"):
                learned += 1
    return str(learned + 1)


def append_learned_rule(
    source: sqlite3.Row,
    values: dict[str, Any],
    mapping_path: str | Path | None = None,
) -> dict[str, str]:
    path = Path(mapping_path or CONFIG_DIR / "mapping_rules.csv")
    version = _next_learned_version(path)
    exam_code = source["exam_code"] or ""
    procedure_text = source["procedure_text"] or source["study_description"] or ""
    proc_pattern = f"^{re.escape(procedure_text)}$" if procedure_text else ""
    rule_id = "learned_" + canonical_key("_".join([exam_code, procedure_text, values["area"], values["type"]]))[:80]
    row = {
        "priority": "1000",
        "active": "true",
        "rule_id": rule_id,
        "rule_version": version,
        "rule_name": f"Learned: {exam_code} {procedure_text}".strip()[:120],
        "action": "generate",
        "match_exam_code": exam_code,
        "match_study_description_regex": "",
        "match_procedure_regex": proc_pattern,
        "match_report_regex": "",
        "exclude_procedure_regex": "",
        "exclude_report_regex": "",
        "case_class": values["case_class"],
        "area": values["area"],
        "type": values["type"],
        "acgme_description": values.get("acgme_description", ""),
        "acgme_def_category": values.get("acgme_def_category", ""),
        "keyword": values.get("keyword", ""),
        "component_label": values.get("component_label") or "dominant_procedure",
        "mapping_confidence": "high",
        "needs_review_reason": "",
        "notes": "Learned from review correction",
    }
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MAPPING_COLUMNS)
        writer.writerow(row)
    row["mapping_rules_file_hash"] = file_sha256(path)
    return row


def apply_mapping_to_matching_unsubmitted(
    conn: sqlite3.Connection,
    source: sqlite3.Row,
    values: dict[str, Any],
    learned_rule: dict[str, str],
    source_label: str = "review_app",
) -> int:
    now = utc_now()
    rows = conn.execute(
        """
        SELECT ge.id, ge.review_status, sc.accession_number, ge.case_date
        FROM generated_entries ge
        JOIN source_cases sc ON sc.id = ge.source_case_id
        WHERE COALESCE(sc.exam_code, '') = COALESCE(?, '')
          AND COALESCE(sc.procedure_text, '') = COALESCE(?, '')
          AND ge.upload_status != 'submitted'
        """,
        (source["exam_code"], source["procedure_text"]),
    ).fetchall()
    for row in rows:
        dedupe_key = "|".join(
            [
                row["accession_number"],
                row["case_date"],
                str(values["case_class"]),
                str(values["area"]),
                str(values["type"]),
                str(values.get("acgme_description") or ""),
                str(values.get("component_label") or "dominant_procedure"),
            ]
        )
        old = row["review_status"]
        conn.execute(
            """
            UPDATE generated_entries
            SET dedupe_key = ?, component_label = ?, case_class = ?, area = ?, type = ?,
                acgme_description = ?, acgme_def_category = ?, keyword = ?, comments = ?,
                mapping_rule_id = ?, mapping_rule_version = ?,
                mapping_rules_file_hash = ?, mapping_rule_name = ?, mapping_confidence = 'high',
                review_status = CASE WHEN role_confidence = 'high' AND compound_flag = 0
                                     THEN 'new_high_confidence' ELSE 'needs_review' END,
                updated_at = ?
            WHERE id = ?
            """,
            (
                dedupe_key,
                values.get("component_label") or "dominant_procedure",
                values["case_class"],
                values["area"],
                values["type"],
                values.get("acgme_description", ""),
                values.get("acgme_def_category", ""),
                values.get("keyword", ""),
                values.get("comments", ""),
                learned_rule["rule_id"],
                learned_rule["rule_version"],
                learned_rule["mapping_rules_file_hash"],
                learned_rule["rule_name"],
                now,
                row["id"],
            ),
        )
        log_event(conn, row["id"], "learned_mapping_applied", old, "new_high_confidence", source_label, learned_rule["rule_id"])
    conn.commit()
    return len(rows)


def learned_rule_count() -> int:
    rules, _ = load_mapping_rules()
    return sum(1 for rule in rules if rule.rule_id.startswith("learned_"))
