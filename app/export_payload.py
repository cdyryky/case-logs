from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .constants import EXPORTS_DIR
from .upload_queue import payload_from_entry


def approved_entries(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT * FROM generated_entries
        WHERE review_status IN ('approved', 'edited')
          AND upload_status IN ('not_uploaded', 'reset', 'failed')
        ORDER BY id
        """
    ).fetchall()
    return [payload_from_entry(row) for row in rows]


def export_approved_json(conn: sqlite3.Connection, path: str | Path | None = None) -> Path:
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(path or EXPORTS_DIR / "approved_cases.json")
    out.write_text(json.dumps({"entries": approved_entries(conn)}, indent=2))
    return out
