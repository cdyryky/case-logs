from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
IMPORTS_DIR = DATA_DIR / "imports"
EXPORTS_DIR = DATA_DIR / "exports"
DEFAULT_DB_PATH = DATA_DIR / "case_logs.sqlite"

REVIEW_STATUSES = {
    "new_high_confidence",
    "needs_review",
    "approved",
    "edited",
    "skipped",
}

UPLOAD_STATUSES = {
    "not_uploaded",
    "claimed",
    "autofilled",
    "submitted",
    "skipped_upload_session",
    "failed",
    "reset",
}

ROLE_ALLOWED = {"Primary", "Secondary"}

DEFAULT_SITE = "University of California (Davis) Medical Center"
DEFAULT_CASE_CLASS = "Interventional Procedures"

