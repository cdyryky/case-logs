from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import date, datetime
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def row_hash(values: dict[str, Any]) -> str:
    safe = {
        k: values.get(k)
        for k in [
            "Accession Number",
            "Exam Code",
            "Institution Name",
            "Patient Birth Date",
            "Principal Result Interpreter",
            "Report Snippet",
            "Study Date",
            "Study Description",
        ]
    }
    raw = repr(sorted((k, str(v)) for k, v in safe.items())).encode()
    return hashlib.sha256(raw).hexdigest()


def canonical_text(value: str | None) -> str:
    if not value:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def canonical_key(value: str | None) -> str:
    text = canonical_text(value)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text


def parse_study_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    text = str(value).strip()
    text = text.replace("Z", "+00:00")
    # Python accepts up to 6 fractional digits; Visage exports may include 7.
    text = re.sub(r"(\.\d{6})\d+([+-]\d\d:\d\d)$", r"\1\2", text)
    text = re.sub(r"(\.\d{6})\d+$", r"\1", text)
    return datetime.fromisoformat(text)


def format_acgme_date(value: Any) -> str:
    dt = parse_study_datetime(value)
    return f"{dt.month}/{dt.day}/{dt.year}"


def parse_birth_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    return parse_study_datetime(text).date()


def age_on_date(birth_date: date, study_dt: datetime) -> int:
    d = study_dt.date()
    return d.year - birth_date.year - ((d.month, d.day) < (birth_date.month, birth_date.day))


def patient_type(study_date: Any, birth_date: Any, pediatric_cutoff: int = 18) -> str:
    bd = parse_birth_date(birth_date)
    if bd is None:
        return "Adult"
    return "Pediatric" if age_on_date(bd, parse_study_datetime(study_date)) < pediatric_cutoff else "Adult"


def case_year_from_date(study_date: Any, graduation_year: int, pgy_max: int = 5) -> int:
    dt = parse_study_datetime(study_date)
    academic_year_end = dt.year + 1 if (dt.month, dt.day) >= (7, 1) else dt.year
    return pgy_max - (graduation_year - academic_year_end)


def parse_principal_result_interpreter(raw: str | None) -> str:
    if not raw:
        return ""
    parts = str(raw).split("^")
    if len(parts) >= 3:
        last = parts[1].replace("-", " ").title().replace(" ", "-")
        first = parts[2].replace("-", " ").title().replace(" ", "-")
        return f"{first} {last}".strip()
    return str(raw).strip()


def extract_procedure_text(report_snippet: str | None) -> str:
    if not report_snippet:
        return ""
    m = re.search(r"PROCEDURE:\s*(.*?)\s*Date of service:", str(report_snippet), flags=re.I | re.S)
    if not m:
        return ""
    return re.sub(r"\s+", " ", m.group(1)).strip()


def _alias_patterns(aliases: list[str]) -> list[re.Pattern[str]]:
    patterns: list[re.Pattern[str]] = []
    for alias in aliases:
        alias = alias.strip()
        if not alias:
            continue
        if "," in alias:
            last, first = [p.strip() for p in alias.split(",", 1)]
            pat = rf"\b{re.escape(last)}\s*,\s*{re.escape(first)}\b"
        else:
            pieces = alias.split()
            pat = r"\b" + r"\s+".join(re.escape(p) for p in pieces) + r"\b"
        patterns.append(re.compile(pat, re.I))
    return patterns


def parse_role_metadata(report_snippet: str | None, aliases: list[str]) -> dict[str, Any]:
    fallback = {
        "role": "Primary",
        "role_confidence": "low",
        "resident_found_in_report": False,
        "resident_position": None,
        "role_parse_source": "fallback",
    }
    if not report_snippet:
        return fallback
    text = str(report_snippet)
    section = re.search(
        r"Resident physician\(s\):\s*(.*?)(?:Advanced practice|Attending physician|Pre-procedure|Procedure details|Findings|$)",
        text,
        flags=re.I | re.S,
    )
    if not section:
        return fallback
    chunk = re.sub(r"\s+", " ", section.group(1)).strip()
    if not chunk:
        return fallback

    patterns = _alias_patterns(aliases)
    matches = [(p.search(chunk), p) for p in patterns]
    found = [m for m, _ in matches if m]
    if not found:
        return fallback

    separators = [m.start() for m in re.finditer(r"\s*(?:;|\band\b)\s*", chunk, flags=re.I)]
    starts = [0] + [s + 1 for s in separators]
    first_match = min(found, key=lambda m: m.start())
    position = 1 + sum(1 for s in starts[1:] if s < first_match.start())
    return {
        "role": "Primary" if position == 1 else "Secondary",
        "role_confidence": "high",
        "resident_found_in_report": True,
        "resident_position": position,
        "role_parse_source": "report_snippet",
    }


REVIEW_FLAG_PATTERNS = {
    "aborted_flag": re.compile(r"\baborted\b", re.I),
    "unsuccessful_flag": re.compile(r"\bunsuccessful\b", re.I),
    "no_procedure_flag": re.compile(
        r"procedure was not performed|no intervention was performed|\bdeferred\b", re.I
    ),
}


def review_flags(report_snippet: str | None) -> dict[str, Any]:
    text = str(report_snippet or "")
    flags = {name: bool(rx.search(text)) for name, rx in REVIEW_FLAG_PATTERNS.items()}
    reasons = [name.replace("_flag", "") for name, value in flags.items() if value]
    flags["needs_review_reason"] = ", ".join(reasons)
    return flags
