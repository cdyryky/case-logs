from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from openpyxl import load_workbook

from .config_io import load_resident_profile
from .constants import DEFAULT_CASE_CLASS, DEFAULT_SITE
from .utils import (
    case_year_from_date,
    extract_procedure_text,
    format_acgme_date,
    parse_principal_result_interpreter,
    parse_role_metadata,
    parse_study_datetime,
    patient_type,
    review_flags,
    row_hash,
)

REQUIRED_COLUMNS = [
    "Accession Number",
    "Exam Code",
    "Institution Name",
    "Modalities DICOM",
    "Modality",
    "Patient Birth Date",
    "Patient ID",
    "Principal Result Interpreter",
    "Report ID",
    "Report Snippet",
    "Study Date",
    "Study Description",
    "Study Instance UID",
]


def iter_raw_rows(xlsx_path: str | Path, sheet_name: str = "raw") -> Iterator[dict[str, Any]]:
    wb = load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb[sheet_name]
    header = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not any(v not in (None, "") for v in row):
            continue
        yield dict(zip(header, row))


def transform_source_row(raw: dict[str, Any], resident_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    profile = resident_profile or load_resident_profile()
    resident = profile["resident"]
    defaults = profile["defaults"]
    role_meta = parse_role_metadata(raw.get("Report Snippet"), resident.get("aliases", []))
    flags = review_flags(raw.get("Report Snippet"))
    study_dt = parse_study_datetime(raw["Study Date"])
    procedure_text = extract_procedure_text(raw.get("Report Snippet")) or str(raw.get("Study Description") or "")
    return {
        "accession_number": str(raw.get("Accession Number") or "").strip(),
        "study_date": study_dt.isoformat(),
        "patient_birth_date": str(raw.get("Patient Birth Date") or ""),
        "exam_code": str(raw.get("Exam Code") or "").strip(),
        "study_description": str(raw.get("Study Description") or "").strip(),
        "report_snippet": str(raw.get("Report Snippet") or ""),
        "procedure_text": procedure_text,
        "institution_name": str(raw.get("Institution Name") or "").strip(),
        "principal_result_interpreter_raw": str(raw.get("Principal Result Interpreter") or "").strip(),
        "attending_name": parse_principal_result_interpreter(raw.get("Principal Result Interpreter")),
        "source_row_hash": row_hash(raw),
        "resident_found_in_report": int(bool(role_meta["resident_found_in_report"])),
        "resident_position": role_meta["resident_position"],
        "role_parse_source": role_meta["role_parse_source"],
        "aborted_flag": int(flags["aborted_flag"]),
        "unsuccessful_flag": int(flags["unsuccessful_flag"]),
        "no_procedure_flag": int(flags["no_procedure_flag"]),
        "needs_review_reason": flags["needs_review_reason"],
        "derived": {
            "case_id": str(raw.get("Accession Number") or "").strip(),
            "case_date": format_acgme_date(raw["Study Date"]),
            "case_year": case_year_from_date(
                raw["Study Date"],
                int(resident["expected_graduation_year"]),
                int(resident.get("pgy_max", 5)),
            ),
            "role": role_meta["role"],
            "role_confidence": role_meta["role_confidence"],
            "site": defaults.get("site", DEFAULT_SITE),
            "patient_type": patient_type(
                raw["Study Date"],
                raw.get("Patient Birth Date"),
                int(defaults.get("pediatric_age_cutoff", 18)),
            ),
            "case_class": defaults.get("case_class", DEFAULT_CASE_CLASS),
        },
    }
