#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path
from typing import Any

import pandas as pd


KEEP_COLUMNS = [
    "Accession Number",
    "Modality",
    "Exam Code",
    "Exam Description",
    "CPT Code",
    "Report Text",
    "Patient Age",
    "Exam Started Date",
    "Report Finalized By",
]

TEXT_COLUMNS_TO_SCRUB = ["Report Text"]

PHI_PATTERNS = [
    (re.compile(r"\bMRN\s*[:#]?\s*[A-Z0-9-]{4,}\b", re.IGNORECASE), "MRN: [REDACTED]"),
    (re.compile(r"\bDOB\s*[:#]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.IGNORECASE), "DOB: [REDACTED]"),
    (re.compile(r"\b(?:patient\s+name|name)\s*[:#]?\s*[A-Z][A-Za-z'`-]+,\s*[A-Z][A-Za-z'`-]+\b", re.IGNORECASE), "Name: [REDACTED]"),
    (re.compile(r"\b(?:patient\s+name|name)\s*[:#]?\s*[A-Z][A-Za-z'`-]+\s+[A-Z][A-Za-z'`-]+\b", re.IGNORECASE), "Name: [REDACTED]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED-SSN]"),
    (re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"), "[REDACTED-PHONE]"),
]


def read_export(path: Path, sheet_name: str | int | None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name or 0, dtype=str, keep_default_na=False)
    raise ValueError(f"Unsupported input type {suffix!r}; use .csv, .xlsx, .xlsm, or .xls")


def scrub_report_text(value: Any) -> str:
    text = "" if value is None else str(value)
    for pattern, replacement in PHI_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def hash_accession(value: Any, salt: str) -> str:
    raw = "" if value is None else str(value).strip()
    if not raw:
        return ""
    digest = hashlib.sha256(f"{salt}:{raw}".encode("utf-8")).hexdigest()[:16]
    return f"ACC-{digest}"


def clean_export(
    input_path: Path,
    output_path: Path,
    sheet_name: str | int | None,
    salt: str,
    keep_original_accession: bool,
    accession_crosswalk_path: Path | None,
) -> int:
    df = read_export(input_path, sheet_name)
    missing = [column for column in KEEP_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {', '.join(missing)}")

    cleaned = df.loc[:, KEEP_COLUMNS].copy()
    for column in TEXT_COLUMNS_TO_SCRUB:
        cleaned[column] = cleaned[column].map(scrub_report_text)

    if not keep_original_accession:
        original_accessions = cleaned["Accession Number"].map(lambda value: "" if value is None else str(value).strip())
        hashed_accessions = original_accessions.map(lambda value: hash_accession(value, salt))
        cleaned["Accession Number"] = hashed_accessions

        if accession_crosswalk_path:
            crosswalk = pd.DataFrame(
                {
                    "Hashed Accession Number": hashed_accessions,
                    "Original Accession Number": original_accessions,
                }
            )
            crosswalk = crosswalk[crosswalk["Original Accession Number"] != ""].drop_duplicates()
            accession_crosswalk_path.parent.mkdir(parents=True, exist_ok=True)
            crosswalk.to_csv(accession_crosswalk_path, index=False, quoting=csv.QUOTE_MINIMAL)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".xlsx":
        cleaned.to_excel(output_path, index=False)
    else:
        cleaned.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return len(cleaned)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep selected mPower export columns and remove obvious PHI before import/review."
    )
    parser.add_argument("input", type=Path, help="Full mPower export, as CSV or XLSX.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Cleaned output path. Defaults to data/exports/<input-stem>-clean.csv.",
    )
    parser.add_argument("--sheet", default=None, help="XLSX sheet name. Defaults to the first sheet.")
    parser.add_argument(
        "--salt",
        default="case-logs",
        help="Salt used when hashing accession numbers. Keep stable if you want repeatable IDs.",
    )
    parser.add_argument(
        "--keep-original-accession",
        action="store_true",
        help="Keep accession numbers unchanged. Only use for local/private processing.",
    )
    parser.add_argument(
        "--accession-crosswalk",
        type=Path,
        help="Optional private CSV mapping hashed accession numbers back to real accession numbers.",
    )
    args = parser.parse_args()

    output = args.output or Path("data/exports") / f"{args.input.stem}-clean.csv"
    row_count = clean_export(
        input_path=args.input,
        output_path=output,
        sheet_name=args.sheet,
        salt=args.salt,
        keep_original_accession=args.keep_original_accession,
        accession_crosswalk_path=args.accession_crosswalk,
    )
    print(f"Wrote {row_count} rows to {output}")
    if args.accession_crosswalk and not args.keep_original_accession:
        print(f"Wrote private accession crosswalk to {args.accession_crosswalk}")


if __name__ == "__main__":
    main()
