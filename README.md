# ACGME IR Case Log Automation

Local-first tooling to convert an mPower IR CSV export into reviewed ACGME case-log entries and autofill the ACGME Add Cases form through a Chrome extension.

## Quick Start

Double-click `Start Case Logs.command` from Finder.

Or run the same launcher from Terminal:

```bash
./scripts/start_local.sh
```

This creates `.venv` if needed, installs Python dependencies, initializes the local database, starts the FastAPI server at `http://127.0.0.1:8765`, and opens the Streamlit review UI at `http://127.0.0.1:8501`.

Useful variants:

```bash
./scripts/start_local.sh api     # only the Chrome extension API
./scripts/start_local.sh review  # only the Streamlit review UI
```

For first-time data prep, run:

```bash
. .venv/bin/activate
python scripts/normalize_dropdowns.py
python -m app.cli import-mpower-csv data/exports/mpower-download-260526-clean.csv
```

Load `extension/` as an unpacked Chrome extension, log into ACGME normally, open Add Cases, then use the extension popup to preview, fill, and mark cases submitted. Keep `./scripts/start_local.sh` running while using the extension.

## Data Safety

The app stores only the source fields needed for case logging and review. `Patient ID`, `Study Instance UID`, and `Report ID` are not persisted or exported. Birth dates are used locally to calculate Adult/Pediatric status.

## Core Workflow

```text
mPower CSV
-> SQLite source_cases
-> mapping_rules.csv
-> generated_entries
-> Streamlit review / batch approve
-> FastAPI localhost queue
-> Chrome extension preview/fill
-> manual ACGME submit
-> extension marks submitted
-> SQLite audit trail
```

## Correcting And Learning Mappings

During review, use **Edit one generated entry** to correct Role, Area, Type, Keyword, Comments, or Component label. Leave **Learn this mapping correction for future imports** enabled to append a high-priority learned rule to `config/mapping_rules.csv`. Leave **Apply learned mapping to matching unsubmitted entries now** enabled to update all current unsubmitted entries with the same `Exam Code` and extracted procedure text.

For unmapped rows, use **Create manual entry from source case**. That can also learn a new mapping rule immediately, so future imports generate the corrected ACGME entry instead of returning to the Unmapped queue.

Known corrected starter mapping:

- `IRTRANJULV` / `Transjugular liver biopsy with pressure measurements` -> `Interventional Procedures / Biopsy / Biopsy transvenous`

## Status Semantics

Review statuses:

- `new_high_confidence`: reviewable high-confidence generated entry.
- `needs_review`: mapping, role, compound, aborted, or other ambiguity requires manual review.
- `approved`: user approved without field edits.
- `edited`: user edited and approved.
- `skipped`: permanent review decision; do not log this entry.

Upload statuses:

- `not_uploaded`: approved/edited and eligible for queue.
- `claimed`: locked by the extension queue.
- `autofilled`: filled into ACGME, not yet marked submitted.
- `submitted`: user submitted in ACGME and marked it submitted.
- `skipped_upload_session`: temporary skip for this upload session.
- `failed`: extension could not fill or validate the form.
- `reset`: recovered from autofilled/failed/skipped back to upload queue.
