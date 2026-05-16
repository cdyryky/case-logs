# ACGME IR Case Log Automation

Local-first tooling to convert a Visage/mPower IR XLSX export into reviewed ACGME case-log entries and autofill the ACGME Add Cases form through a Chrome extension.

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

python scripts/normalize_dropdowns.py
python -m app.cli init-db
python -m app.cli import-xlsx visage-data-export.xlsx
streamlit run app/review_app.py
uvicorn app.api:api --host 127.0.0.1 --port 8765
```

Load `extension/` as an unpacked Chrome extension, log into ACGME normally, open Add Cases, then use the extension popup to preview, fill, and mark cases submitted.

## Data Safety

The app stores only the source fields needed for case logging and review. `Patient ID`, `Study Instance UID`, and `Report ID` are not persisted or exported. Birth dates are used locally to calculate Adult/Pediatric status.

## Core Workflow

```text
Visage/mPower XLSX
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
