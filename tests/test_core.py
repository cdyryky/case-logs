from __future__ import annotations

import csv
import json
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
from datetime import datetime

from app.candidates import (
    ALGORITHM_VERSION,
    add_manual_candidate,
    add_manual_candidates,
    approve_candidate_review,
    build_match_candidates,
    load_acgme_targets,
    load_candidates,
    search_acgme_targets,
    store_match_candidates,
)
from app.learning import append_learned_rule
from app.importer import import_mpower_csv, insert_generated_entries, insert_source_case
from app.llm_client import LLMClientError, LLMSettings, OllamaClient
from app.llm_mapping import LLM_OUTPUT_SCHEMA, build_prompt, llm_entries_for_source
from app.matching import match_tokens
from app.mapper import load_mapping_rules, map_source_case
from app.models import init_db
from app.parser import parse_mpower_report, parse_mpower_role_metadata, transform_mpower_row
from app.review_queue import best_report_context_for_source, load_next_candidate_group, remap_unresolved_cases, review_counts
from app.review_queue import clear_deterministic_review_state
from app.upload_queue import (
    claim_next,
    claim_next_group,
    get_current,
    save_group_edit,
    update_group_upload_status,
    update_upload_status,
)
from app.utils import case_year_from_date, format_acgme_date, patient_type


def transform_report_fixture(raw: dict[str, object]) -> dict[str, object]:
    report_text = str(raw.get("Report Snippet") or "")
    if "Cody Key" in report_text and "Procedural Personnel" not in report_text and "PROCEDURE PERSONNEL" not in report_text:
        report_text = f"{report_text}\n\nProcedural Personnel\nResident physician(s): Cody Key, MD\n"
    return transform_mpower_row(
        {
            "Accession Number": raw.get("Accession Number"),
            "Modality": raw.get("Modality") or raw.get("Modalities DICOM") or "IR",
            "Exam Code": raw.get("Exam Code"),
            "Exam Description": raw.get("Study Description"),
            "CPT Code": raw.get("CPT Code") or "",
            "Report Text": report_text,
            "Patient Age": raw.get("Patient Age") or "46",
            "Exam Started Date": raw.get("Study Date"),
            "Report Finalized By": raw.get("Report Finalized By") or "Attending, Example",
        }
    )


class FakeLLMClient:
    def __init__(self, responses: list[dict[str, object]] | None = None, error: Exception | None = None) -> None:
        self.responses = list(responses or [])
        self.error = error
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.settings = SimpleNamespace(model="gemma4:latest")

    def generate_json(self, prompt: str, schema: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
        self.calls.append((prompt, schema))
        if self.error:
            raise self.error
        if not self.responses:
            raise AssertionError("FakeLLMClient received more calls than expected")
        payload = self.responses.pop(0)
        return payload, {"response": json.dumps(payload)}


class CoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_mapping_mode = os.environ.get("ACGME_MAPPING_MODE")
        os.environ["ACGME_MAPPING_MODE"] = "legacy"

    def tearDown(self) -> None:
        if self._old_mapping_mode is None:
            os.environ.pop("ACGME_MAPPING_MODE", None)
        else:
            os.environ["ACGME_MAPPING_MODE"] = self._old_mapping_mode

    def test_init_db_commits_bootstrap_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "case_logs.sqlite"
            conn = sqlite3.connect(db_path, timeout=0.1)
            conn.row_factory = sqlite3.Row
            init_db(conn)
            self.assertFalse(conn.in_transaction)

            other = sqlite3.connect(db_path, timeout=0.1)
            try:
                other.execute(
                    """
                    INSERT INTO entry_events(entry_id, event_type, timestamp, source)
                    VALUES (NULL, 'smoke', '2026-01-01T00:00:00+00:00', 'test')
                    """
                )
                other.commit()
            finally:
                other.close()
                conn.close()

    def test_case_year_boundaries(self) -> None:
        self.assertEqual(case_year_from_date("2025-07-01T00:00:00-07:00", 2026), 5)
        self.assertEqual(case_year_from_date("2025-06-30T00:00:00-07:00", 2026), 4)
        self.assertEqual(case_year_from_date("2024-07-01T00:00:00-07:00", 2026), 4)

    def test_patient_type_cutoff(self) -> None:
        self.assertEqual(patient_type("2026-05-15T00:00:00-07:00", datetime(2008, 5, 16)), "Pediatric")
        self.assertEqual(patient_type("2026-05-15T00:00:00-07:00", datetime(2008, 5, 15)), "Adult")

    def test_date_format(self) -> None:
        self.assertEqual(format_acgme_date("2026-05-15T09:57:03.0000000-07:00"), "5/15/2026")

    def test_parser_role_and_mapping(self) -> None:
        raw = {
            "Accession Number": 202605150844,
            "Exam Code": "IRFLUROCASCACCESS",
            "Institution Name": "UC Davis Health",
            "Patient Birth Date": datetime(1981, 12, 8),
            "Principal Result Interpreter": "25727^MORSHEDI^MAUD^MOSTAFA",
            "Report Snippet": (
                "PROCEDURE: Venous port placement\n\n"
                "Procedural Personnel\n"
                "Attending physician(s): Example Attending, MD\n"
                "Resident physician(s): Cody Key, MD\n\n"
                "IMPRESSION:\n"
                "Successful venous port placement.\n\n"
                "PROCEDURE SUMMARY:\n"
                "- Venous port placement\n"
                "- Additional procedure(s): None\n\n"
                "PROCEDURE DETAILS:\n"
                "Details omitted.\n"
            ),
            "Study Date": "2026-05-15T09:57:03.0000000-07:00",
            "Study Description": "IR FLUOROSCOPY GUIDED VASCULAR ACCESS DEVICE PLACEMENT",
        }
        source = transform_report_fixture(raw)
        parsed = json.loads(source["parsed_report_json"])
        self.assertEqual(source["derived"]["role"], "Primary")
        self.assertEqual(source["derived"]["role_confidence"], "high")
        self.assertEqual(parsed["procedure_title"], "Venous port placement")
        rules, rules_hash = load_mapping_rules()
        entries, update = map_source_case(source, rules, rules_hash)
        self.assertEqual(update["source_mapping_status"], "generated")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["area"], "Venous Access General")
        self.assertEqual(entries[0]["type"], "Venous port placement")
        self.assertEqual(entries[0]["review_status"], "new_high_confidence")

    def test_mpower_transform_parses_decimal_age_and_metadata(self) -> None:
        raw = {
            "Accession Number": "202501300043",
            "Modality": "US",
            "Exam Code": "USGUDPARAC",
            "Exam Description": "US GUIDED PARACENTESIS",
            "CPT Code": "",
            "Report Text": (
                "ULTRASOUND-GUIDED PARACENTESIS\n"
                "EXAM DATE: 1/30/2025 8:00 AM\n\n"
                "PROCEDURE PERSONNEL:\n"
                "Attending: Example Attending, MD\n"
                "Other: Cody Key, M.D.\n\n"
                "TECHNIQUE:\n"
                "Ultrasound-guided diagnostic paracentesis was performed.\n"
            ),
            "Patient Age": "1.58",
            "Exam Started Date": "2025-01-30 08:00:00-08:00",
            "Report Finalized By": "Attending, Example",
            "__source_row_number": 2,
        }
        source = transform_mpower_row(raw, duplicate_accession=True)
        parsed = json.loads(source["parsed_report_json"])
        self.assertEqual(source["source_format"], "mpower_csv")
        self.assertEqual(source["source_row_number"], 2)
        self.assertEqual(source["patient_age_years"], "1.58")
        self.assertEqual(source["derived"]["patient_type"], "Pediatric")
        self.assertEqual(source["attending_name"], "Example Attending")
        self.assertEqual(source["cpt_code"], "")
        self.assertEqual(source["role_parse_source"], "personnel_other_line")
        self.assertEqual(source["derived"]["role"], "Primary")
        self.assertEqual(parsed["procedure_title"], "ULTRASOUND-GUIDED PARACENTESIS")
        self.assertIn("duplicate_accession", source["needs_review_reason"])

    def test_mpower_report_parser_handles_numbered_procedures_and_prefixed_summary(self) -> None:
        report = (
            "PROCEDURES:\n"
            "1. Inferior vena cava filter insertion\n"
            "2. Genitourinary catheter exchange\n"
            "3. Renal transarterial embolization\n\n"
            "Date of service: 7/27/2025 3:53 PM\n\n"
            "Procedural Personnel\n"
            "Attending physician(s): Example Attending, MD\n"
            "Resident physician(s): Cody Key, MD\n\n"
            "IMPRESSION:\n"
            "1. Insertion of inferior vena cava filter.\n"
            "2. Right renal angiography with embolization.\n\n"
            "IVC FILTER PLACEMENT PROCEDURE SUMMARY:\n"
            "- IVC filter insertion under fluoroscopic guidance\n"
            "- Additional procedure(s): None\n\n"
            "RENAL ARTERIOGRAPHY AND EMBOLIZATION PROCEDURE SUMMARY:\n"
            "- Renal angiography and embolization\n"
            "- Additional procedure(s): Cone-beam CT\n\n"
            "PROCEDURE DETAILS:\n"
            "Details omitted.\n"
        )
        parsed = parse_mpower_report(report, ["Cody Key"])
        self.assertEqual(parsed["procedure_title_source"], "procedures_label")
        self.assertIn("Renal transarterial embolization", parsed["procedure_list"])
        self.assertEqual(len(parsed["procedure_summary_sections"]), 2)
        self.assertIn("Cone-beam CT", parsed["candidate_procedure_phrases"])
        self.assertNotIn("missing_impression", parsed["parse_warnings"])

    def test_mpower_report_parser_handles_date_first_resident_block(self) -> None:
        report = (
            "DATE OF PROCEDURE: 3/27/25\n\n"
            "PROCEDURE:\n"
            "1. Lumbar spinal puncture with fluoroscopic guidance\n"
            "2. Digital subtraction myelogram\n\n"
            "SURGEON:\n"
            "Example Surgeon, MD\n\n"
            "RESIDENT:\n"
            "Clayton Example, MD\n"
            "Cody Key, MD\n\n"
            "SEDATION:\n"
            "Moderate sedation.\n"
        )
        parsed = parse_mpower_report(report, ["Cody Key"])
        role = parse_mpower_role_metadata(report, ["Cody Key"])
        self.assertIn("Digital subtraction myelogram", parsed["procedure_title"])
        self.assertEqual(role["role_parse_source"], "resident_heading")
        self.assertEqual(role["role"], "Secondary")
        self.assertEqual(role["role_confidence"], "high")

    def test_mpower_report_parser_excludes_signature_footer_from_impression(self) -> None:
        report = (
            "US GUIDED SUPERFICIAL BIOPSY/ASPIRATIONS\n"
            "EXAM DATE: 2/6/2024 9:49 AM\n\n"
            "PROCEDURE PERSONNEL:\n"
            "Attending: Example Attending\n"
            "Other: Cody Key\n\n"
            "IMPRESSION:\n"
            "1. Successful ultrasound-guided abdominal fat pad core needle biopsy.\n\n"
            "Preliminary Report - subject to revision until finalized: Cody Key, MD on 2/6/2024 10:21 AM\n"
            "Attending note: I was physically present for the key portion(s) of the procedure and immediately available for the entire procedure.\n"
            "I have personally reviewed the images of this study and agree with the above report.\n"
            "Final Report Electronically Signed By: Example Attending on 2/6/2024 11:08 AM\n"
        )
        parsed = parse_mpower_report(report, ["Cody Key"])
        self.assertEqual(
            parsed["impression"],
            "1. Successful ultrasound-guided abdominal fat pad core needle biopsy.",
        )
        self.assertIn(
            "Successful ultrasound-guided abdominal fat pad core needle biopsy.",
            parsed["candidate_procedure_phrases"],
        )
        self.assertFalse(
            any("Preliminary Report" in phrase or "Final Report" in phrase for phrase in parsed["candidate_procedure_phrases"])
        )

    def test_mpower_report_parser_uses_findings_impression_as_impression(self) -> None:
        report = (
            "ULTRASOUND-GUIDED PARACENTESIS\n"
            "EXAM DATE: 2/6/2024 11:48 AM\n\n"
            "PROCEDURE PERSONNEL:\n"
            "Attending: Example Attending\n"
            "Other: Cody Key\n\n"
            "FINDINGS/IMPRESSION:\n"
            "1. Successful ultrasound-guided diagnostic paracentesis with removal of 0.1L of serous fluid.\n\n"
            "Preliminary Report - subject to revision until finalized: Cody Key, MD on 2/6/2024 1:46 PM\n"
        )
        parsed = parse_mpower_report(report, ["Cody Key"])
        self.assertEqual(
            parsed["impression"],
            "1. Successful ultrasound-guided diagnostic paracentesis with removal of 0.1L of serous fluid.",
        )
        self.assertEqual(parsed["fallback_findings"], "")
        self.assertNotIn("missing_impression", parsed["parse_warnings"])
        self.assertIn(
            "Successful ultrasound-guided diagnostic paracentesis with removal of 0.1L of serous fluid.",
            parsed["candidate_procedure_phrases"],
        )

    def test_mpower_report_parser_handles_colonless_procedure_summary(self) -> None:
        report = (
            "PROCEDURE: Genitourinary catheter exchange\n\n"
            "Procedural Personnel\n"
            "Attending physician(s): Example Attending, MD\n"
            "Resident physician(s): Cody Key, MD\n\n"
            "IMPRESSION:\n"
            "1. Serial UPJ ureteroplasty.\n"
            "2. Transplant nephroureteral stent exchange/upsize.\n\n"
            "Plan:\n"
            "Tube(s) capped. Return in 2 weeks.\n\n"
            "PROCEDURE SUMMARY\n"
            "- Target organ: Transplant kidney\n"
            "- Antegrade nephrostogram(s) via the existing access\n"
            "- Nephroureteral tube exchange\n"
            "- Additional procedure(s): None\n\n"
            "PROCEDURE DETAILS:\n"
            "Details omitted.\n"
        )
        parsed = parse_mpower_report(report, ["Cody Key"])
        self.assertEqual(len(parsed["procedure_summary_sections"]), 1)
        bullets = parsed["procedure_summary_sections"][0]["bullets"]
        self.assertIn("Target organ: Transplant kidney", bullets)
        self.assertIn("Nephroureteral tube exchange", bullets)
        self.assertEqual(parsed["procedure_summary_sections"][0]["additional_procedures"], [])
        self.assertNotIn("missing_procedure_summary", parsed["parse_warnings"])

    def test_report_parser_handles_inline_impression_in_report_text(self) -> None:
        report = (
            "Cody Key, MD. IMPRESSION: SUCCESSFUL PLACEMENT OF A RIGHT CHEST 8F SINGLE LUMEN POWERPORT "
            "VIA THE RIGHT INTERNAL JUGULAR VEIN. THE CATHETER IS READY FOR IMMEDIATE USE."
        )
        parsed = parse_mpower_report(report, ["Cody Key"])
        self.assertEqual(
            parsed["impression"],
            "SUCCESSFUL PLACEMENT OF A RIGHT CHEST 8F SINGLE LUMEN POWERPORT VIA THE RIGHT INTERNAL JUGULAR VEIN. THE CATHETER IS READY FOR IMMEDIATE USE.",
        )
        self.assertEqual(parsed["procedure_title"], "")
        self.assertNotIn("missing_impression", parsed["parse_warnings"])
        self.assertIn("THE CATHETER IS READY FOR IMMEDIATE USE.", parsed["candidate_procedure_phrases"][0])

    def test_report_parser_handles_inline_procedure_summary_in_report_text(self) -> None:
        report = (
            "Resident physician(s): Cody Key M.D. PROCEDURE SUMMARY: - Target organ: Left kidney "
            "- Image-guided heat-based ablation - Additional procedure(s): Fine-needle aspiration biopsy."
        )
        parsed = parse_mpower_report(report, ["Cody Key"])
        summaries = parsed["procedure_summary_sections"]
        self.assertEqual(len(summaries), 1)
        self.assertEqual(parsed["procedure_title"], "")
        self.assertIn("Target organ: Left kidney", summaries[0]["bullets"])
        self.assertIn("Image-guided heat-based ablation", summaries[0]["bullets"])
        self.assertEqual(summaries[0]["additional_procedures"], ["Fine-needle aspiration biopsy."])
        self.assertNotIn("missing_procedure_summary", parsed["parse_warnings"])

    def test_report_context_prefers_richer_duplicate_accession_report(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        init_db(conn)
        accession = "202604221898"
        short_source = transform_report_fixture(
            {
                "Accession Number": accession,
                "Exam Code": "IRBIOPSY",
                "Institution Name": "UC Davis Health",
                "Patient Birth Date": datetime(1980, 1, 1),
                "Principal Result Interpreter": "25727^VU^CATHERINE",
                "Report Snippet": (
                    "PROCEDURE: Ultrasound-guided biopsy Date of service: 4/22/2026 1:00 PM "
                    "Procedural Personnel Attending physician(s): Catherine Vu, MD"
                ),
                "Study Date": "2026-04-22T13:00:00.0000000-07:00",
                "Study Description": "US GUIDED NEEDLE PLACEMENT WITH RADIOLOGIST",
            }
        )
        full_source = transform_mpower_row(
            {
                "Accession Number": accession,
                "Modality": "US",
                "Exam Code": "IRBIOPSY",
                "Exam Description": "US GUIDED NEEDLE PLACEMENT WITH RADIOLOGIST",
                "CPT Code": "",
                "Report Text": (
                    "PROCEDURE: Ultrasound-guided biopsy\n\n"
                    "IMPRESSION:\n\n"
                    "Ultrasound-guided non-targeted biopsy of left renal cortex with specimen(s)\n"
                    "sent to pathology.\n\n"
                    "Plan:\n"
                    "Postprocedural monitoring.\n\n"
                    "PROCEDURE SUMMARY:\n"
                    "- Percutaneous US-guided coaxial core needle biopsy\n"
                    "- Additional procedure(s): None\n\n"
                    "PROCEDURE DETAILS:\n"
                    "Details omitted.\n"
                ),
                "Patient Age": "46",
                "Exam Started Date": "2026-04-22 13:00:00-07:00",
                "Report Finalized By": "Vu, Catherine",
            }
        )
        short_id, _ = insert_source_case(conn, short_source, "short-report.csv", 1)
        full_id, _ = insert_source_case(conn, full_source, "mpower.csv", 1)
        short_row = conn.execute("SELECT * FROM source_cases WHERE id = ?", (short_id,)).fetchone()
        context = best_report_context_for_source(conn, short_row)
        self.assertEqual(context["source_case_id"], full_id)
        self.assertIn("left renal cortex", context["parsed"]["impression"])
        self.assertEqual(len(context["parsed"]["procedure_summary_sections"]), 1)

    def test_mpower_role_ignores_other_outside_personnel_section(self) -> None:
        report = (
            "PROCEDURE: Drainage catheter check\n\n"
            "Procedural Personnel\n"
            "Attending physician(s): Example Attending, MD\n"
            "Resident physician(s): None\n\n"
            "FINDINGS:\n"
            "Other: Cody Key observed a small residual collection.\n"
        )
        role = parse_mpower_role_metadata(report, ["Cody Key"])
        self.assertEqual(role["role_confidence"], "low")
        self.assertEqual(role["role_parse_source"], "personnel_no_resident_match")

    def test_mpower_role_marks_fellow_only_match_low_confidence(self) -> None:
        report = (
            "PROCEDURE: Fluid collection aspiration\n\n"
            "Procedural Personnel\n"
            "Attending physician(s): Example Attending, MD\n"
            "Fellow physician(s): Cody Key, MD\n"
            "Resident physician(s): None\n"
            "Advanced practice provider(s): None\n\n"
            "IMPRESSION:\n"
            "Successful aspiration.\n"
        )
        role = parse_mpower_role_metadata(report, ["Cody Key"])
        self.assertEqual(role["role_parse_source"], "non_resident_personnel_line")
        self.assertEqual(role["role_confidence"], "low")
        self.assertTrue(role["resident_found_in_report"])

    def test_acgme_target_loader_uses_confirmed_csv(self) -> None:
        targets = load_acgme_targets()
        self.assertEqual(len(targets), 365)
        self.assertEqual(len({target.acgme_code for target in targets}), 365)
        by_description = {target.acgme_description: target for target in targets}
        retrieval = by_description["IVC filter retrieval"]
        self.assertEqual(retrieval.acgme_code, "31742")
        self.assertEqual(retrieval.area, "Venous Interventions")
        self.assertEqual(retrieval.type, "Venous foreign body retrieval")
        self.assertEqual(retrieval.acgme_def_category, "Venous Intervention")
        placement = by_description["IVC filter placement"]
        self.assertEqual(placement.acgme_code, "31740")
        nephroureteral = by_description["Nephroureteral stent change"]
        self.assertEqual(nephroureteral.acgme_code, "31852")
        self.assertEqual(nephroureteral.area, "GU Intervention")
        self.assertEqual(nephroureteral.type, "GU tube/stent exchange")
        self.assertEqual(nephroureteral.acgme_def_category, "Catheter exchange")
        stricture = by_description["GU stricture dilation"]
        self.assertEqual(stricture.acgme_code, "31840")

    def test_llm_schema_prompt_and_client_context_settings(self) -> None:
        source = transform_mpower_row(
            {
                "Accession Number": "202601050001",
                "Modality": "IR",
                "Exam Code": "USGUDPARAC",
                "Exam Description": "US GUIDED PARACENTESIS",
                "CPT Code": "",
                "Report Text": (
                    "PROCEDURE: Ultrasound-guided paracentesis\n\n"
                    "IMPRESSION:\n"
                    "Successful paracentesis.\n\n"
                    "TECHNIQUE:\n"
                    "Technique details should not be sent.\n"
                ),
                "Patient Age": "42",
                "Exam Started Date": "2026-01-05 10:00:00-08:00",
                "Report Finalized By": "Attending, Example",
            }
        )
        prompt = build_prompt(source)
        self.assertNotIn("procedure_label", json.dumps(LLM_OUTPUT_SCHEMA))
        self.assertNotIn("procedure_label", prompt)
        self.assertIn("Successful paracentesis", prompt)
        self.assertNotIn("Technique details should not be sent", prompt)

        captured: dict[str, object] = {}

        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {"response": json.dumps({"procedures": [], "warnings": []})}

        def fake_post(url: str, json: dict[str, object], timeout: float) -> FakeResponse:
            captured["url"] = url
            captured["json"] = json
            captured["timeout"] = timeout
            return FakeResponse()

        client = OllamaClient(LLMSettings(base_url="http://127.0.0.1:11434", model="gemma4:latest", timeout_seconds=3, num_ctx=8192))
        with patch("app.llm_client.httpx.post", fake_post):
            client.generate_json("prompt", LLM_OUTPUT_SCHEMA)
        payload = captured["json"]
        self.assertEqual(payload["model"], "gemma4:latest")
        self.assertGreaterEqual(payload["options"]["num_ctx"], 8192)

    def test_llm_extraction_validates_codes_and_retries_invalid_response(self) -> None:
        source = transform_mpower_row(
            {
                "Accession Number": "202601060001",
                "Modality": "US",
                "Exam Code": "USGUDPARAC",
                "Exam Description": "US GUIDED PARACENTESIS",
                "CPT Code": "49083",
                "Report Text": (
                    "PROCEDURE: Ultrasound-guided paracentesis\n\n"
                    "IMPRESSION:\n"
                    "Successful ultrasound-guided paracentesis.\n"
                ),
                "Patient Age": "42",
                "Exam Started Date": "2026-01-06 10:00:00-08:00",
                "Report Finalized By": "Attending, Example",
            }
        )
        os.environ["ACGME_MAPPING_MODE"] = "llm"
        client = FakeLLMClient(
            [
                {
                    "procedures": [
                        {
                            "acgme_code": "99999",
                            "evidence_excerpt": "Successful ultrasound-guided paracentesis.",
                            "confidence": "high",
                            "rationale": "Invalid code.",
                        }
                    ],
                    "warnings": [],
                },
                {
                    "procedures": [
                        {
                            "acgme_code": "31896",
                            "evidence_excerpt": "Successful ultrasound-guided paracentesis.",
                            "confidence": "high",
                            "rationale": "Paracentesis performed.",
                        }
                    ],
                    "warnings": [],
                },
            ]
        )
        extraction = llm_entries_for_source(source, client=client)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(len(extraction.entries), 1)
        self.assertEqual(extraction.entries[0]["acgme_code"], "31896")
        self.assertEqual(extraction.entries[0]["review_status"], "needs_review")
        self.assertEqual(extraction.entries[0]["area"], "Drainage Procedures")

    def test_import_mpower_csv_uses_llm_for_compound_case_and_reports_progress(self) -> None:
        os.environ["ACGME_MAPPING_MODE"] = "llm"
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        init_db(conn)
        row = {
            "Accession Number": "202601070001",
            "Modality": "IR",
            "Exam Code": "IRIVCFIL",
            "Exam Description": "IR IVC FILTER PLACEMENT",
            "CPT Code": "",
            "Report Text": (
                "PROCEDURES:\n"
                "1. Inferior vena cava filter insertion\n"
                "2. Renal transarterial embolization\n\n"
                "IMPRESSION:\n"
                "1. Insertion of inferior vena cava filter.\n"
                "2. Right renal angiography with embolization.\n"
            ),
            "Patient Age": "42",
            "Exam Started Date": "2026-01-07 10:00:00-08:00",
            "Report Finalized By": "Attending, Example",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mpower.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            events: list[dict[str, object]] = []
            summary = import_mpower_csv(
                conn,
                path,
                progress_callback=events.append,
                llm_client=FakeLLMClient(
                    [
                        {
                            "procedures": [
                                {
                                    "acgme_code": "31740",
                                    "evidence_excerpt": "Insertion of inferior vena cava filter.",
                                    "confidence": "high",
                                    "rationale": "IVC filter placement performed.",
                                },
                                {
                                    "acgme_code": "31682",
                                    "evidence_excerpt": "Right renal angiography with embolization.",
                                    "confidence": "medium",
                                    "rationale": "Arterial embolization performed.",
                                },
                            ],
                            "warnings": ["compound case"],
                        }
                    ]
                ),
            )
        self.assertEqual(summary["generated_entries_count"], 2)
        rows = conn.execute("SELECT acgme_code, review_status, compound_flag, evidence_excerpt, llm_model FROM generated_entries ORDER BY id").fetchall()
        self.assertEqual([row["acgme_code"] for row in rows], ["31740", "31682"])
        self.assertEqual({row["review_status"] for row in rows}, {"needs_review"})
        self.assertEqual({row["compound_flag"] for row in rows}, {1})
        self.assertTrue(all(row["evidence_excerpt"] for row in rows))
        self.assertEqual({row["llm_model"] for row in rows}, {"gemma4:latest"})
        source_status = conn.execute("SELECT source_mapping_status, needs_review_reason FROM source_cases").fetchone()
        self.assertEqual(source_status["source_mapping_status"], "generated_llm")
        self.assertIn("compound case", source_status["needs_review_reason"])
        self.assertTrue(any(event["phase"] == "Mapping with local LLM" for event in events))

    def test_import_mpower_csv_commit_per_row_makes_entries_visible_during_import(self) -> None:
        os.environ["ACGME_MAPPING_MODE"] = "llm"
        rows = [
            {
                "Accession Number": "202601070010",
                "Modality": "US",
                "Exam Code": "USGUDPARAC",
                "Exam Description": "US GUIDED PARACENTESIS",
                "CPT Code": "49083",
                "Report Text": "PROCEDURE: Paracentesis\n\nIMPRESSION:\nSuccessful paracentesis.\n",
                "Patient Age": "42",
                "Exam Started Date": "2026-01-07 10:00:00-08:00",
                "Report Finalized By": "Attending, Example",
            },
            {
                "Accession Number": "202601070011",
                "Modality": "IR",
                "Exam Code": "IRIVCFIL",
                "Exam Description": "IR IVC FILTER PLACEMENT",
                "CPT Code": "",
                "Report Text": "PROCEDURE: IVC filter placement\n\nIMPRESSION:\nSuccessful IVC filter placement.\n",
                "Patient Age": "42",
                "Exam Started Date": "2026-01-07 11:00:00-08:00",
                "Report Finalized By": "Attending, Example",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "case_logs.sqlite"
            csv_path = Path(tmp) / "mpower.csv"
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout = 30000")
            conn.execute("PRAGMA journal_mode = WAL")
            init_db(conn)
            visible_after_first_commit: list[int] = []
            source_visible_before_mapping: list[int] = []

            def on_progress(event: dict[str, object]) -> None:
                if event.get("phase") == "Queued for mapping" and event.get("row") == 1:
                    other = sqlite3.connect(db_path, timeout=30)
                    try:
                        other.execute("PRAGMA busy_timeout = 30000")
                        source_visible_before_mapping.append(other.execute("SELECT COUNT(*) FROM source_cases").fetchone()[0])
                    finally:
                        other.close()
                if event.get("phase") == "Committed row" and event.get("row") == 1:
                    other = sqlite3.connect(db_path, timeout=30)
                    try:
                        other.execute("PRAGMA busy_timeout = 30000")
                        visible_after_first_commit.append(other.execute("SELECT COUNT(*) FROM generated_entries").fetchone()[0])
                    finally:
                        other.close()

            import_mpower_csv(
                conn,
                csv_path,
                progress_callback=on_progress,
                commit_per_row=True,
                llm_client=FakeLLMClient(
                    [
                        {
                            "procedures": [
                                {
                                    "acgme_code": "31896",
                                    "evidence_excerpt": "Successful paracentesis.",
                                    "confidence": "high",
                                    "rationale": "Paracentesis performed.",
                                }
                            ],
                            "warnings": [],
                        },
                        {
                            "procedures": [
                                {
                                    "acgme_code": "31740",
                                    "evidence_excerpt": "Successful IVC filter placement.",
                                    "confidence": "high",
                                    "rationale": "IVC filter placement performed.",
                                }
                            ],
                            "warnings": [],
                        },
                    ]
                ),
            )
            conn.close()
        self.assertEqual(source_visible_before_mapping, [1])
        self.assertEqual(visible_after_first_commit, [1])

    def test_import_mpower_csv_falls_back_when_llm_unavailable(self) -> None:
        os.environ["ACGME_MAPPING_MODE"] = "llm"
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        init_db(conn)
        row = {
            "Accession Number": "202601080001",
            "Modality": "IR",
            "Exam Code": "IRFLUROCASCACCESS",
            "Exam Description": "IR FLUOROSCOPY GUIDED VASCULAR ACCESS DEVICE PLACEMENT",
            "CPT Code": "",
            "Report Text": (
                "PROCEDURE: Venous port placement\n\n"
                "Procedural Personnel\n"
                "Resident physician(s): Cody Key, MD\n\n"
                "IMPRESSION:\n"
                "Successful venous port placement.\n"
            ),
            "Patient Age": "42",
            "Exam Started Date": "2026-01-08 10:00:00-08:00",
            "Report Finalized By": "Attending, Example",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mpower.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            summary = import_mpower_csv(
                conn,
                path,
                llm_client=FakeLLMClient(error=LLMClientError("connection refused")),
            )
        self.assertEqual(summary["generated_entries_count"], 1)
        source_status = conn.execute("SELECT source_mapping_status, needs_review_reason FROM source_cases").fetchone()
        self.assertEqual(source_status["source_mapping_status"], "llm_failed_fallback_generated")
        self.assertIn("LLM failed; legacy fallback used", source_status["needs_review_reason"])
        row = conn.execute("SELECT type FROM generated_entries").fetchone()
        self.assertEqual(row["type"], "Venous port placement")

    def test_import_mpower_csv_can_skip_llm_for_single_import(self) -> None:
        os.environ["ACGME_MAPPING_MODE"] = "llm"
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        init_db(conn)
        row = {
            "Accession Number": "202601080002",
            "Modality": "IR",
            "Exam Code": "IRFLUROCASCACCESS",
            "Exam Description": "IR FLUOROSCOPY GUIDED VASCULAR ACCESS DEVICE PLACEMENT",
            "CPT Code": "",
            "Report Text": (
                "PROCEDURE: Venous port placement\n\n"
                "Procedural Personnel\n"
                "Resident physician(s): Cody Key, MD\n\n"
                "IMPRESSION:\n"
                "Successful venous port placement.\n"
            ),
            "Patient Age": "42",
            "Exam Started Date": "2026-01-08 11:00:00-08:00",
            "Report Finalized By": "Attending, Example",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mpower.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            events: list[dict[str, object]] = []
            summary = import_mpower_csv(
                conn,
                path,
                progress_callback=events.append,
                llm_client=FakeLLMClient(error=LLMClientError("should not call llm")),
                mapping_mode="legacy",
            )
        self.assertEqual(summary["generated_entries_count"], 1)
        self.assertTrue(any(event["phase"] == "Mapping with legacy rules" for event in events))
        source_status = conn.execute("SELECT source_mapping_status, needs_review_reason FROM source_cases").fetchone()
        self.assertNotIn("LLM failed", source_status["needs_review_reason"] or "")

    def test_candidate_generation_uses_parsed_multi_procedure_evidence(self) -> None:
        raw = {
            "Accession Number": "202507270453",
            "Modality": "IR",
            "Exam Code": "IRIVCFIL",
            "Exam Description": "IR IVC FILTER PLACEMENT",
            "CPT Code": "",
            "Report Text": (
                "PROCEDURES:\n"
                "1. Inferior vena cava (IVC) filter insertion\n"
                "2. Renal transarterial embolization\n\n"
                "Procedural Personnel\n"
                "Attending physician(s): Example Attending, MD\n"
                "Resident physician(s): Cody Key, MD\n\n"
                "IMPRESSION:\n"
                "1. Insertion of inferior vena cava filter.\n"
                "2. Right renal angiography with embolization.\n"
            ),
            "Patient Age": "42",
            "Exam Started Date": "2025-07-27 15:53:00-07:00",
            "Report Finalized By": "Attending, Example",
            "__source_row_number": 2,
        }
        source = transform_mpower_row(raw)
        candidates = build_match_candidates(source)
        labels = {(candidate["area"], candidate["type"]) for candidate in candidates}
        self.assertIn(("Venous Interventions", "IVC filter placement"), labels)
        self.assertIn(("Arterial Interventions", "Arterial embolization"), labels)
        self.assertTrue(
            any(
                candidate["type"] == "IVC filter placement"
                and candidate["confidence"] in {"high", "medium"}
                and candidate["default_checked"] == 1
                for candidate in candidates
            )
        )

    def test_candidate_generation_selects_specific_gu_events_without_generic_overmatch(self) -> None:
        raw = {
            "Accession Number": "202605130816",
            "Modality": "IR",
            "Exam Code": "IRTUBECHGL",
            "Exam Description": "IR NEPHROSTOMY TUBE / STENT CHECK/CHANGE",
            "CPT Code": "",
            "Report Text": (
                "PROCEDURE: Genitourinary catheter exchange\n\n"
                "Procedural Personnel\n"
                "Attending physician(s): Example Attending, MD\n"
                "Resident physician(s): Cody Key, MD\n\n"
                "IMPRESSION:\n"
                "1. Serial UPJ ureteroplasty.\n"
                "2. Transplant nephroureteral stent exchange/upsize.\n\n"
                "Plan:\n"
                "Tube(s) capped. Return in 2 weeks.\n\n"
                "PROCEDURE SUMMARY\n"
                "- Target organ: Transplant kidney\n"
                "- Antegrade nephrostogram(s) via the existing access\n"
                "- Nephroureteral tube exchange\n"
                "- Additional procedure(s): None\n\n"
                "PROCEDURE DETAILS:\n"
                "Details omitted.\n"
            ),
            "Patient Age": "42",
            "Exam Started Date": "2026-05-13 10:00:00-07:00",
            "Report Finalized By": "Attending, Example",
            "__source_row_number": 2,
        }
        source = transform_mpower_row(raw)
        candidates = build_match_candidates(source)
        checked = [
            candidate
            for candidate in candidates
            if candidate["default_checked"] == 1
        ]
        checked_labels = {
            (
                candidate["acgme_code"],
                candidate["area"],
                candidate["type"],
                candidate["acgme_description"],
                candidate["acgme_def_category"],
            )
            for candidate in checked
        }
        self.assertIn(
            (
                "31852",
                "GU Intervention",
                "GU tube/stent exchange",
                "Nephroureteral stent change",
                "Catheter exchange",
            ),
            checked_labels,
        )
        self.assertIn(("31840", "GU Intervention", "GU stricture dilation", "GU stricture dilation", "GU Intervention"), checked_labels)
        self.assertNotIn(("", "GU Intervention", "GU tube/stent exchange", "", ""), checked_labels)
        selected_types = {(candidate["area"], candidate["type"]) for candidate in checked}
        self.assertNotIn(("Drainage Procedures", "Drainage tube exchange"), selected_types)
        self.assertNotIn(("Venous Interventions", "Venous thrombolysis catheter change"), selected_types)
        self.assertNotIn(("Arterial Interventions", "Thrombolysis catheter change arterial"), selected_types)
        self.assertNotIn(("Biliary Interventions", "Biliary tube maintenance"), selected_types)

    def test_acgme_search_adds_checked_manual_candidate(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        init_db(conn)
        raw = {
            "Accession Number": "202601020001",
            "Modality": "IR",
            "Exam Code": "IRUNKNOWN",
            "Exam Description": "IR UNKNOWN",
            "CPT Code": "",
            "Report Text": "PROCEDURE: Unknown procedure\n\nResident physician(s): Cody Key, MD",
            "Patient Age": "42",
            "Exam Started Date": "2026-01-02 10:00:00-08:00",
            "Report Finalized By": "Attending, Example",
        }
        source = transform_mpower_row(raw)
        source_id, _ = insert_source_case(conn, source, "sample.csv", 1)
        results = search_acgme_targets("paracentesis")
        self.assertTrue(results)
        candidate_id = next(iter(add_manual_candidates(conn, source_id, [results[0]])))
        row = conn.execute("SELECT * FROM source_match_candidates WHERE id = ?", (candidate_id,)).fetchone()
        self.assertEqual(row["source_kind"], "manual_search")
        self.assertEqual(row["default_checked"], 1)
        self.assertEqual(row["confidence"], "manual")

    def test_acgme_search_finds_description_and_def_category_targets(self) -> None:
        filter_results = search_acgme_targets("filter")
        filter_codes = {row["acgme_code"] for row in filter_results}
        self.assertTrue({"31740", "31742"} <= filter_codes)
        self.assertTrue(any(row["label"].startswith("31742 / IVC filter retrieval") for row in filter_results))
        nephroureteral = search_acgme_targets("nephroureteral stent change")
        self.assertTrue(
            any(
                row["acgme_code"] == "31852"
                and row["area"] == "GU Intervention"
                and row["type"] == "GU tube/stent exchange"
                and row["acgme_description"] == "Nephroureteral stent change"
                and row["acgme_def_category"] == "Catheter exchange"
                for row in nephroureteral
            )
        )
        catheter_exchange = search_acgme_targets("catheter exchange")
        self.assertTrue(any(row["acgme_def_category"] == "Catheter exchange" for row in catheter_exchange))
        stricture = search_acgme_targets("GU stricture dilation")
        self.assertTrue(any(row["acgme_code"] == "31840" and row["area"] == "GU Intervention" and row["type"] == "GU stricture dilation" for row in stricture))

    def test_candidate_generation_selects_ivc_filter_retrieval_code(self) -> None:
        raw = {
            "Accession Number": "202605130877",
            "Modality": "IR",
            "Exam Code": "IRIVCFILRM",
            "Exam Description": "IR IVC FILTER REMOVAL",
            "CPT Code": "",
            "Report Text": (
                "PROCEDURE: Inferior vena cava (IVC) filter retrieval\n\n"
                "Procedural Personnel\n"
                "Attending physician(s): Example Attending, MD\n"
                "Resident physician(s): Cody Key, MD\n\n"
                "IMPRESSION:\n"
                "1. Successful IVC filter retrieval.\n"
            ),
            "Patient Age": "42",
            "Exam Started Date": "2026-05-13 12:50:57-07:00",
            "Report Finalized By": "Attending, Example",
            "__source_row_number": 2,
        }
        source = transform_mpower_row(raw)
        candidates = build_match_candidates(source)
        checked = [candidate for candidate in candidates if candidate["default_checked"] == 1]
        self.assertTrue(
            any(
                candidate["acgme_code"] == "31742"
                and candidate["acgme_description"] == "IVC filter retrieval"
                and candidate["area"] == "Venous Interventions"
                and candidate["type"] == "Venous foreign body retrieval"
                and candidate["acgme_def_category"] == "Venous Intervention"
                for candidate in checked
            )
        )

    def test_candidate_generation_selects_biliary_stent_without_generic_overmatch(self) -> None:
        raw = {
            "Accession Number": "202605060600",
            "Modality": "IR",
            "Exam Code": "IRBILDRCH",
            "Exam Description": "IR BILIARY DRAIN CHECK/CHANGE",
            "CPT Code": "",
            "Report Text": (
                "PROCEDURE: Transhepatic cholangiogram and biliary stent placement\n\n"
                "Procedural Personnel\n"
                "Resident physician(s): Cody Key, MD\n\n"
                "IMPRESSION:\n"
                "1. Balloon sweep of the indwelling metal CBD stent.\n"
                "2. Conversion of the right posterior and left internal-external biliary to internal plastic stents.\n\n"
                "PROCEDURE SUMMARY\n"
                "- Percutaneous transhepatic cholangiogram through existing access\n"
                "- Biliary stent placement as described below\n"
                "- Biliary drain placement: Not performed\n"
                "- Additional procedure(s): Cholangioplasty\n"
            ),
            "Patient Age": "42",
            "Exam Started Date": "2026-05-06 10:00:00-07:00",
            "Report Finalized By": "Attending, Example",
            "__source_row_number": 2,
        }
        source = transform_mpower_row(raw)
        checked = [candidate for candidate in build_match_candidates(source) if candidate["default_checked"] == 1]
        checked_codes = {candidate["acgme_code"] for candidate in checked}
        self.assertEqual(checked_codes, {"31809", "31810"})
        self.assertNotIn("31665", checked_codes)
        self.assertNotIn("31812", checked_codes)

    def test_negative_biliary_drain_placement_does_not_create_checked_drain_candidate(self) -> None:
        raw = {
            "Accession Number": "202605060601",
            "Modality": "IR",
            "Exam Code": "IRBILDRCH",
            "Exam Description": "IR BILIARY DRAIN CHECK/CHANGE",
            "CPT Code": "",
            "Report Text": (
                "PROCEDURE: Biliary stent placement\n\n"
                "Resident physician(s): Cody Key, MD\n\n"
                "IMPRESSION:\n"
                "Successful biliary stent placement.\n\n"
                "PROCEDURE SUMMARY\n"
                "- Biliary drain placement: Not performed\n"
            ),
            "Patient Age": "42",
            "Exam Started Date": "2026-05-06 10:00:00-07:00",
            "Report Finalized By": "Attending, Example",
            "__source_row_number": 2,
        }
        source = transform_mpower_row(raw)
        checked = [candidate for candidate in build_match_candidates(source) if candidate["default_checked"] == 1]
        self.assertFalse(any(candidate["type"] == "Drainage tube placement" for candidate in checked))

    def test_us_guided_biopsy_requires_site_before_selecting_adrenal(self) -> None:
        raw = {
            "Accession Number": "202605060700",
            "Modality": "US",
            "Exam Code": "USSPRFCLBXASPR",
            "Exam Description": "US GUIDED SUPERFICIAL BIOPSY/ASPIRATION",
            "CPT Code": "",
            "Report Text": (
                "US GUIDED SUPERFICIAL BIOPSY/ASPIRATION\n\n"
                "Resident physician(s): Cody Key, MD\n\n"
                "IMPRESSION:\n"
                "Successful ultrasound-guided biopsy of a superficial soft tissue lesion.\n"
            ),
            "Patient Age": "42",
            "Exam Started Date": "2026-05-06 10:00:00-07:00",
            "Report Finalized By": "Attending, Example",
            "__source_row_number": 2,
        }
        source = transform_mpower_row(raw)
        checked = [candidate for candidate in build_match_candidates(source) if candidate["default_checked"] == 1]
        self.assertFalse(any(candidate["acgme_code"] == "31871" for candidate in checked))
        self.assertTrue(any(candidate["acgme_code"] == "31873" for candidate in checked))

    def test_hepatic_radioembolization_selects_radioembolization_not_uterine(self) -> None:
        raw = {
            "Accession Number": "202605060800",
            "Modality": "IR",
            "Exam Code": "IRRADIOEMB",
            "Exam Description": "Hepatic radioembolization - Radioisotope administration",
            "CPT Code": "",
            "Report Text": (
                "PROCEDURE: Hepatic radioembolization - Radioisotope administration\n\n"
                "Resident physician(s): Cody Key, MD\n\n"
                "IMPRESSION:\n"
                "Successful Y90 radioembolization of hepatic tumor.\n"
            ),
            "Patient Age": "42",
            "Exam Started Date": "2026-05-06 10:00:00-07:00",
            "Report Finalized By": "Attending, Example",
            "__source_row_number": 2,
        }
        source = transform_mpower_row(raw)
        checked_codes = {candidate["acgme_code"] for candidate in build_match_candidates(source) if candidate["default_checked"] == 1}
        self.assertIn("31680", checked_codes)
        self.assertNotIn("31681", checked_codes)

    def test_transvaginal_pelvic_drainage_creates_one_checked_drainage_candidate(self) -> None:
        raw = {
            "Accession Number": "202605060900",
            "Modality": "US",
            "Exam Code": "USGUIDEDRAIN",
            "Exam Description": "Transvaginal pelvic drainage catheter placement with ultrasound guidance",
            "CPT Code": "",
            "Report Text": (
                "PROCEDURE: Transvaginal pelvic drainage catheter placement with ultrasound guidance\n\n"
                "Resident physician(s): Cody Key, MD\n\n"
                "IMPRESSION:\n"
                "Successful transvaginal pelvic drainage catheter placement.\n"
            ),
            "Patient Age": "42",
            "Exam Started Date": "2026-05-06 10:00:00-07:00",
            "Report Finalized By": "Attending, Example",
            "__source_row_number": 2,
        }
        source = transform_mpower_row(raw)
        checked = [candidate for candidate in build_match_candidates(source) if candidate["default_checked"] == 1]
        drainage = [candidate for candidate in checked if candidate["type"] == "Drainage tube placement"]
        self.assertEqual(len(drainage), 1)
        self.assertEqual(drainage[0]["acgme_code"], "31902")

    def test_candidate_generation_splits_angioplasty_and_stenting(self) -> None:
        raw = {
            "Accession Number": "202605061000",
            "Modality": "IR",
            "Exam Code": "IRLEANGIO",
            "Exam Description": "Left lower extremity angiography and intervention",
            "CPT Code": "",
            "Report Text": (
                "PROCEDURE: Left lower extremity angiography with angioplasty and stenting\n\n"
                "Resident physician(s): Cody Key, MD\n\n"
                "IMPRESSION:\n"
                "Successful balloon angioplasty and stent placement of the left superficial femoral artery.\n"
            ),
            "Patient Age": "42",
            "Exam Started Date": "2026-05-06 10:00:00-07:00",
            "Report Finalized By": "Attending, Example",
            "__source_row_number": 2,
        }
        source = transform_mpower_row(raw)
        checked = [candidate for candidate in build_match_candidates(source) if candidate["default_checked"] == 1]
        checked_codes = {candidate["acgme_code"] for candidate in checked}
        self.assertIn("31653", checked_codes)
        self.assertIn("31662", checked_codes)

    def test_candidate_generation_splits_venoplasty_and_venous_stent(self) -> None:
        raw = {
            "Accession Number": "202605061001",
            "Modality": "IR",
            "Exam Code": "IRVENOGRAM",
            "Exam Description": "Venography with intervention",
            "CPT Code": "",
            "Report Text": (
                "PROCEDURE: Left iliac venography with venoplasty and stenting\n\n"
                "Resident physician(s): Cody Key, MD\n\n"
                "IMPRESSION:\n"
                "Successful venoplasty and stent placement of the left common iliac vein.\n"
            ),
            "Patient Age": "42",
            "Exam Started Date": "2026-05-06 10:00:00-07:00",
            "Report Finalized By": "Attending, Example",
            "__source_row_number": 2,
        }
        source = transform_mpower_row(raw)
        checked_codes = {candidate["acgme_code"] for candidate in build_match_candidates(source) if candidate["default_checked"] == 1}
        self.assertIn("31707", checked_codes)
        self.assertIn("31713", checked_codes)

    def test_candidate_generation_prefers_covered_stent_graft_target(self) -> None:
        raw = {
            "Accession Number": "202605061002",
            "Modality": "IR",
            "Exam Code": "IRILIAC",
            "Exam Description": "Iliac artery repair",
            "CPT Code": "",
            "Report Text": (
                "PROCEDURE: Right iliac artery repair with covered stent\n\n"
                "Resident physician(s): Cody Key, MD\n\n"
                "IMPRESSION:\n"
                "Successful exclusion of the right external iliac artery injury with covered stent placement.\n"
            ),
            "Patient Age": "42",
            "Exam Started Date": "2026-05-06 10:00:00-07:00",
            "Report Finalized By": "Attending, Example",
            "__source_row_number": 2,
        }
        source = transform_mpower_row(raw)
        checked_codes = {candidate["acgme_code"] for candidate in build_match_candidates(source) if candidate["default_checked"] == 1}
        self.assertIn("31695", checked_codes)
        self.assertNotIn("31662", checked_codes)

    def test_candidate_generation_splits_thrombectomy_and_thrombolysis(self) -> None:
        raw = {
            "Accession Number": "202605061003",
            "Modality": "IR",
            "Exam Code": "IRPETHROMB",
            "Exam Description": "Pulmonary embolism intervention",
            "CPT Code": "",
            "Report Text": (
                "PROCEDURE: Pulmonary artery thrombectomy and catheter-directed thrombolysis\n\n"
                "Resident physician(s): Cody Key, MD\n\n"
                "IMPRESSION:\n"
                "Successful pulmonary artery mechanical thrombectomy and catheter-directed tPA thrombolysis.\n"
            ),
            "Patient Age": "42",
            "Exam Started Date": "2026-05-06 10:00:00-07:00",
            "Report Finalized By": "Attending, Example",
            "__source_row_number": 2,
        }
        source = transform_mpower_row(raw)
        checked_codes = {candidate["acgme_code"] for candidate in build_match_candidates(source) if candidate["default_checked"] == 1}
        self.assertIn("31756", checked_codes)
        self.assertIn("31758", checked_codes)

    def test_candidate_generation_keeps_specific_embolization_targets(self) -> None:
        cases = [
            ("uterine fibroid embolization", "Successful uterine artery embolization.", "31681"),
            ("prostate artery embolization", "Successful prostate artery embolization.", "31682"),
            ("bronchial artery embolization", "Successful bronchial artery embolization for hemoptysis.", "31683"),
        ]
        for idx, (title, impression, expected_code) in enumerate(cases, start=1):
            raw = {
                "Accession Number": f"20260506110{idx}",
                "Modality": "IR",
                "Exam Code": "IREMBOL",
                "Exam Description": title,
                "CPT Code": "",
                "Report Text": (
                    f"PROCEDURE: {title}\n\n"
                    "Resident physician(s): Cody Key, MD\n\n"
                    "IMPRESSION:\n"
                    f"{impression}\n"
                ),
                "Patient Age": "42",
                "Exam Started Date": "2026-05-06 10:00:00-07:00",
                "Report Finalized By": "Attending, Example",
                "__source_row_number": 2,
            }
            source = transform_mpower_row(raw)
            checked_codes = {candidate["acgme_code"] for candidate in build_match_candidates(source) if candidate["default_checked"] == 1}
            self.assertIn(expected_code, checked_codes)
            self.assertNotIn("31684", checked_codes)

    def test_candidate_generation_distinguishes_gu_device_events(self) -> None:
        raw = {
            "Accession Number": "202605061004",
            "Modality": "IR",
            "Exam Code": "IRGU",
            "Exam Description": "GU interventions",
            "CPT Code": "",
            "Report Text": (
                "PROCEDURES:\n"
                "1. Left nephrostomy tube placement\n"
                "2. Right nephroureteral stent exchange\n"
                "3. Double-J ureteral stent exchange\n\n"
                "Resident physician(s): Cody Key, MD\n\n"
                "IMPRESSION:\n"
                "Successful left nephrostomy tube placement, right nephroureteral stent exchange, and double-J stent exchange.\n"
            ),
            "Patient Age": "42",
            "Exam Started Date": "2026-05-06 10:00:00-07:00",
            "Report Finalized By": "Attending, Example",
            "__source_row_number": 2,
        }
        source = transform_mpower_row(raw)
        checked_codes = {candidate["acgme_code"] for candidate in build_match_candidates(source) if candidate["default_checked"] == 1}
        self.assertIn("31836", checked_codes)
        self.assertIn("31852", checked_codes)
        self.assertIn("31853", checked_codes)

    def test_candidate_generation_maps_cholangioplasty_to_biliary_stricture_dilation(self) -> None:
        raw = {
            "Accession Number": "202605061005",
            "Modality": "IR",
            "Exam Code": "IRBILDRCH",
            "Exam Description": "Biliary drain check",
            "CPT Code": "",
            "Report Text": (
                "PROCEDURE: Cholangioplasty\n\n"
                "Resident physician(s): Cody Key, MD\n\n"
                "IMPRESSION:\n"
                "Successful cholangioplasty of a biliary stricture.\n"
            ),
            "Patient Age": "42",
            "Exam Started Date": "2026-05-06 10:00:00-07:00",
            "Report Finalized By": "Attending, Example",
            "__source_row_number": 2,
        }
        source = transform_mpower_row(raw)
        checked_codes = {candidate["acgme_code"] for candidate in build_match_candidates(source) if candidate["default_checked"] == 1}
        self.assertIn("31809", checked_codes)
        self.assertFalse({"31652", "31653", "31656"} & checked_codes)

    def test_candidate_generation_does_not_infer_intervention_from_diagnostic_or_planned_language(self) -> None:
        diagnostic = transform_mpower_row(
            {
                "Accession Number": "202605061006",
                "Modality": "IR",
                "Exam Code": "IRANGIO",
                "Exam Description": "Diagnostic angiography",
                "CPT Code": "",
                "Report Text": (
                    "PROCEDURE: Diagnostic lower extremity angiography\n\n"
                    "Resident physician(s): Cody Key, MD\n\n"
                    "IMPRESSION:\n"
                    "Diagnostic angiography demonstrated stenosis. No intervention was performed.\n"
                ),
                "Patient Age": "42",
                "Exam Started Date": "2026-05-06 10:00:00-07:00",
                "Report Finalized By": "Attending, Example",
                "__source_row_number": 2,
            }
        )
        planned = transform_mpower_row(
            {
                "Accession Number": "202605061007",
                "Modality": "IR",
                "Exam Code": "IRANGIO",
                "Exam Description": "Diagnostic angiography",
                "CPT Code": "",
                "Report Text": (
                    "PROCEDURE: Diagnostic pelvic angiography\n\n"
                    "Resident physician(s): Cody Key, MD\n\n"
                    "IMPRESSION:\n"
                    "Diagnostic angiography performed. Possible embolization and planned stent placement were discussed.\n"
                ),
                "Patient Age": "42",
                "Exam Started Date": "2026-05-06 10:00:00-07:00",
                "Report Finalized By": "Attending, Example",
                "__source_row_number": 2,
            }
        )
        for source in (diagnostic, planned):
            checked = [candidate for candidate in build_match_candidates(source) if candidate["default_checked"] == 1]
            self.assertFalse(any(candidate["type"] in {"Arterial PTA", "Arterial stent", "Arterial embolization"} for candidate in checked))

    def test_approve_checked_candidates_creates_entries_and_rejects_unchecked(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        init_db(conn)
        raw = {
            "Accession Number": "202601030001",
            "Modality": "US",
            "Exam Code": "USGUDPARAC",
            "Exam Description": "US GUIDED PARACENTESIS",
            "CPT Code": "49083",
            "Report Text": (
                "ULTRASOUND-GUIDED PARACENTESIS\n\n"
                "PROCEDURE PERSONNEL:\n"
                "Attending: Example Attending\n"
                "Other: Cody Key\n\n"
                "FINDINGS/IMPRESSION:\n"
                "1. Successful ultrasound-guided paracentesis.\n"
            ),
            "Patient Age": "42",
            "Exam Started Date": "2026-01-03 10:00:00-08:00",
            "Report Finalized By": "Attending, Example",
        }
        source = transform_mpower_row(raw)
        source_id, _ = insert_source_case(conn, source, "sample.csv", 1)
        candidates = build_match_candidates(source)
        store_match_candidates(conn, source_id, candidates)
        add_manual_candidate(conn, source_id, search_acgme_targets("biopsy")[0])
        rows = load_candidates(conn, source_id)
        checked = {int(row["id"]) for row in rows if row["type"] == "Paracentesis"}
        self.assertTrue(checked)
        approve_candidate_review(conn, source_id, checked)
        generated = conn.execute("SELECT acgme_code, type, review_status FROM generated_entries WHERE source_case_id = ?", (source_id,)).fetchall()
        self.assertEqual([row["type"] for row in generated], ["Paracentesis"])
        self.assertEqual(generated[0]["acgme_code"], "31896")
        self.assertEqual(generated[0]["review_status"], "approved")
        statuses = {
            row["user_status"]
            for row in conn.execute("SELECT user_status FROM source_match_candidates WHERE source_case_id = ?", (source_id,))
        }
        self.assertTrue({"accepted", "rejected"} <= statuses)
        signal_count = conn.execute("SELECT COUNT(*) FROM mapping_learning_signals WHERE source_case_id = ?", (source_id,)).fetchone()[0]
        self.assertGreaterEqual(signal_count, 2)

    def test_candidate_queue_lazily_creates_candidates(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        init_db(conn)
        raw = {
            "Accession Number": "202601040001",
            "Modality": "US",
            "Exam Code": "USGUDPARAC",
            "Exam Description": "US GUIDED PARACENTESIS",
            "CPT Code": "49083",
            "Report Text": (
                "ULTRASOUND-GUIDED PARACENTESIS\n\n"
                "PROCEDURE PERSONNEL:\n"
                "Attending: Example Attending\n"
                "Other: Cody Key\n\n"
                "FINDINGS/IMPRESSION:\n"
                "1. Successful ultrasound-guided paracentesis.\n"
            ),
            "Patient Age": "42",
            "Exam Started Date": "2026-01-04 10:00:00-08:00",
            "Report Finalized By": "Attending, Example",
        }
        source = transform_mpower_row(raw)
        source_id, _ = insert_source_case(conn, source, "sample.csv", 1)
        entries, _ = map_source_case(source, *load_mapping_rules())
        insert_generated_entries(conn, source_id, entries)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM source_match_candidates").fetchone()[0], 0)
        queued_source, candidates = load_next_candidate_group(conn)
        self.assertEqual(queued_source["id"], source_id)
        self.assertTrue(candidates)
        self.assertGreater(conn.execute("SELECT COUNT(*) FROM source_match_candidates").fetchone()[0], 0)

    def test_candidate_queue_ignores_stale_algorithm_candidates(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        init_db(conn)
        raw = {
            "Accession Number": "202601040010",
            "Modality": "US",
            "Exam Code": "USGUDPARAC",
            "Exam Description": "US GUIDED PARACENTESIS",
            "CPT Code": "49083",
            "Report Text": (
                "PROCEDURE: Paracentesis\n\n"
                "Resident physician(s): Cody Key\n\n"
                "IMPRESSION:\n"
                "Successful ultrasound-guided paracentesis.\n"
            ),
            "Patient Age": "42",
            "Exam Started Date": "2026-01-04 10:00:00-08:00",
            "Report Finalized By": "Attending, Example",
        }
        source = transform_mpower_row(raw)
        source_id, _ = insert_source_case(conn, source, "sample.csv", 1)
        now = "2026-01-04T18:00:00+00:00"
        conn.execute(
            """
            INSERT INTO source_match_candidates(
              source_case_id, candidate_key, case_class, acgme_code, area, type, acgme_description, acgme_def_category,
              keyword, component_label, score, confidence, default_checked, match_reason, matched_phrases_json,
              evidence_snippet, event_key, event_label, source_kind, algorithm_version, created_at, updated_at
            )
            VALUES (?, ?, 'Interventional Procedures', '31896', 'Drainage Procedures', 'Paracentesis', 'Paracentesis', 'Paracentesis',
              '', 'old_component', 0.9, 'high', 1, 'old stale candidate', '[]', '', '', '', 'event_alias', 'candidate_v1_old', ?, ?)
            """,
            (source_id, f"{source_id}|old", now, now),
        )
        self.assertEqual(review_counts(conn)["needs_review"], 0)
        queued_source, candidates = load_next_candidate_group(conn)
        self.assertEqual(queued_source["id"], source_id)
        self.assertTrue(candidates)
        self.assertTrue(all(candidate["algorithm_version"] == ALGORITHM_VERSION for candidate in candidates))

    def test_clear_deterministic_review_state_removes_pending_candidates_and_non_llm_entries(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        init_db(conn)
        raw = {
            "Accession Number": "202601040011",
            "Modality": "US",
            "Exam Code": "USGUDPARAC",
            "Exam Description": "US GUIDED PARACENTESIS",
            "CPT Code": "49083",
            "Report Text": (
                "PROCEDURE: Paracentesis\n\n"
                "Resident physician(s): Cody Key\n\n"
                "IMPRESSION:\n"
                "Successful ultrasound-guided paracentesis.\n"
            ),
            "Patient Age": "42",
            "Exam Started Date": "2026-01-04 10:00:00-08:00",
            "Report Finalized By": "Attending, Example",
        }
        source = transform_mpower_row(raw)
        source_id, _ = insert_source_case(conn, source, "sample.csv", 1)
        store_match_candidates(conn, source_id, build_match_candidates(source))
        entries, update = map_source_case(source, *load_mapping_rules())
        insert_generated_entries(conn, source_id, entries)
        conn.execute("UPDATE source_cases SET source_mapping_status = ? WHERE id = ?", (update["source_mapping_status"], source_id))
        conn.commit()

        summary = clear_deterministic_review_state(conn)

        self.assertGreater(summary["deleted_candidates"], 0)
        self.assertGreater(summary["skipped_entries"], 0)
        self.assertEqual(summary["reset_sources"], 1)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM source_match_candidates WHERE user_status = 'pending'").fetchone()[0], 0)
        self.assertEqual(conn.execute("SELECT review_status FROM generated_entries").fetchone()["review_status"], "skipped")
        self.assertEqual(conn.execute("SELECT source_mapping_status FROM source_cases WHERE id = ?", (source_id,)).fetchone()[0], "unmapped")

    def test_import_mpower_csv_dedupes_exact_hash_and_persists_parsed_json(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        init_db(conn)
        row = {
            "Accession Number": "202601010001",
            "Modality": "IR",
            "Exam Code": "IRUNKNOWN",
            "Exam Description": "IR EXAMPLE PROCEDURE",
            "CPT Code": "",
            "Report Text": (
                "PROCEDURE: Example procedure\n\n"
                "Procedural Personnel\n"
                "Attending physician(s): Example Attending, MD\n"
                "Resident physician(s): Cody Key, MD\n\n"
                "IMPRESSION:\n"
                "Successful example procedure.\n"
            ),
            "Patient Age": "18",
            "Exam Started Date": "2026-01-01 10:00:00-08:00",
            "Report Finalized By": "Attending, Example",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mpower.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
                writer.writerow(row)
            summary = import_mpower_csv(conn, path)
        self.assertEqual(summary["row_count"], 2)
        self.assertEqual(summary["new_source_cases"], 1)
        self.assertEqual(summary["duplicate_source_cases"], 1)
        source = conn.execute("SELECT * FROM source_cases").fetchone()
        self.assertEqual(source["source_format"], "mpower_csv")
        self.assertEqual(source["source_row_number"], 2)
        self.assertEqual(source["duplicate_accession_flag"], 1)
        self.assertEqual(json.loads(source["parsed_report_json"])["procedure_title"], "Example procedure")

    def test_import_mpower_csv_persists_after_commit_and_reopen(self) -> None:
        row = {
            "Accession Number": "202601010002",
            "Modality": "IR",
            "Exam Code": "IRUNKNOWN",
            "Exam Description": "IR EXAMPLE PROCEDURE",
            "CPT Code": "",
            "Report Text": (
                "PROCEDURE: Example procedure\n\n"
                "Procedural Personnel\n"
                "Attending physician(s): Example Attending, MD\n"
                "Resident physician(s): Cody Key, MD\n\n"
                "IMPRESSION:\n"
                "Successful example procedure.\n"
            ),
            "Patient Age": "42",
            "Exam Started Date": "2026-01-01 11:00:00-08:00",
            "Report Finalized By": "Attending, Example",
        }
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "case_logs.sqlite"
            csv_path = Path(tmp) / "mpower.csv"
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            init_db(conn)
            import_mpower_csv(conn, csv_path)
            conn.commit()
            conn.close()

            reopened = sqlite3.connect(db_path)
            reopened.row_factory = sqlite3.Row
            stored = reopened.execute("SELECT source_format, parsed_report_json FROM source_cases").fetchone()
            reopened.close()
        self.assertEqual(stored["source_format"], "mpower_csv")
        self.assertEqual(json.loads(stored["parsed_report_json"])["procedure_title"], "Example procedure")

    def test_duplicate_import_creates_scope_membership_and_can_review_existing_source(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        init_db(conn)
        row = {
            "Accession Number": "202601010003",
            "Modality": "IR",
            "Exam Code": "IRFLUROCASCACCESS",
            "Exam Description": "IR FLUOROSCOPY GUIDED VASCULAR ACCESS DEVICE PLACEMENT",
            "CPT Code": "",
            "Report Text": (
                "PROCEDURE: Venous port placement\n\n"
                "Procedural Personnel\n"
                "Resident physician(s): Cody Key, MD\n\n"
                "IMPRESSION:\n"
                "Successful venous port placement.\n"
            ),
            "Patient Age": "42",
            "Exam Started Date": "2026-01-01 12:00:00-08:00",
            "Report Finalized By": "Attending, Example",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mpower.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            first = import_mpower_csv(conn, path)
            second = import_mpower_csv(conn, path)
        self.assertEqual(first["new_source_cases"], 1)
        self.assertEqual(second["new_source_cases"], 0)
        self.assertEqual(second["duplicate_source_cases"], 1)
        memberships = conn.execute(
            """
            SELECT import_id, source_case_id, inserted_source_case, generated_entries_count
            FROM import_source_cases
            ORDER BY import_id
            """
        ).fetchall()
        self.assertEqual(len(memberships), 2)
        self.assertEqual(memberships[0]["source_case_id"], memberships[1]["source_case_id"])
        self.assertEqual(memberships[1]["import_id"], second["import_id"])
        self.assertEqual(memberships[1]["inserted_source_case"], 0)
        self.assertEqual(review_counts(conn, second["import_id"])["needs_review"], 1)
        source, _ = load_next_candidate_group(conn, second["import_id"])
        self.assertEqual(source["id"], memberships[1]["source_case_id"])

    def test_scoped_review_counts_exclude_prior_import_backlog(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        init_db(conn)
        mapped_row = {
            "Accession Number": "202601010004",
            "Modality": "IR",
            "Exam Code": "IRFLUROCASCACCESS",
            "Exam Description": "IR FLUOROSCOPY GUIDED VASCULAR ACCESS DEVICE PLACEMENT",
            "CPT Code": "",
            "Report Text": "PROCEDURE: Venous port placement\n\nIMPRESSION:\nSuccessful venous port placement.\n",
            "Patient Age": "42",
            "Exam Started Date": "2026-01-01 13:00:00-08:00",
            "Report Finalized By": "Attending, Example",
        }
        unmapped_row = {
            **mapped_row,
            "Accession Number": "202601010005",
            "Exam Code": "IRUNKNOWN",
            "Exam Description": "IR UNKNOWN PROCEDURE",
            "Report Text": "PROCEDURE: Unknown procedure\n\nIMPRESSION:\nUnknown procedure performed.\n",
            "Exam Started Date": "2026-01-01 14:00:00-08:00",
        }
        with tempfile.TemporaryDirectory() as tmp:
            first_path = Path(tmp) / "first.csv"
            second_path = Path(tmp) / "second.csv"
            for path, row in [(first_path, mapped_row), (second_path, unmapped_row)]:
                with path.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow(row)
            first = import_mpower_csv(conn, first_path)
            second = import_mpower_csv(conn, second_path)
        self.assertGreater(review_counts(conn)["needs_review"], 0)
        self.assertEqual(review_counts(conn, second["import_id"])["needs_review"], 0)
        self.assertEqual(review_counts(conn, second["import_id"])["unmapped_total"], 1)
        source, _ = load_next_candidate_group(conn, second["import_id"])
        self.assertEqual(source["accession_number"], unmapped_row["Accession Number"])

    def test_port_removal_targets_exact_acgme_row(self) -> None:
        raw = {
            "Accession Number": 202606040299,
            "Exam Code": "IRPORTREM",
            "Institution Name": "UC Davis Health",
            "Patient Birth Date": datetime(1970, 1, 1),
            "Principal Result Interpreter": "12471^VU^CATHERINE^TRAM",
            "Report Snippet": "PROCEDURE: IR PORT REMOVAL Date of service: 6/4/2026 Resident physician(s): Cody Key, MD",
            "Study Date": "2026-06-04T10:41:53.0000000-07:00",
            "Study Description": "IR PORT REMOVAL",
        }
        source = transform_report_fixture(raw)
        rules, rules_hash = load_mapping_rules()
        entries, _ = map_source_case(source, rules, rules_hash)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["area"], "Venous Access General")
        self.assertEqual(entries[0]["type"], "Venous access explant")
        self.assertEqual(entries[0]["acgme_description"], "Port removal")
        self.assertEqual(entries[0]["acgme_def_category"], "Removal")
        self.assertIn("Port removal", entries[0]["dedupe_key"])

    def test_tunneled_catheter_removal_targets_exact_acgme_row(self) -> None:
        raw = {
            "Accession Number": 202606040300,
            "Exam Code": "IRCVCTUNREM",
            "Institution Name": "UC Davis Health",
            "Patient Birth Date": datetime(1970, 1, 1),
            "Principal Result Interpreter": "12471^VU^CATHERINE^TRAM",
            "Report Snippet": "PROCEDURE: Tunneled central venous catheter removal Date of service: 6/4/2026 Resident physician(s): Cody Key, MD",
            "Study Date": "2026-06-04T10:41:53.0000000-07:00",
            "Study Description": "IR TUNNELED CATHETER REMOVAL",
        }
        source = transform_report_fixture(raw)
        rules, rules_hash = load_mapping_rules()
        entries, _ = map_source_case(source, rules, rules_hash)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["type"], "Venous access explant")
        self.assertEqual(entries[0]["acgme_description"], "Tunneled catheter removal")
        self.assertEqual(entries[0]["acgme_def_category"], "Removal")

    def test_ivc_filter_retrieval_targets_venous_foreign_body_retrieval(self) -> None:
        raw = {
            "Accession Number": 202605130877,
            "Exam Code": "IRIVCFILRM",
            "Institution Name": "UC Davis Health",
            "Patient Birth Date": datetime(1970, 1, 1),
            "Principal Result Interpreter": "12471^VU^CATHERINE^TRAM",
            "Report Snippet": "PROCEDURE: Inferior vena cava (IVC) filter retrieval Date of service: 5/13/2026 Resident physician(s): Cody Key, MD",
            "Study Date": "2026-05-13T12:50:57-07:00",
            "Study Description": "IR IVC FILTER REMOVAL",
        }
        source = transform_report_fixture(raw)
        rules, rules_hash = load_mapping_rules()
        entries, update = map_source_case(source, rules, rules_hash)
        self.assertEqual(update["source_mapping_status"], "generated")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["area"], "Venous Interventions")
        self.assertEqual(entries[0]["type"], "Venous foreign body retrieval")
        self.assertEqual(entries[0]["acgme_def_category"], "Venous Intervention")

    def test_ivc_filter_insertion_targets_ivc_filter_placement(self) -> None:
        raw = {
            "Accession Number": 202603031356,
            "Exam Code": "IRIVCFIL",
            "Institution Name": "UC Davis Health",
            "Patient Birth Date": datetime(1970, 1, 1),
            "Principal Result Interpreter": "12471^VU^CATHERINE^TRAM",
            "Report Snippet": "PROCEDURE: Inferior vena cava (IVC) filter insertion Date of service: 3/3/2026 Resident physician(s): Cody Key, MD",
            "Study Date": "2026-03-03T11:35:44-08:00",
            "Study Description": "IR IVC FILTER PLACEMENT",
        }
        source = transform_report_fixture(raw)
        rules, rules_hash = load_mapping_rules()
        entries, update = map_source_case(source, rules, rules_hash)
        self.assertEqual(update["source_mapping_status"], "generated")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["area"], "Venous Interventions")
        self.assertEqual(entries[0]["type"], "IVC filter placement")
        self.assertEqual(entries[0]["acgme_def_category"], "Venous Intervention")

    def test_venous_plasty_targets_venous_pta(self) -> None:
        raw = {
            "Accession Number": 202606070001,
            "Exam Code": "IRVENOGRAM",
            "Report Snippet": (
                "PROCEDURE: Left iliac venoplasty\n\n"
                "Procedural Personnel\n"
                "Attending physician(s): Example Attending, MD\n"
                "Resident physician(s): Cody Key, MD\n\n"
                "IMPRESSION:\n"
                "Successful left iliac venoplasty.\n\n"
                "PROCEDURE SUMMARY:\n"
                "- Left iliac venoplasty\n"
                "- Additional procedure(s): None\n\n"
                "PROCEDURE DETAILS:\n"
                "Details omitted.\n"
            ),
            "Study Date": "2026-06-07T10:41:53-07:00",
            "Study Description": "IR VENOGRAPHY AND INTERVENTION",
        }
        source = transform_report_fixture(raw)
        rules, rules_hash = load_mapping_rules()
        entries, update = map_source_case(source, rules, rules_hash)
        self.assertEqual(update["source_mapping_status"], "generated")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["area"], "Venous Interventions")
        self.assertEqual(entries[0]["type"], "Venous PTA")
        self.assertEqual(entries[0]["acgme_def_category"], "Venous Intervention")

    def test_parenthetical_ivc_alias_normalization(self) -> None:
        tokens = match_tokens("Inferior vena cava (IVC) filter retrieval")
        self.assertIn("ivc", tokens)
        self.assertTrue({"inferior", "vena", "cava"} <= tokens)
        self.assertTrue({"retrieval", "removal"} <= tokens)

    def test_fuzzy_fallback_suggests_without_auto_generating_low_confidence(self) -> None:
        raw = {
            "Accession Number": 202606010001,
            "Exam Code": "IRUNKNOWN",
            "Institution Name": "UC Davis Health",
            "Patient Birth Date": datetime(1970, 1, 1),
            "Principal Result Interpreter": "12471^VU^CATHERINE^TRAM",
            "Report Snippet": "PROCEDURE: Venous port insertion Date of service: 6/1/2026 Resident physician(s): Cody Key, MD",
            "Study Date": "2026-06-01T10:41:53-07:00",
            "Study Description": "IR VENOUS PORT INSERTION",
        }
        source = transform_report_fixture(raw)
        rules, rules_hash = load_mapping_rules()
        entries, update = map_source_case(source, rules, rules_hash)
        self.assertEqual(entries, [])
        self.assertEqual(update["source_mapping_status"], "unmapped")
        self.assertTrue(update.get("mapping_suggestions"))
        self.assertEqual(update["mapping_suggestions"][0]["type"], "Venous port placement")

    def test_transjugular_liver_biopsy_maps_transvenous(self) -> None:
        raw = {
            "Accession Number": 202605131138,
            "Exam Code": "IRTRANJULV",
            "Institution Name": "UC Davis Health",
            "Patient Birth Date": datetime(1975, 5, 6),
            "Principal Result Interpreter": "12471^VU^CATHERINE^TRAM",
            "Report Snippet": "PROCEDURE: Transjugular liver biopsy with pressure measurements Date of service: 5/13/2026 Resident physician(s): Cody Key, MD",
            "Study Date": "2026-05-13T10:41:53.0000000-07:00",
            "Study Description": "IR TRANSJUGULAR LIVER BIOPSY",
        }
        source = transform_report_fixture(raw)
        rules, rules_hash = load_mapping_rules()
        entries, _ = map_source_case(source, rules, rules_hash)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["area"], "Biopsy")
        self.assertEqual(entries[0]["type"], "Biopsy transvenous")

    def test_remap_unresolved_cases_generates_from_stored_source(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        init_db(conn)
        raw = {
            "Accession Number": 202605130877,
            "Exam Code": "IRIVCFILRM",
            "Institution Name": "UC Davis Health",
            "Patient Birth Date": datetime(1970, 1, 1),
            "Principal Result Interpreter": "12471^VU^CATHERINE^TRAM",
            "Report Snippet": "PROCEDURE: Inferior vena cava (IVC) filter retrieval Date of service: 5/13/2026 Resident physician(s): Cody Key, MD",
            "Study Date": "2026-05-13T12:50:57-07:00",
            "Study Description": "IR IVC FILTER REMOVAL",
        }
        source = transform_report_fixture(raw)
        source_id, _ = insert_source_case(conn, source, "sample.csv", 1)
        summary = remap_unresolved_cases(conn)
        self.assertEqual(summary["generated_entries"], 1)
        row = conn.execute("SELECT type FROM generated_entries WHERE source_case_id = ?", (source_id,)).fetchone()
        self.assertEqual(row["type"], "Venous foreign body retrieval")

    def test_queue_locking_and_statuses(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        init_db(conn)
        raw = {
            "Accession Number": 1,
            "Exam Code": "IRFLUROCASCACCESS",
            "Institution Name": "UC Davis Health",
            "Patient Birth Date": datetime(1981, 12, 8),
            "Principal Result Interpreter": "25727^MORSHEDI^MAUD^MOSTAFA",
            "Report Snippet": "PROCEDURE: Venous port placement Date of service: 5/15/2026 Resident physician(s): Cody Key, MD",
            "Study Date": "2026-05-15T09:57:03.0000000-07:00",
            "Study Description": "IR FLUOROSCOPY GUIDED VASCULAR ACCESS DEVICE PLACEMENT",
        }
        source = transform_report_fixture(raw)
        source_id, _ = insert_source_case(conn, source, "sample.csv", 1)
        rules, rules_hash = load_mapping_rules()
        entries, _ = map_source_case(source, rules, rules_hash)
        entries[0]["review_status"] = "approved"
        insert_generated_entries(conn, source_id, entries)
        first = claim_next(conn)
        second = claim_next(conn)
        self.assertEqual(first["local_entry_id"], second["local_entry_id"])
        self.assertIn("acgme_description", first)
        self.assertIn("acgme_def_category", first)
        self.assertEqual(get_current(conn)["local_entry_id"], first["local_entry_id"])
        update_upload_status(conn, first["local_entry_id"], "autofilled", "autofilled")
        row = update_upload_status(conn, first["local_entry_id"], "submitted", "submitted")
        self.assertEqual(row["upload_status"], "submitted")

    def test_group_queue_claim_submit_skip_and_edit(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        init_db(conn)
        raw = {
            "Accession Number": 3,
            "Exam Code": "IRFLUROCASCACCESS",
            "Institution Name": "UC Davis Health",
            "Patient Birth Date": datetime(1981, 12, 8),
            "Principal Result Interpreter": "25727^MORSHEDI^MAUD^MOSTAFA",
            "Report Snippet": "PROCEDURE: Venous port placement Date of service: 5/15/2026 Resident physician(s): Cody Key, MD",
            "Study Date": "2026-05-15T09:57:03.0000000-07:00",
            "Study Description": "IR FLUOROSCOPY GUIDED VASCULAR ACCESS DEVICE PLACEMENT",
        }
        source = transform_report_fixture(raw)
        source_id, _ = insert_source_case(conn, source, "sample.csv", 1)
        rules, rules_hash = load_mapping_rules()
        entries, _ = map_source_case(source, rules, rules_hash)
        entries[0]["review_status"] = "approved"
        second = {**entries[0]}
        second.update(
            {
                "dedupe_key": second["dedupe_key"] + "|second",
                "area": "Venous Access General",
                "type": "Venous access explant",
                "acgme_description": "Port removal",
                "acgme_def_category": "Removal",
                "component_label": "secondary_procedure",
            }
        )
        insert_generated_entries(conn, source_id, [entries[0], second])

        group = claim_next_group(conn)
        self.assertEqual(group["source_case_id"], source_id)
        self.assertEqual(len(group["codes"]), 2)
        self.assertEqual(
            {row["upload_status"] for row in conn.execute("SELECT upload_status FROM generated_entries")},
            {"claimed"},
        )

        edited = save_group_edit(
            conn,
            source_id,
            [
                group["codes"][0],
                {
                    "case_class": "Interventional Procedures",
                    "area": "Biopsy",
                    "type": "Biopsy abdominal/retroperitoneal",
                    "acgme_description": "Biopsy - liver",
                    "acgme_def_category": "Biopsy; Image guided bx/drainage",
                    "component_label": "manual_added",
                },
            ],
        )
        self.assertEqual(len(edited["codes"]), 2)
        self.assertIn("Biopsy abdominal/retroperitoneal", {code["type"] for code in edited["codes"]})
        deselected = conn.execute(
            "SELECT upload_status FROM generated_entries WHERE component_label = 'secondary_procedure'"
        ).fetchone()
        self.assertEqual(deselected["upload_status"], "skipped_upload_session")

        update_group_upload_status(conn, source_id, "autofilled", "autofilled")
        update_group_upload_status(conn, source_id, "submitted", "submitted")
        submitted = conn.execute(
            "SELECT count(*) FROM generated_entries WHERE source_case_id = ? AND upload_status = 'submitted'",
            (source_id,),
        ).fetchone()[0]
        self.assertEqual(submitted, 2)

        raw2 = {**raw, "Accession Number": 4}
        source2 = transform_report_fixture(raw2)
        source2_id, _ = insert_source_case(conn, source2, "sample.csv", 1)
        next_entry = {**entries[0], "dedupe_key": entries[0]["dedupe_key"] + "|next", "review_status": "approved"}
        insert_generated_entries(conn, source2_id, [next_entry])
        next_group = claim_next_group(conn)
        self.assertEqual(next_group["source_case_id"], source2_id)
        update_group_upload_status(conn, source2_id, "skipped_upload_session", "skip_upload")
        skipped = conn.execute(
            "SELECT upload_status FROM generated_entries WHERE source_case_id = ?", (source2_id,)
        ).fetchone()
        self.assertEqual(skipped["upload_status"], "skipped_upload_session")

    def test_learned_rule_persists_acgme_row_target(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        init_db(conn)
        raw = {
            "Accession Number": 2,
            "Exam Code": "IRPORTREM",
            "Institution Name": "UC Davis Health",
            "Patient Birth Date": datetime(1981, 12, 8),
            "Principal Result Interpreter": "25727^MORSHEDI^MAUD^MOSTAFA",
            "Report Snippet": "PROCEDURE: IR PORT REMOVAL Date of service: 5/15/2026 Resident physician(s): Cody Key, MD",
            "Study Date": "2026-05-15T09:57:03.0000000-07:00",
            "Study Description": "IR PORT REMOVAL",
        }
        source = transform_report_fixture(raw)
        source_id, _ = insert_source_case(conn, source, "sample.csv", 1)
        db_source = conn.execute("SELECT * FROM source_cases WHERE id = ?", (source_id,)).fetchone()
        values = {
            "case_class": "Interventional Procedures",
            "area": "Venous Access General",
            "type": "Venous access explant",
            "acgme_description": "Port removal",
            "acgme_def_category": "Removal",
            "component_label": "dominant_procedure",
            "keyword": "",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mapping_rules.csv"
            path.write_text(Path("config/mapping_rules.csv").read_text())
            learned = append_learned_rule(db_source, values, path)
            self.assertEqual(learned["acgme_description"], "Port removal")
            self.assertEqual(learned["acgme_def_category"], "Removal")
            loaded, _ = load_mapping_rules(path)
            self.assertTrue(any(rule.acgme_description == "Port removal" for rule in loaded if rule.rule_id == learned["rule_id"]))


if __name__ == "__main__":
    unittest.main()
