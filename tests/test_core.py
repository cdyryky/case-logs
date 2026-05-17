from __future__ import annotations

import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from datetime import datetime

from app.learning import append_learned_rule
from app.importer import import_mpower_csv, insert_generated_entries, insert_source_case
from app.matching import match_tokens
from app.mapper import load_mapping_rules, map_source_case
from app.models import init_db
from app.parser import parse_mpower_report, parse_mpower_role_metadata, transform_mpower_row, transform_source_row
from app.review_queue import remap_unresolved_cases
from app.upload_queue import (
    claim_next,
    claim_next_group,
    get_current,
    save_group_edit,
    update_group_upload_status,
    update_upload_status,
)
from app.utils import case_year_from_date, format_acgme_date, patient_type


class CoreTests(unittest.TestCase):
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
            "Report Snippet": "PROCEDURE: Venous port placement Date of service: 5/15/2026 Resident physician(s): Cody Key, MD Advanced practice provider(s): None",
            "Study Date": "2026-05-15T09:57:03.0000000-07:00",
            "Study Description": "IR FLUOROSCOPY GUIDED VASCULAR ACCESS DEVICE PLACEMENT",
        }
        source = transform_source_row(raw)
        self.assertEqual(source["derived"]["role"], "Primary")
        self.assertEqual(source["derived"]["role_confidence"], "high")
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
        source = transform_source_row(raw)
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
        source = transform_source_row(raw)
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
        source = transform_source_row(raw)
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
        source = transform_source_row(raw)
        rules, rules_hash = load_mapping_rules()
        entries, update = map_source_case(source, rules, rules_hash)
        self.assertEqual(update["source_mapping_status"], "generated")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["area"], "Venous Interventions")
        self.assertEqual(entries[0]["type"], "IVC filter placement")
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
        source = transform_source_row(raw)
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
        source = transform_source_row(raw)
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
        source = transform_source_row(raw)
        source_id, _ = insert_source_case(conn, source, "sample.xlsx", 1)
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
        source = transform_source_row(raw)
        source_id, _ = insert_source_case(conn, source, "sample.xlsx", 1)
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
        source = transform_source_row(raw)
        source_id, _ = insert_source_case(conn, source, "sample.xlsx", 1)
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
        source2 = transform_source_row(raw2)
        source2_id, _ = insert_source_case(conn, source2, "sample.xlsx", 1)
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
        source = transform_source_row(raw)
        source_id, _ = insert_source_case(conn, source, "sample.xlsx", 1)
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
