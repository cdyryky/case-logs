from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from datetime import datetime

from app.learning import append_learned_rule
from app.importer import insert_generated_entries, insert_source_case
from app.mapper import load_mapping_rules, map_source_case
from app.models import init_db
from app.parser import transform_source_row
from app.upload_queue import claim_next, get_current, update_upload_status
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
