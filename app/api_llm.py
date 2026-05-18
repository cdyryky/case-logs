from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Any

from .candidates import CandidateTarget, load_acgme_targets
from .constants import DEFAULT_CASE_CLASS
from .models import log_event, utc_now
from .review_queue import parse_report_context_json, source_row_to_mapping_source
from .utils import canonical_key

API_LLM_PROMPT_VERSION = "api_llm_v1"
API_LLM_RULE_ID = "api-llm"
API_ACCEPTED_STATUSES = ("accepted_auto", "accepted_manual", "edited_manual")
API_PENDING_STATUSES = ("pending_review", "error")
API_ERROR_TARGET = "ERROR"

API_LLM_TERMINOLOGY = """<terminology>
Use these terms as aids for synonyms only. Match clinical meaning, not exact wording.
General matching:
Ignore capitalization, hyphenation, plural/singular, and minor spelling variants.
Treat "placement", "insertion", "placed", "inserted", "creation", and "new access" as procedural placement when clinically appropriate.
Treat "exchange", "replacement", "changed", "upsized", "downsized", "converted", and "revision" as exchange/change when an existing tube/stent/catheter is present.
Treat "removal", "removed", "discontinued", "pulled", "explant", and "retrieval" as removal/retrieval.
Choose the most specific target: e.g. uterine artery embolization > other arterial embolization; TIPS revision stent > generic venous stent.
Do not infer a procedure from diagnostic findings alone unless the Impression states it was performed.
Core action synonyms:
angioplasty/PTA: angioplasty, balloon angioplasty, balloon dilation, PTA, venoplasty.
stent: stent placement, stenting, endovascular stent, relining, stent revision.
stent_graft: stent graft, covered stent, endograft, endoprosthesis, TEVAR, EVAR, flow diversion.
thrombolysis: thrombolysis, lysis, catheter-directed thrombolysis, CDT, tPA, alteplase, EKOS.
thrombectomy: thrombectomy, mechanical thrombectomy, aspiration thrombectomy, embolectomy, FlowTriever, ClotTriever, Penumbra, AngioJet.
embolization: embolization, coil embolization, plug embolization, particle embolization, gelfoam, glue, NBCA, Onyx, devascularization, occlusion.
tumor_embolization: TACE, DEB-TACE, bland embolization, TAE, radioembolization, TARE, Y90, SIRT.
biopsy: biopsy, bx, core biopsy, needle biopsy, FNA, tissue sampling, marrow biopsy.
aspiration_drainage: aspiration, abscess aspiration, drain placement, drainage catheter, pigtail catheter, chest tube, PleurX, paracentesis, thoracentesis.
tube_check_exchange: tube check, catheter check, drain check, sinogram, abscessogram, cholangiogram via tube, nephrostogram, exchange, replacement, reposition, revision, conversion, internalization, upsizing, downsizing.
ablation: ablation, microwave ablation, MWA, radiofrequency ablation, RFA, cryoablation, ethanol ablation, sclerotherapy.
nerve_block: block, injection, steroid injection, anesthetic injection, celiac plexus block/neurolysis, epidural steroid injection, blood patch, facet injection.
High-yield aliases:
aorta: TEVAR, thoracic endograft, EVAR, AAA repair, infrarenal endograft, AUI, iliac limbs.
lower_extremity_artery: iliac, femoral, popliteal, tibial, peroneal, pedal arteries.
visceral_artery: celiac, hepatic, splenic, left gastric, GDA, SMA, IMA, renal, adrenal, mesenteric, bronchial, intercostal, lumbar.
venous: venogram, venoplasty, central venous stenosis, SVC, IVC, brachiocephalic, subclavian, iliac, femoral, renal, gonadal, hepatic, portal veins.
dialysis_access: fistulagram, graftogram, AVF, AVG, dialysis access angioplasty, declot, thrombectomy/thrombolysis of AV graft/fistula.
venous_access: CVC, tunneled line, tunneled dialysis catheter, TDC, port, port-a-cath, mediport, PICC, midline, Hohn, Hickman, Broviac, Trialysis.
biliary: PTC, PTBD, external biliary drain, internal-external biliary drain, biliary tube exchange, biliary stent, bilioplasty, cholecystostomy.
enteric_tubes: gastrostomy, G tube, PEG, gastrojejunostomy, GJ tube, jejunostomy, J tube, G to GJ conversion.
genitourinary: nephrostomy, PCN, nephroureteral stent, NUS, antegrade ureteral stent, double-J, JJ stent, ureteroplasty, pyelogram, cystostomy.
lymphatic: lymphangiogram, intranodal lymphangiography, thoracic duct embolization, TDE, lymphatic leak embolization.
vascular_malformation: AVM embolization, venous malformation sclerotherapy/embolization, lymphatic malformation sclerotherapy/embolization.
spine_msk: vertebroplasty, kyphoplasty, sacroplasty, arthrogram, joint injection, bone biopsy, spine biopsy, myelogram, lumbar puncture.
Important distinctions:
Diagnostic angiography/venography is not PTA, stent, embolization, thrombolysis, or thrombectomy unless those actions are explicitly stated.
Aspiration alone is not drainage tube placement unless a catheter/tube/drain is placed.
Tube check is distinct from tube exchange/change/removal.
Covered stent/stent graft should map to stent graft targets when available, not generic stent.
Chemoembolization, bland embolization, and radioembolization should stay separate.
Arterial, venous, portal, pulmonary, neuro, and dialysis-access thrombectomy/thrombolysis should stay separate.
Double-J ureteral stent, nephroureteral stent, and nephrostomy tube are distinct devices.
</terminology>"""


class ApiLLMError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedTarget:
    target: CandidateTarget | None
    raw_target: str
    is_error: bool = False
    error: str = ""


def stable_llm_case_id(source: dict[str, Any]) -> str:
    seed = "|".join(
        str(source.get(key) or "")
        for key in ["source_format", "source_row_hash", "study_date", "exam_code"]
    )
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"llm_{digest}"


def target_label(target: CandidateTarget) -> str:
    return f"{target.acgme_code} | {target.acgme_description or target.type}"


def target_lookup_by_code() -> dict[str, CandidateTarget]:
    return {target.acgme_code: target for target in load_acgme_targets() if target.acgme_code}


def allowed_target_labels() -> list[str]:
    return [target_label(target) for target in sorted(load_acgme_targets(), key=lambda t: t.acgme_code)]


def require_api_pathway(conn: sqlite3.Connection, session_pathway: str | None) -> None:
    if (session_pathway or "").strip().lower() != "api":
        raise ApiLLMError("API LLM import rejected: Streamlit session is not locked to the API pathway.")
    latest = conn.execute("SELECT mapping_pathway FROM imports ORDER BY id DESC LIMIT 1").fetchone()
    if latest and latest["mapping_pathway"] != "api":
        raise ApiLLMError(
            f"API LLM import rejected: SQLite session is locked to '{latest['mapping_pathway']}', not 'api'."
        )


def parse_target_string(value: object) -> ParsedTarget:
    raw = str(value or "").strip()
    if raw == API_ERROR_TARGET:
        return ParsedTarget(None, raw, is_error=True)
    if "|" not in raw:
        return ParsedTarget(None, raw, error="target must be code-prefixed as '<code> | <description>'")
    code, description = [part.strip() for part in raw.split("|", 1)]
    if not re.fullmatch(r"\d+", code):
        return ParsedTarget(None, raw, error=f"target code is not numeric: {code or '<missing>'}")
    target = target_lookup_by_code().get(code)
    if not target:
        return ParsedTarget(None, raw, error=f"unknown ACGME target code: {code}")
    expected = target.acgme_description or target.type
    if description != expected:
        return ParsedTarget(None, raw, error=f"target description mismatch for {code}: expected '{expected}'")
    return ParsedTarget(target, raw)


def extracted_case_text(source: sqlite3.Row) -> dict[str, str | None]:
    parsed = parse_report_context_json(source["parsed_report_json"] if "parsed_report_json" in source.keys() else None)
    impression = re.sub(r"\s+", " ", str(parsed.get("impression") or "")).strip()
    summary_values: list[str] = []
    for section in parsed.get("procedure_summary_sections") or []:
        if not isinstance(section, dict):
            continue
        summary_values.extend(str(item).strip() for item in section.get("bullets") or [] if str(item).strip())
        summary_values.extend(str(item).strip() for item in section.get("additional_procedures") or [] if str(item).strip())
    procedure_summary = " | ".join(list(dict.fromkeys(summary_values))) or None
    return {"impression": impression, "procedure_summary": procedure_summary}


def export_case_payload(conn: sqlite3.Connection, import_id: int | None = None) -> list[dict[str, str | None]]:
    scope = """
      AND EXISTS (
        SELECT 1 FROM import_source_cases isc
        WHERE isc.source_case_id = sc.id AND isc.import_id = ?
      )
    """ if import_id is not None else ""
    rows = conn.execute(
        f"""
        SELECT sc.*
        FROM source_cases sc
        WHERE sc.mapping_pathway = 'api'
          {scope}
        ORDER BY sc.id
        """,
        (import_id,) if import_id is not None else (),
    ).fetchall()
    payload = []
    for row in rows:
        text = extracted_case_text(row)
        item: dict[str, str | None] = {
            "case_id": row["llm_case_id"],
            "impression": text["impression"] or "",
        }
        if text["procedure_summary"]:
            item["procedure_summary"] = text["procedure_summary"]
        else:
            item["procedure_summary"] = None
        payload.append(item)
    return payload


def build_api_llm_prompt(case_payload: list[dict[str, Any]]) -> str:
    schema = [
        {
            "case_id": "string",
            "mappings": [
                {
                    "target": "exact code-prefixed Procedure Target string or ERROR",
                    "phrase": "verbatim supporting phrase from Impression",
                    "confidence": 0,
                }
            ],
        }
    ]
    rules = [
        "Determine the number of distinct procedures performed in each case.",
        "Output a single valid JSON array.",
        "Each output object must include the original case_id.",
        "The target value must be copied exactly from the Procedure Targets list.",
        "Do not reword, abbreviate, normalize, or invent targets.",
        "The phrase value should be the verbatim excerpt from the Impression that best supports the mapping.",
        "The confidence value must be an integer from 0 to 100.",
        "Choose the most specific clinically appropriate target.",
        "Match by clinical meaning, not surface word overlap.",
        'If no confident target can be assigned, use "ERROR" as the target.',
        "Output JSON only. No explanation, markdown, or commentary.",
    ]
    return "\n\n".join(
        [
            "You are an interventional radiology procedure logger. Map each procedure described in the provided report text to an allowed Procedure Target.",
            "Mapping rules:\n" + "\n".join(f"{idx}. {rule}" for idx, rule in enumerate(rules, start=1)),
            "Output schema:\n" + json.dumps(schema, indent=2),
            API_LLM_TERMINOLOGY,
            "Procedure Targets:\n" + json.dumps(allowed_target_labels(), indent=2),
            "Case payload:\n" + json.dumps(case_payload, indent=2),
        ]
    )


def _source_by_llm_case_id(conn: sqlite3.Connection) -> dict[str, sqlite3.Row]:
    return {
        row["llm_case_id"]: row
        for row in conn.execute(
            "SELECT * FROM source_cases WHERE mapping_pathway = 'api' AND llm_case_id IS NOT NULL"
        ).fetchall()
    }


def validate_llm_output(conn: sqlite3.Connection, text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return [], [{"case_id": "", "target": "", "error": f"invalid JSON: {exc.msg}"}], None
    if not isinstance(payload, list):
        return [], [{"case_id": "", "target": "", "error": "top-level JSON value must be an array"}], payload

    sources = _source_by_llm_case_id(conn)
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for case_index, case_item in enumerate(payload):
        if not isinstance(case_item, dict):
            invalid.append({"case_id": "", "target": "", "error": f"case item {case_index + 1} must be an object"})
            continue
        case_id = str(case_item.get("case_id") or "").strip()
        if not case_id:
            invalid.append({"case_id": "", "target": "", "error": "missing case_id"})
            continue
        source = sources.get(case_id)
        if not source:
            invalid.append({"case_id": case_id, "target": "", "error": "case_id does not exist in current API import"})
            continue
        mappings = case_item.get("mappings")
        if not isinstance(mappings, list):
            invalid.append({"case_id": case_id, "target": "", "error": "mappings must be an array"})
            continue
        for mapping_index, mapping in enumerate(mappings):
            if not isinstance(mapping, dict):
                invalid.append({"case_id": case_id, "target": "", "error": f"mapping {mapping_index + 1} must be an object"})
                continue
            target_value = mapping.get("target")
            phrase = str(mapping.get("phrase") or "").strip()
            confidence = mapping.get("confidence")
            parsed_target = parse_target_string(target_value)
            errors = []
            if "target" not in mapping:
                errors.append("missing target")
            elif parsed_target.error:
                errors.append(parsed_target.error)
            if not phrase:
                errors.append("missing phrase")
            if not isinstance(confidence, int) or isinstance(confidence, bool) or confidence < 0 or confidence > 100:
                errors.append("confidence must be an integer from 0 to 100")
            if errors:
                invalid.append({"case_id": case_id, "target": str(target_value or ""), "phrase": phrase, "error": "; ".join(errors)})
                continue
            valid.append(
                {
                    "source_case_id": int(source["id"]),
                    "case_id": case_id,
                    "raw_target": parsed_target.raw_target,
                    "target": parsed_target.target,
                    "target_is_error": parsed_target.is_error,
                    "phrase": phrase,
                    "confidence": int(confidence),
                    "raw_case": case_item,
                    "raw_mapping": mapping,
                }
            )
    return valid, invalid, payload


def _entry_values_for_mapping(source: sqlite3.Row, mapping: dict[str, Any], payload: Any) -> dict[str, Any]:
    derived_source = source_row_to_mapping_source(source)
    derived = derived_source["derived"]
    target = mapping["target"]
    is_error = bool(mapping["target_is_error"])
    index = mapping.get("index", 1)
    confidence = int(mapping["confidence"])
    component_label = f"{mapping['case_id']}_{index:03d}"
    if target:
        case_class = target.case_class or DEFAULT_CASE_CLASS
        acgme_code = target.acgme_code
        area = target.area
        typ = target.type
        description = target.acgme_description
        def_category = target.acgme_def_category
        keyword = target.keyword
    else:
        case_class = DEFAULT_CASE_CLASS
        acgme_code = ""
        area = API_ERROR_TARGET
        typ = API_ERROR_TARGET
        description = ""
        def_category = ""
        keyword = ""
    dedupe_key = "|".join(
        [
            "api-llm",
            mapping["case_id"],
            str(index),
            acgme_code or API_ERROR_TARGET,
            canonical_key(mapping["phrase"])[:80],
        ]
    )
    status = "error" if is_error else "pending_review"
    return {
        "source_case_id": source["id"],
        "dedupe_key": dedupe_key,
        "component_label": component_label,
        "case_id": derived["case_id"],
        "case_date": derived["case_date"],
        "case_year": derived["case_year"],
        "role": derived["role"],
        "site": derived["site"],
        "patient_type": derived["patient_type"],
        "case_class": case_class,
        "acgme_code": acgme_code,
        "area": area,
        "type": typ,
        "acgme_description": description,
        "acgme_def_category": def_category,
        "keyword": keyword,
        "comments": f"API LLM confidence {confidence}; target={mapping['raw_target']}",
        "mapping_rule_id": API_LLM_RULE_ID,
        "mapping_rule_version": "1",
        "mapping_rules_file_hash": API_LLM_PROMPT_VERSION,
        "mapping_rule_name": "Out-of-band API LLM mapping",
        "mapping_confidence": str(confidence),
        "role_confidence": derived["role_confidence"],
        "compound_flag": 1,
        "review_status": status,
        "evidence_excerpt": mapping["phrase"],
        "llm_model": "external-api",
        "llm_prompt_version": API_LLM_PROMPT_VERSION,
        "llm_raw_response_json": json.dumps({"payload": payload, "mapping": mapping["raw_mapping"]}, ensure_ascii=False, sort_keys=True),
        "mapping_pathway": "api",
    }


def import_validated_mappings(
    conn: sqlite3.Connection,
    valid_mappings: list[dict[str, Any]],
    raw_payload: Any,
) -> int:
    now = utc_now()
    sources = {row["id"]: row for row in conn.execute("SELECT * FROM source_cases WHERE mapping_pathway = 'api'").fetchall()}
    counters: dict[str, int] = {}
    inserted = 0
    for mapping in valid_mappings:
        source = sources.get(mapping["source_case_id"])
        if not source:
            continue
        counters[mapping["case_id"]] = counters.get(mapping["case_id"], 0) + 1
        values = _entry_values_for_mapping(source, {**mapping, "index": counters[mapping["case_id"]]}, raw_payload)
        cur = conn.execute(
            """
            INSERT OR REPLACE INTO generated_entries(
              source_case_id, dedupe_key, component_label, case_id, case_date, case_year, role, site,
              patient_type, case_class, acgme_code, area, type, acgme_description, acgme_def_category, keyword, comments,
              mapping_rule_id, mapping_rule_version, mapping_rules_file_hash, mapping_rule_name, mapping_confidence,
              role_confidence, compound_flag, review_status, upload_status, evidence_excerpt, llm_model,
              llm_prompt_version, llm_raw_response_json, mapping_pathway, created_at, updated_at
            )
            VALUES (
              :source_case_id, :dedupe_key, :component_label, :case_id, :case_date, :case_year, :role, :site,
              :patient_type, :case_class, :acgme_code, :area, :type, :acgme_description, :acgme_def_category, :keyword, :comments,
              :mapping_rule_id, :mapping_rule_version, :mapping_rules_file_hash, :mapping_rule_name, :mapping_confidence,
              :role_confidence, :compound_flag, :review_status, 'not_uploaded', :evidence_excerpt, :llm_model,
              :llm_prompt_version, :llm_raw_response_json, :mapping_pathway, :created_at, :updated_at
            )
            """,
            {**values, "created_at": now, "updated_at": now},
        )
        entry_id = int(cur.lastrowid)
        log_event(conn, entry_id, "api_llm_imported", None, values["review_status"], "api_llm", values["comments"])
        inserted += 1
    conn.execute(
        """
        UPDATE source_cases
        SET source_mapping_status = 'api_llm_imported'
        WHERE id IN (
          SELECT DISTINCT source_case_id FROM generated_entries WHERE mapping_pathway = 'api'
        )
        """
    )
    return inserted


def import_llm_output_text(conn: sqlite3.Connection, text: str) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    valid, invalid, raw_payload = validate_llm_output(conn, text)
    inserted = import_validated_mappings(conn, valid, raw_payload) if valid else 0
    return inserted, valid, invalid


def set_api_review_status(conn: sqlite3.Connection, entry_ids: list[int], status: str, source: str = "review_app") -> None:
    if status not in {*API_ACCEPTED_STATUSES, "rejected", "error", "pending_review"}:
        raise ApiLLMError(f"Unsupported API LLM review status: {status}")
    now = utc_now()
    for entry_id in entry_ids:
        row = conn.execute(
            "SELECT review_status FROM generated_entries WHERE id = ? AND mapping_pathway = 'api'",
            (entry_id,),
        ).fetchone()
        if not row:
            continue
        conn.execute(
            "UPDATE generated_entries SET review_status = ?, updated_at = ? WHERE id = ?",
            (status, now, entry_id),
        )
        log_event(conn, entry_id, status, row["review_status"], status, source)


def update_api_mapping_target(conn: sqlite3.Connection, entry_id: int, target: CandidateTarget, status: str = "edited_manual") -> None:
    if status not in {"edited_manual", "accepted_manual"}:
        raise ApiLLMError(f"Unsupported manual status: {status}")
    row = conn.execute(
        "SELECT review_status FROM generated_entries WHERE id = ? AND mapping_pathway = 'api'",
        (entry_id,),
    ).fetchone()
    if not row:
        raise ApiLLMError(f"API LLM entry not found: {entry_id}")
    now = utc_now()
    conn.execute(
        """
        UPDATE generated_entries
        SET case_class = ?, acgme_code = ?, area = ?, type = ?, acgme_description = ?,
            acgme_def_category = ?, keyword = ?, review_status = ?, updated_at = ?
        WHERE id = ?
        """,
        (
            target.case_class or DEFAULT_CASE_CLASS,
            target.acgme_code,
            target.area,
            target.type,
            target.acgme_description,
            target.acgme_def_category,
            target.keyword,
            status,
            now,
            entry_id,
        ),
    )
    log_event(conn, entry_id, status, row["review_status"], status, "review_app", target_label(target))
