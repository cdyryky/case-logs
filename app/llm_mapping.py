from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .candidates import CandidateTarget, load_acgme_targets
from .constants import DEFAULT_CASE_CLASS
from .llm_client import LLMClientError, LLMResponseError, OllamaClient, ollama_health
from .utils import canonical_key

PROMPT_VERSION = "llm_acgme_v1"

LLM_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "procedures": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "acgme_code": {"type": "string"},
                    "evidence_excerpt": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "rationale": {"type": "string"},
                },
                "required": ["acgme_code", "evidence_excerpt", "confidence", "rationale"],
            },
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["procedures", "warnings"],
}

ABBREVIATIONS = {
    r"\bIVC\b": "inferior vena cava (IVC)",
    r"\bPCN\b": "percutaneous nephrostomy (PCN)",
    r"\bNUS\b": "nephroureteral stent (NUS)",
    r"\bNU\b": "nephroureteral (NU)",
    r"\bGJ\b": "gastrojejunostomy (GJ)",
    r"\bG tube\b": "gastrostomy tube",
    r"\bY[- ]?90\b": "Y90 radioembolization",
    r"\bTIPS\b": "transjugular intrahepatic portosystemic shunt (TIPS)",
    r"\bUAE\b": "uterine artery embolization (UAE)",
    r"\bPTA\b": "percutaneous transluminal angioplasty (PTA)",
    r"\bCVC\b": "central venous catheter (CVC)",
    r"\bTDC\b": "tunneled dialysis catheter (TDC)",
}


class LLMMappingError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMExtraction:
    entries: list[dict[str, Any]]
    warnings: list[str]
    raw_response: dict[str, Any]
    model: str
    prompt_version: str = PROMPT_VERSION


def _target_lookup() -> dict[str, CandidateTarget]:
    return {target.acgme_code: target for target in load_acgme_targets() if target.acgme_code}


def _target_catalog(targets: dict[str, CandidateTarget]) -> str:
    lines = []
    for code in sorted(targets):
        target = targets[code]
        lines.append(
            " | ".join(
                part
                for part in [
                    target.acgme_code,
                    target.acgme_description,
                    target.area,
                    target.type,
                    target.acgme_def_category,
                ]
                if part
            )
        )
    return "\n".join(lines)


def _compact_text(value: object, max_chars: int = 1400) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:max_chars].strip()


def normalize_prompt_text(value: str) -> str:
    text = value
    for pattern, replacement in ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.I)
    return text


def parsed_report(source: dict[str, Any]) -> dict[str, Any]:
    raw = source.get("parsed_report_json")
    if not raw:
        return {}
    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def build_llm_context(source: dict[str, Any]) -> dict[str, Any]:
    parsed = parsed_report(source)
    summaries: list[str] = []
    for section in parsed.get("procedure_summary_sections") or []:
        if not isinstance(section, dict):
            continue
        heading = section.get("heading") or "PROCEDURE SUMMARY"
        lines = []
        lines.extend(str(item) for item in section.get("bullets") or [] if str(item).strip())
        lines.extend(f"Additional procedure: {item}" for item in section.get("additional_procedures") or [] if str(item).strip())
        if lines:
            summaries.append(f"{heading}: " + " | ".join(lines))

    context = {
        "accession_number": source.get("accession_number", ""),
        "exam_code": source.get("exam_code", ""),
        "study_description": source.get("study_description", ""),
        "procedure_title": parsed.get("procedure_title") or source.get("procedure_text", ""),
        "procedure_list": parsed.get("procedure_list") or [],
        "impression": parsed.get("impression", ""),
        "fallback_findings": parsed.get("fallback_findings", ""),
        "procedure_summaries": summaries,
        "candidate_procedure_phrases": parsed.get("candidate_procedure_phrases") or [],
        "parse_warnings": parsed.get("parse_warnings") or [],
        "role_parse_source": source.get("role_parse_source", ""),
        "resident_found_in_report": bool(source.get("resident_found_in_report")),
    }
    normalized: dict[str, Any] = {}
    for key, value in context.items():
        if isinstance(value, list):
            normalized[key] = [normalize_prompt_text(_compact_text(item, 500)) for item in value[:20]]
        elif isinstance(value, str):
            normalized[key] = normalize_prompt_text(_compact_text(value))
        else:
            normalized[key] = value
    return normalized


def build_prompt(source: dict[str, Any], validation_error: str | None = None) -> str:
    targets = _target_lookup()
    context = build_llm_context(source)
    correction = f"\nPrevious response was invalid: {validation_error}\nReturn corrected JSON only." if validation_error else ""
    return (
        "You map Interventional Radiology dictations to official ACGME case-log target codes.\n"
        "Use only the ACGME target catalog below. Return every distinct performed diagnostic or therapeutic intervention.\n"
        "Do not include procedures described as not performed, aborted before intervention, deferred, planned, or only considered.\n"
        "Return JSON matching the provided schema. Do not include any fields outside the schema.\n"
        "For each procedure, provide the official acgme_code, a verbatim evidence excerpt from the case context, "
        "confidence high/medium/low, and a concise rationale.\n"
        f"{correction}\n\n"
        "CASE CONTEXT:\n"
        f"{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
        "OFFICIAL ACGME TARGET CATALOG:\n"
        f"{_target_catalog(targets)}"
    )


def _validate_response(payload: dict[str, Any]) -> list[dict[str, Any]]:
    procedures = payload.get("procedures")
    if not isinstance(procedures, list):
        raise LLMMappingError("Response field 'procedures' must be a list.")
    targets = _target_lookup()
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, item in enumerate(procedures):
        if not isinstance(item, dict):
            raise LLMMappingError(f"Procedure {idx + 1} must be an object.")
        extra = set(item) - {"acgme_code", "evidence_excerpt", "confidence", "rationale"}
        if extra:
            raise LLMMappingError(f"Procedure {idx + 1} contains unsupported fields: {', '.join(sorted(extra))}.")
        code = str(item.get("acgme_code") or "").strip()
        if code not in targets:
            raise LLMMappingError(f"Procedure {idx + 1} used invalid ACGME code: {code or '<missing>'}.")
        confidence = str(item.get("confidence") or "").strip().lower()
        if confidence not in {"high", "medium", "low"}:
            raise LLMMappingError(f"Procedure {idx + 1} has invalid confidence: {confidence or '<missing>'}.")
        evidence = _compact_text(item.get("evidence_excerpt"), 700)
        rationale = _compact_text(item.get("rationale"), 500)
        key = "|".join([code, canonical_key(evidence)])
        if key in seen:
            continue
        seen.add(key)
        validated.append(
            {
                "acgme_code": code,
                "evidence_excerpt": evidence,
                "confidence": confidence,
                "rationale": rationale,
                "target": targets[code],
            }
        )
    return validated


def _warnings(payload: dict[str, Any]) -> list[str]:
    warnings = payload.get("warnings")
    if not isinstance(warnings, list):
        return []
    return [_compact_text(item, 200) for item in warnings if _compact_text(item, 200)]


def llm_entries_for_source(
    source: dict[str, Any],
    client: OllamaClient | None = None,
) -> LLMExtraction:
    client = client or OllamaClient()
    validation_error: str | None = None
    last_raw: dict[str, Any] = {}
    for attempt in range(2):
        prompt = build_prompt(source, validation_error)
        try:
            payload, raw = client.generate_json(prompt, LLM_OUTPUT_SCHEMA)
        except (LLMClientError, LLMResponseError) as exc:
            raise LLMMappingError(str(exc)) from exc
        last_raw = raw
        try:
            procedures = _validate_response(payload)
        except LLMMappingError as exc:
            validation_error = str(exc)
            if attempt == 0:
                continue
            raise
        entries = []
        compound = int(len(procedures) > 1)
        for index, procedure in enumerate(procedures, start=1):
            target: CandidateTarget = procedure["target"]
            component_label = f"llm_{target.acgme_code}_{index}"
            dedupe_key = "|".join(
                [
                    source["accession_number"],
                    source["derived"]["case_date"],
                    DEFAULT_CASE_CLASS,
                    target.acgme_code,
                    target.area,
                    target.type,
                    target.acgme_description,
                    component_label,
                ]
            )
            comments = "; ".join(
                item
                for item in [
                    source.get("needs_review_reason", ""),
                    procedure["rationale"],
                ]
                if item
            )
            entries.append(
                {
                    "dedupe_key": dedupe_key,
                    "component_label": component_label,
                    "case_id": source["derived"]["case_id"],
                    "case_date": source["derived"]["case_date"],
                    "case_year": source["derived"]["case_year"],
                    "role": source["derived"]["role"],
                    "site": source["derived"]["site"],
                    "patient_type": source["derived"]["patient_type"],
                    "case_class": target.case_class or DEFAULT_CASE_CLASS,
                    "acgme_code": target.acgme_code,
                    "area": target.area,
                    "type": target.type,
                    "acgme_description": target.acgme_description,
                    "acgme_def_category": target.acgme_def_category,
                    "keyword": target.keyword,
                    "comments": comments,
                    "mapping_rule_id": f"llm:{client.settings.model}",
                    "mapping_rule_version": "1",
                    "mapping_rules_file_hash": PROMPT_VERSION,
                    "mapping_rule_name": "Local LLM extraction",
                    "mapping_confidence": procedure["confidence"],
                    "role_confidence": source["derived"]["role_confidence"],
                    "compound_flag": compound,
                    "review_status": "needs_review",
                    "evidence_excerpt": procedure["evidence_excerpt"],
                    "llm_model": client.settings.model,
                    "llm_prompt_version": PROMPT_VERSION,
                    "llm_raw_response_json": json.dumps(last_raw, ensure_ascii=False, sort_keys=True),
                }
            )
        return LLMExtraction(
            entries=entries,
            warnings=_warnings(payload),
            raw_response=last_raw,
            model=client.settings.model,
        )
    raise LLMMappingError("LLM extraction failed validation.")


def llm_health() -> dict[str, Any]:
    return ollama_health()
