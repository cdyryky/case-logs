from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config_io import load_dropdowns
from .constants import CONFIG_DIR
from .utils import canonical_key, canonical_text, file_sha256


@dataclass
class MappingRule:
    priority: int
    active: bool
    rule_id: str
    rule_version: str
    rule_name: str
    action: str
    match_exam_code: str
    match_study_description_regex: str
    match_procedure_regex: str
    match_report_regex: str
    exclude_procedure_regex: str
    exclude_report_regex: str
    case_class: str
    area: str
    type: str
    acgme_description: str
    acgme_def_category: str
    keyword: str
    component_label: str
    mapping_confidence: str
    needs_review_reason: str
    notes: str


def _truthy(value: str | None) -> bool:
    return canonical_text(value) in {"true", "1", "yes", "y"}


def load_mapping_rules(path: str | Path | None = None) -> tuple[list[MappingRule], str]:
    path = Path(path or CONFIG_DIR / "mapping_rules.csv")
    rules: list[MappingRule] = []
    with path.open(newline="") as f:
        for raw in csv.DictReader(f):
            rules.append(
                MappingRule(
                    priority=int(raw.get("priority") or 0),
                    active=_truthy(raw.get("active")),
                    rule_id=raw.get("rule_id") or raw.get("rule_name") or "",
                    rule_version=raw.get("rule_version") or "1",
                    rule_name=raw.get("rule_name") or "",
                    action=raw.get("action") or "generate",
                    match_exam_code=raw.get("match_exam_code") or "",
                    match_study_description_regex=raw.get("match_study_description_regex") or "",
                    match_procedure_regex=raw.get("match_procedure_regex") or "",
                    match_report_regex=raw.get("match_report_regex") or "",
                    exclude_procedure_regex=raw.get("exclude_procedure_regex") or "",
                    exclude_report_regex=raw.get("exclude_report_regex") or "",
                    case_class=raw.get("case_class") or "",
                    area=raw.get("area") or "",
                    type=raw.get("type") or "",
                    acgme_description=raw.get("acgme_description") or "",
                    acgme_def_category=raw.get("acgme_def_category") or "",
                    keyword=raw.get("keyword") or "",
                    component_label=raw.get("component_label") or "dominant_procedure",
                    mapping_confidence=raw.get("mapping_confidence") or "medium",
                    needs_review_reason=raw.get("needs_review_reason") or "",
                    notes=raw.get("notes") or "",
                )
            )
    return sorted(rules, key=lambda r: r.priority, reverse=True), file_sha256(path)


def _rx_match(pattern: str, text: str) -> bool:
    return bool(pattern and re.search(pattern, text or "", flags=re.I | re.S))


def rule_matches(rule: MappingRule, source: dict[str, Any]) -> bool:
    if not rule.active:
        return False
    if rule.match_exam_code and canonical_text(rule.match_exam_code) != canonical_text(source.get("exam_code")):
        return False
    if rule.match_study_description_regex and not _rx_match(
        rule.match_study_description_regex, source.get("study_description", "")
    ):
        return False
    if rule.match_procedure_regex and not _rx_match(rule.match_procedure_regex, source.get("procedure_text", "")):
        return False
    if rule.match_report_regex and not _rx_match(rule.match_report_regex, source.get("report_snippet", "")):
        return False
    if rule.exclude_procedure_regex and _rx_match(rule.exclude_procedure_regex, source.get("procedure_text", "")):
        return False
    if rule.exclude_report_regex and _rx_match(rule.exclude_report_regex, source.get("report_snippet", "")):
        return False
    return any(
        [
            rule.match_exam_code,
            rule.match_study_description_regex,
            rule.match_procedure_regex,
            rule.match_report_regex,
        ]
    )


def validate_dropdown(rule: MappingRule, dropdowns: dict[str, Any]) -> None:
    if rule.action != "generate":
        return
    valid = {
        (
            canonical_key(item["class"]["visible_label"]),
            canonical_key(item["area"]["visible_label"]),
            canonical_key(item["type"]["visible_label"]),
        )
        for item in dropdowns.get("case_options", [])
    }
    key = (canonical_key(rule.case_class), canonical_key(rule.area), canonical_key(rule.type))
    if key not in valid:
        raise ValueError(
            f"Mapping rule '{rule.rule_name}' targets invalid dropdown combination: "
            f"{rule.case_class} / {rule.area} / {rule.type}"
        )


def validate_rules(rules: list[MappingRule], dropdowns: dict[str, Any] | None = None) -> None:
    config = dropdowns or load_dropdowns()
    for rule in rules:
        validate_dropdown(rule, config)


def map_source_case(
    source: dict[str, Any],
    rules: list[MappingRule],
    mapping_rules_file_hash: str,
    dropdowns: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dropdown_config = dropdowns or load_dropdowns()
    validate_rules(rules, dropdown_config)
    generated: list[dict[str, Any]] = []
    source_update = {"source_mapping_status": "unmapped", "needs_review_reason": source.get("needs_review_reason") or ""}

    for rule in rules:
        if not rule_matches(rule, source):
            continue
        if rule.action == "exclude":
            source_update["source_mapping_status"] = "excluded"
            source_update["needs_review_reason"] = rule.needs_review_reason or "Excluded by mapping rule"
            return [], source_update
        if rule.action == "flag_only":
            source_update["source_mapping_status"] = "flag_only"
            source_update["needs_review_reason"] = rule.needs_review_reason or "Flagged by mapping rule"
            continue
        if rule.action != "generate":
            continue

        derived = source["derived"]
        review_reasons = [r for r in [source.get("needs_review_reason"), rule.needs_review_reason] if r]
        role_confidence = derived["role_confidence"]
        mapping_confidence = rule.mapping_confidence
        review_status = "new_high_confidence" if mapping_confidence == "high" and role_confidence == "high" else "needs_review"
        if review_reasons:
            review_status = "needs_review"
        component_label = rule.component_label or "dominant_procedure"
        dedupe_key = "|".join(
            [
                source["accession_number"],
                derived["case_date"],
                rule.case_class,
                rule.area,
                rule.type,
                rule.acgme_description,
                component_label,
            ]
        )
        generated.append(
            {
                "dedupe_key": dedupe_key,
                "component_label": component_label,
                "case_id": derived["case_id"],
                "case_date": derived["case_date"],
                "case_year": derived["case_year"],
                "role": derived["role"],
                "site": derived["site"],
                "patient_type": derived["patient_type"],
                "case_class": rule.case_class,
                "area": rule.area,
                "type": rule.type,
                "acgme_description": rule.acgme_description,
                "acgme_def_category": rule.acgme_def_category,
                "keyword": rule.keyword,
                "comments": "; ".join(review_reasons),
                "mapping_rule_id": rule.rule_id,
                "mapping_rule_version": rule.rule_version,
                "mapping_rules_file_hash": mapping_rules_file_hash,
                "mapping_rule_name": rule.rule_name,
                "mapping_confidence": mapping_confidence,
                "role_confidence": role_confidence,
                "compound_flag": 0,
                "review_status": review_status,
            }
        )

    unique: dict[str, dict[str, Any]] = {}
    for entry in generated:
        unique[entry["dedupe_key"]] = entry
    entries = list(unique.values())
    if entries:
        compound = int(len(entries) > 1 or any(e["component_label"] != "dominant_procedure" for e in entries))
        for entry in entries:
            entry["compound_flag"] = compound
            if compound:
                entry["review_status"] = "needs_review"
        source_update["source_mapping_status"] = "generated"
    return entries, source_update
