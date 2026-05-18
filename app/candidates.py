from __future__ import annotations

import csv
import json
import re
import sqlite3
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any

from .constants import CONFIG_DIR
from .mapper import MappingRule, load_mapping_rules, rule_matches
from .matching import match_tokens, normalize_match_text
from .models import log_event, utc_now
from .utils import canonical_key

ALGORITHM_VERSION = "candidate_v5_multi_action_events"
ACGME_TARGETS_PATH = CONFIG_DIR / "acgme_targets.csv"


@dataclass(frozen=True)
class CandidateTarget:
    case_class: str
    area: str
    type: str
    acgme_code: str = ""
    acgme_description: str = ""
    acgme_def_category: str = ""
    keyword: str = ""
    component_label: str = "dominant_procedure"
    source_kind: str = "dropdown"
    rule_id: str = ""
    rule_name: str = ""


@dataclass(frozen=True)
class ProcedureEvent:
    key: str
    label: str
    phrases: tuple[str, ...]
    target_area: str
    target_type: str
    target_description: str = ""
    target_def_category: str = ""
    reason: str = ""
    score: float = 0.95


def parsed_report(source: dict[str, Any] | sqlite3.Row) -> dict[str, Any]:
    raw = source["parsed_report_json"] if isinstance(source, sqlite3.Row) else source.get("parsed_report_json")
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def evidence_phrases(source: dict[str, Any] | sqlite3.Row) -> list[str]:
    parsed = parsed_report(source)
    values: list[str] = []
    for value in [
        source["exam_code"] if isinstance(source, sqlite3.Row) else source.get("exam_code", ""),
        source["study_description"] if isinstance(source, sqlite3.Row) else source.get("study_description", ""),
        source["procedure_text"] if isinstance(source, sqlite3.Row) else source.get("procedure_text", ""),
        source["modality"] if isinstance(source, sqlite3.Row) and "modality" in source.keys() else (source.get("modality", "") if isinstance(source, dict) else ""),
        source["cpt_code"] if isinstance(source, sqlite3.Row) and "cpt_code" in source.keys() else (source.get("cpt_code", "") if isinstance(source, dict) else ""),
        parsed.get("procedure_title", ""),
        parsed.get("impression", ""),
        parsed.get("fallback_findings", ""),
    ]:
        if value:
            values.extend(str(value).splitlines())
    values.extend(str(v) for v in parsed.get("procedure_list") or [])
    values.extend(str(v) for v in parsed.get("candidate_procedure_phrases") or [])
    for section in parsed.get("procedure_summary_sections") or []:
        if not isinstance(section, dict):
            continue
        values.extend(str(v) for v in section.get("bullets") or [])
        values.extend(str(v) for v in section.get("additional_procedures") or [])
    cleaned = [re.sub(r"\s+", " ", value).strip() for value in values if str(value).strip()]
    return list(dict.fromkeys(cleaned))


def target_text(target: CandidateTarget) -> str:
    return " ".join(
        value
        for value in [
            target.acgme_code,
            target.area,
            target.type,
            target.acgme_description,
            target.acgme_def_category,
            target.keyword,
            target.rule_name,
        ]
        if value
    )


@lru_cache(maxsize=512)
def cached_target_text(target: CandidateTarget) -> str:
    return target_text(target)


@lru_cache(maxsize=512)
def cached_target_tokens(target: CandidateTarget) -> frozenset[str]:
    return frozenset(match_tokens(cached_target_text(target)))


@lru_cache(maxsize=512)
def cached_normalized_target_text(target: CandidateTarget) -> str:
    return normalize_match_text(cached_target_text(target))


def target_key(target: CandidateTarget) -> str:
    return "|".join(
        [
            target.case_class,
            target.acgme_code,
            target.area,
            target.type,
            target.acgme_description,
            target.acgme_def_category,
        ]
    )


def _target_from_parts(
    area: str,
    typ: str,
    description: str = "",
    def_cat: str = "",
    code: str = "",
    source_kind: str = "dictionary",
) -> CandidateTarget:
    return CandidateTarget(
        case_class="Interventional Procedures",
        area=re.sub(r"\s+", " ", str(area or "").replace("\u00a0", " ")).strip(),
        type=re.sub(r"\s+", " ", str(typ or "").replace("\u00a0", " ")).strip(),
        acgme_code=re.sub(r"\s+", " ", str(code or "").replace("\u00a0", " ")).strip(),
        acgme_description=re.sub(r"\s+", " ", str(description or "").replace("\u00a0", " ")).strip(),
        acgme_def_category=re.sub(r"\s+", " ", str(def_cat or "").replace("\u00a0", " ")).strip(),
        source_kind=source_kind,
    )


def confidence_for_score(score: float) -> str:
    if score >= 0.78:
        return "high"
    if score >= 0.58:
        return "medium"
    return "low"


def prepared_phrases(phrases: list[str]) -> list[tuple[str, set[str], str]]:
    return [(phrase, match_tokens(phrase), normalize_match_text(phrase)) for phrase in phrases]


def score_target(
    target: CandidateTarget,
    phrase_features: list[tuple[str, set[str], str]],
    exact_rule_match: bool = False,
) -> tuple[float, list[str], str]:
    target_tokens = set(cached_target_tokens(target))
    if not target_tokens:
        return 0.0, [], "No target tokens."
    target_norm = cached_normalized_target_text(target)
    best_score = 0.0
    matched: list[str] = []
    for phrase, phrase_tokens, phrase_norm in phrase_features:
        if not phrase_tokens:
            continue
        shared = phrase_tokens & target_tokens
        if not shared and not exact_rule_match:
            continue
        coverage = len(shared) / max(len(target_tokens), 1)
        overlap = len(shared) / max(len(phrase_tokens | target_tokens), 1)
        phrase_ratio = SequenceMatcher(None, phrase_norm, target_norm).ratio()
        score = (coverage * 0.55) + (overlap * 0.25) + (phrase_ratio * 0.20)
        if shared and any(token in normalize_match_text(phrase) for token in ("embolization", "biopsy", "paracentesis", "angiography", "venography")):
            score += 0.08
        if score > best_score:
            best_score = score
            matched = [phrase]
        elif score >= 0.58 and len(matched) < 5:
            matched.append(phrase)
    if exact_rule_match:
        best_score = max(best_score, 0.88)
    reason = f"Matched {', '.join(sorted((match_tokens(' '.join(matched)) & target_tokens))[:8]) or 'parsed report text'}."
    return min(best_score, 1.0), matched, reason


def load_acgme_targets(path: str | None = None) -> list[CandidateTarget]:
    target_path = ACGME_TARGETS_PATH if path is None else path
    seen: dict[str, CandidateTarget] = {}
    with open(target_path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"Code", "Description", "Defined Category", "Area", "Type"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing ACGME target columns: {', '.join(sorted(missing))}")
        for row in reader:
            target = _target_from_parts(
                row.get("Area", ""),
                row.get("Type", ""),
                row.get("Description", ""),
                row.get("Defined Category", ""),
                row.get("Code", ""),
                "acgme_csv",
            )
            if target.acgme_code and target.area and target.type:
                seen[target.acgme_code] = target
    return list(seen.values())


@lru_cache(maxsize=1)
def cached_official_targets() -> tuple[CandidateTarget, ...]:
    return tuple(load_acgme_targets())


def rule_targets(rules: list[MappingRule]) -> list[CandidateTarget]:
    official = list(cached_official_targets())
    targets: list[CandidateTarget] = []
    for rule in rules:
        if not rule.active or rule.action != "generate":
            continue
        targets.append(_resolve_rule_target(rule, official))
    return targets


@lru_cache(maxsize=1)
def cached_rules_and_targets() -> tuple[tuple[MappingRule, ...], tuple[CandidateTarget, ...]]:
    rules, _ = load_mapping_rules()
    return tuple(rules), tuple(rule_targets(rules))


def learning_target_key(value: dict[str, Any] | sqlite3.Row) -> str:
    return "|".join(
        [
            value["case_class"],
            value["acgme_code"] or "" if "acgme_code" in value.keys() else "",
            value["area"],
            value["type"],
            value["acgme_description"] or "",
            value["acgme_def_category"] or "",
        ]
    )


def load_learning_signals(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT *
        FROM mapping_learning_signals
        ORDER BY id DESC
        LIMIT 1000
        """
    ).fetchall()


def apply_learning_signals(
    source: dict[str, Any] | sqlite3.Row,
    candidates: list[dict[str, Any]],
    signals: list[sqlite3.Row] | None,
) -> list[dict[str, Any]]:
    if not signals:
        return candidates
    source_exam = source["exam_code"] if isinstance(source, sqlite3.Row) else source.get("exam_code", "")
    source_tokens = match_tokens(
        source["procedure_text"] if isinstance(source, sqlite3.Row) else source.get("procedure_text", ""),
        source["study_description"] if isinstance(source, sqlite3.Row) else source.get("study_description", ""),
    )
    adjusted: list[dict[str, Any]] = []
    for candidate in candidates:
        item = {**candidate}
        boost = 0.0
        key = learning_target_key(item)
        for signal in signals:
            if learning_target_key(signal) != key:
                continue
            same_exam = signal["exam_code"] and signal["exam_code"] == source_exam
            signal_tokens = match_tokens(signal["procedure_text"] or "")
            token_overlap = bool(source_tokens & signal_tokens)
            if same_exam or token_overlap:
                boost += 0.08 * float(signal["learning_weight"])
        if boost:
            item["score"] = round(max(0.0, min(1.0, float(item["score"]) + boost)), 4)
            item["confidence"] = confidence_for_score(float(item["score"]))
            item["default_checked"] = int(
                bool(item.get("default_checked"))
                and item.get("source_kind") in {"event_alias", "rule"}
                and item["confidence"] == "high"
            )
            item["match_reason"] = f"{item['match_reason']} Learning signal adjusted score."
        adjusted.append(item)
    return sorted(adjusted, key=lambda row: row["score"], reverse=True)


def _source_domain_text(source: dict[str, Any] | sqlite3.Row, phrases: list[str]) -> str:
    values = [
        source["exam_code"] if isinstance(source, sqlite3.Row) else source.get("exam_code", ""),
        source["study_description"] if isinstance(source, sqlite3.Row) else source.get("study_description", ""),
        source["procedure_text"] if isinstance(source, sqlite3.Row) else source.get("procedure_text", ""),
        " ".join(phrases),
    ]
    return normalize_match_text(" ".join(str(value or "") for value in values))


def _has_domain_support(target: CandidateTarget, source_text: str) -> bool:
    area = canonical_key(target.area)
    typ = canonical_key(target.type)
    target_norm = normalize_match_text(target_text(target))
    domain_patterns = {
        "gu_intervention": r"\b(?:gu|genitourinary|nephrostomy|nephroureteral|ureter|ureteral|upj|pcn|kidney|renal pelvis|urinary)\b",
        "drainage_procedures": r"\b(?:drainage|abscess|fluid collection|paracentesis|thoracentesis|aspiration|peritoneal|pleural)\b",
        "venous_interventions": r"\b(?:venous|vein|ivc|inferior vena cava|filter|venography|venogram|dvt)\b",
        "arterial_interventions": r"\b(?:arterial|artery|arteriography|angiography|transarterial|embolization|embolize)\b",
        "biliary_interventions": r"\b(?:biliary|bile duct|cholecystostomy|gallbladder|cholangiogram)\b",
        "gi_intervention": r"\b(?:gastrostomy|gastrojejunostomy|jejunostomy|gi|g tube|gj tube)\b",
        "venous_access_general": r"\b(?:port|central venous|cvc|tunneled|picc|dialysis catheter|venous access)\b",
        "diagnostic_arteriography": r"\b(?:arteriography|angiography|arteriogram|celiac|hepatic|visceral angiogram)\b",
        "dialysis_shunt_management": r"\b(?:dialysis|fistula|fistulagram|graft|declot|av access)\b",
        "biopsy": r"\b(?:biopsy|core needle|fine needle|fn[ab])\b",
        "portal_interventions": r"\b(?:portal|tips|transjugular intrahepatic)\b",
        "pulmonary_arterial_interventions": r"\b(?:pulmonary artery|pulmonary arterial|pulmonary embolism|pe thrombectomy)\b",
    }
    pattern = domain_patterns.get(area)
    if not pattern:
        return True
    if re.search(pattern, source_text, re.I):
        return True
    if area == "arterial_interventions" and re.search(r"\bembol", source_text):
        return True
    if area == "venous_interventions" and "ivc" in target_norm and re.search(r"\bfilter\b", source_text):
        return True
    if typ == "paracentesis" and re.search(r"\bparacentesis\b", source_text):
        return True
    return False


def _is_generic_device_match(shared: set[str]) -> bool:
    generic = {"catheter", "tube", "drain", "exchange", "exchanged", "change", "changed", "stent", "check", "checked"}
    return bool(shared) and shared <= generic


def _has_negative_evidence(value: str) -> bool:
    text = normalize_match_text(value)
    return bool(
        re.search(
            r"\b(?:not performed|not attempted|no intervention|procedure was not performed|deferred|aborted|unsuccessful|without placement|planned|plan for|possible|considered|will consider|may need)\b",
            text,
        )
    )


def _positive_phrases(phrases: list[str]) -> list[str]:
    return [phrase for phrase in phrases if not _has_negative_evidence(phrase)]


def _positive_source_text(source: dict[str, Any] | sqlite3.Row, phrases: list[str]) -> str:
    return _source_domain_text(source, _positive_phrases(phrases))


def _target_requires_specific_evidence(target: CandidateTarget, source_text: str) -> bool:
    desc = canonical_key(target.acgme_description)
    typ = canonical_key(target.type)
    area = canonical_key(target.area)

    organ_terms = {
        "biopsy_adrenal": r"\badrenal\b",
        "biopsy_biliary": r"\b(?:biliary|bile duct|gallbladder)\b",
        "biopsy_spleen": r"\b(?:spleen|splenic)\b",
        "biopsy_genitourinary": r"\b(?:renal|kidney|nephro|ureter|bladder|prostate|testicular|genitourinary|gu)\b",
        "biopsy_lung": r"\b(?:lung|pulmonary)\b",
        "biopsy_mediastinum": r"\bmediastin",
        "biopsy_soft_tissue": r"\b(?:soft tissue|subcutaneous|superficial|muscle|fat pad)\b",
        "biopsy_joint": r"\bjoint\b",
        "biopsy_bone_marrow": r"\b(?:bone marrow|marrow)\b",
        "biopsy_cervical_nodal": r"\b(?:cervical|neck).*\b(?:node|nodal|lymph)\b|\b(?:node|nodal|lymph).*\b(?:cervical|neck)\b",
        "biopsy_thyroid": r"\bthyroid\b",
        "biopsy_lymph_node": r"\b(?:lymph node|nodal|node biopsy)\b",
    }
    if desc in organ_terms:
        return bool(re.search(organ_terms[desc], source_text))

    if desc == "uterine_artery_embolization":
        return bool(re.search(r"\b(?:uterine|uterus|fibroid|uae)\b", source_text))
    if desc == "embolization_of_tumor_radioembolization":
        return bool(re.search(r"\b(?:radioembolization|y-?90|yttrium|radioisotope|therasphere|sir-?spheres)\b", source_text))
    if desc == "stent_visceral_artery":
        return bool(re.search(r"\b(?:arterial|artery|visceral|celiac|mesenteric|hepatic artery|splenic artery)\b", source_text))
    if area == "biliary_interventions" and typ == "biliary_tube_maintenance":
        return bool(re.search(r"\bbiliary\b.*\b(?:drain|tube|catheter)\b.*\b(?:exchange|exchanged|internaliz)", source_text))
    return True


def _rule_can_default_check(target: CandidateTarget, source_text: str, shared_tokens: set[str]) -> bool:
    if not _has_domain_support(target, source_text):
        return False
    if _is_generic_device_match(shared_tokens):
        return False
    return _target_requires_specific_evidence(target, source_text)


def _find_target(
    targets: list[CandidateTarget],
    area: str,
    typ: str,
    description: str = "",
    def_cat: str = "",
) -> CandidateTarget:
    desired = (
        canonical_key(area),
        canonical_key(typ),
        canonical_key(description),
        canonical_key(def_cat),
    )
    for target in targets:
        key = (
            canonical_key(target.area),
            canonical_key(target.type),
            canonical_key(target.acgme_description),
            canonical_key(target.acgme_def_category),
        )
        if key == desired:
            return target
    if description:
        for target in targets:
            key = (
                canonical_key(target.area),
                canonical_key(target.type),
                canonical_key(target.acgme_description),
            )
            if key == desired[:3]:
                return target
    for target in targets:
        if canonical_key(target.area) == desired[0] and canonical_key(target.type) == desired[1]:
            return target
    return _target_from_parts(area, typ, description, def_cat, source_kind="fallback")


def _resolve_rule_target(rule: MappingRule, targets: list[CandidateTarget]) -> CandidateTarget:
    rule_area = canonical_key(rule.area)
    rule_type = canonical_key(rule.type)
    rule_desc = canonical_key(rule.acgme_description)
    rule_def = canonical_key(rule.acgme_def_category)
    compatible = [
        target
        for target in targets
        if canonical_key(target.area) == rule_area
        and canonical_key(target.type) == rule_type
        and (not rule_desc or canonical_key(target.acgme_description) == rule_desc)
        and (not rule_def or canonical_key(target.acgme_def_category) == rule_def)
    ]
    if compatible:
        rule_tokens = match_tokens(rule.rule_name, rule.match_procedure_regex, rule.match_report_regex, rule.match_study_description_regex)
        compatible.sort(
            key=lambda target: (
                len(rule_tokens & match_tokens(target.acgme_description, target.acgme_def_category)),
                bool(target.acgme_description),
                target.acgme_code,
            ),
            reverse=True,
        )
        target = compatible[0]
        return CandidateTarget(
            case_class=rule.case_class,
            area=target.area,
            type=target.type,
            acgme_code=target.acgme_code,
            acgme_description=target.acgme_description,
            acgme_def_category=target.acgme_def_category,
            keyword=rule.keyword,
            component_label=rule.component_label or "dominant_procedure",
            source_kind="rule",
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
        )
    return CandidateTarget(
        case_class=rule.case_class,
        area=rule.area,
        type=rule.type,
        acgme_description=rule.acgme_description,
        acgme_def_category=rule.acgme_def_category,
        keyword=rule.keyword,
        component_label=rule.component_label or "dominant_procedure",
        source_kind="rule",
        rule_id=rule.rule_id,
        rule_name=rule.rule_name,
    )


def _combine_event_phrases(phrases: list[str], pattern: str) -> tuple[str, ...]:
    matched = [
        phrase
        for phrase in _positive_phrases(phrases)
        if re.search(pattern, normalize_match_text(phrase), re.I)
    ]
    return tuple(dict.fromkeys(matched))


ACTION_PATTERNS = {
    "pta": (
        r"\b(?:angioplasty|balloon (?:dilation|dilatation)|dilation|dilatation|pta|venoplasty|"
        r"cutting balloon|scoring balloon|high pressure balloon)\b"
    ),
    "stent": r"\b(?:stent placement|stenting|stented|endovascular stent|bare metal stent|self expanding stent|balloon expandable stent|relining)\b",
    "stent_graft": r"\b(?:stent graft|covered stent|endograft|endoprosthesis|tevar|evar|aorto uni iliac|aui)\b",
    "atherectomy": r"\b(?:atherectomy|plaque excision|directional atherectomy|orbital atherectomy|rotational atherectomy|laser atherectomy)\b",
    "thrombolysis": r"\b(?:thrombolysis|lysis catheter|catheter directed thrombolysis|cdt|tpa|alteplase|ekos|thrombolytic infusion)\b",
    "thrombectomy": r"\b(?:mechanical thrombectomy|aspiration thrombectomy|suction thrombectomy|thrombectomy|clot extraction|embolectomy|thrombus maceration|flowtriever|clottriever|penumbra|angiojet)\b",
    "biliary_stricture_dilation": r"\b(?:bilioplasty|cholangioplasty|biliary stricture dilation|bile duct dilation)\b",
    "drain_check": r"\b(?:tube check|catheter check|drain check|sinogram|abscessogram)\b",
}


def _append_event(events: list[ProcedureEvent], event: ProcedureEvent) -> None:
    if not event.phrases:
        return
    if any(existing.key == event.key for existing in events):
        return
    events.append(event)


def _has_any(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, re.I))


def _arterial_pta_or_stent_description(source_text: str, action: str) -> tuple[str, str, str, str]:
    if _has_any(source_text, r"\b(?:tevar|thoracic aorta|descending thoracic|thoracic endograft)\b"):
        if action == "stent_graft":
            return "Arterial Interventions", "Aortic stent grafting", "Stent graft thoracic aorta", "Aortic Stent Grafting"
        return "Arterial Interventions", f"Arterial {action}", f"{'PTA' if action == 'PTA' else 'Stent'} - aorta", "Arterial PTA or Stent"
    if _has_any(source_text, r"\b(?:evar|abdominal aorta|aaa|infrarenal|aorto uni iliac|aui|aortic)\b"):
        if action == "stent_graft":
            return "Arterial Interventions", "Aortic stent grafting", "Straight tube stent graft abdominal aorta", "Aortic Stent Grafting"
        return "Arterial Interventions", f"Arterial {action}", f"{'PTA' if action == 'PTA' else 'Stent'} - aorta", "Arterial PTA or Stent"
    if action == "stent_graft" and _has_any(source_text, r"\b(?:iliac|common iliac|external iliac|internal iliac|hypogastric)\b"):
        return "Arterial Interventions", "Aortic stent grafting", "Iliac artery repair with stent graft", "Aortic Stent Grafting"
    if _has_any(source_text, r"\b(?:pulmonary artery|pulmonary arterial|pa stent|pa angioplasty)\b"):
        return "Pulmonary Arterial Interventions", f"Pulmonary artery {'PTA' if action == 'PTA' else 'stent'}", f"Pulmonary artery {'PTA' if action == 'PTA' else 'stent'}", "Arterial PTA or Stent"
    if _has_any(source_text, r"\b(?:carotid|vertebral|intracranial|extracranial|mca|aca|pca|basilar|cerebral|neuro)\b"):
        return "Neurovascular Interventions", f"Neuro arterial {'PTA' if action == 'PTA' else 'stent'}", f"Neuro - arterial {'PTA' if action == 'PTA' else 'stent'}", "Neuro dx/intervention"
    if _has_any(source_text, r"\b(?:iliac artery|common femoral|external iliac artery|internal iliac artery|sfa|superficial femoral|profunda|femoral artery|popliteal|tibial|peroneal|pedal|runoff)\b"):
        return "Arterial Interventions", f"Arterial {'PTA' if action == 'PTA' else 'stent'}", f"{'PTA' if action == 'PTA' else 'Stent'} - lower extremity artery", "Arterial PTA or Stent"
    if _has_any(source_text, r"\b(?:subclavian artery|axillary artery|brachial artery|radial artery|ulnar artery|upper extremity artery)\b"):
        return "Arterial Interventions", f"Arterial {'PTA' if action == 'PTA' else 'stent'}", f"{'PTA artery' if action == 'PTA' else 'Stent -'} upper extremity artery", "Arterial PTA or Stent"
    if _has_any(source_text, r"\b(?:celiac|hepatic artery|splenic artery|left gastric|gastroduodenal|gda|sma|ima|renal artery|mesenteric|bronchial|intercostal|lumbar|visceral)\b"):
        return "Arterial Interventions", f"Arterial {'PTA' if action == 'PTA' else 'stent'}", f"{'PTA artery' if action == 'PTA' else 'Stent -'} visceral artery", "Arterial PTA or Stent"
    return "Arterial Interventions", f"Arterial {'PTA' if action == 'PTA' else 'stent'}", f"{'PTA' if action == 'PTA' else 'Stent'} - lower extremity artery", "Arterial PTA or Stent"


def _venous_pta_or_stent_description(source_text: str, action: str) -> tuple[str, str, str, str]:
    if _has_any(source_text, r"\b(?:portal vein|tips|transjugular intrahepatic)\b"):
        return "Portal Interventions", f"Portal vein {'PTA' if action == 'PTA' else 'stent'}", f"Portal vein {'PTA' if action == 'PTA' else 'stent'}", "Venous Intervention"
    if _has_any(source_text, r"\bsvc\b|\bsuperior vena cava\b"):
        return "Venous Interventions", f"Venous {'PTA' if action == 'PTA' else 'stent'}", f"{'PTA' if action == 'PTA' else 'Stent'} SVC", "Venous Intervention"
    if _has_any(source_text, r"\bivc\b|\binferior vena cava\b"):
        return "Venous Interventions", f"Venous {'PTA' if action == 'PTA' else 'stent'}", f"{'PTA' if action == 'PTA' else 'Stent'} IVC", "Venous Intervention"
    if _has_any(source_text, r"\b(?:brachiocephalic|subclavian vein|central venous|central vein)\b"):
        desc = "PTA venous central" if action == "PTA" else "Stent brachiocephalic/subclavian vein"
        return "Venous Interventions", f"Venous {'PTA' if action == 'PTA' else 'stent'}", desc, "Venous Intervention"
    if _has_any(source_text, r"\b(?:renal vein|gonadal|adrenal|hepatic vein|visceral vein)\b"):
        desc = "PTA visceral/renal vein" if action == "PTA" else "Stent visceral/renal vein"
        return "Venous Interventions", f"Venous {'PTA' if action == 'PTA' else 'stent'}", desc, "Venous Intervention"
    if _has_any(source_text, r"\b(?:upper extremity vein|axillary vein|brachial vein|basilic|cephalic)\b"):
        desc = "PTA upper extremity vein" if action == "PTA" else "Stent brachiocephalic/subclavian vein"
        return "Venous Interventions", f"Venous {'PTA' if action == 'PTA' else 'stent'}", desc, "Venous Intervention"
    return "Venous Interventions", f"Venous {'PTA' if action == 'PTA' else 'stent'}", f"{'PTA' if action == 'PTA' else 'Stent'} lower extremity vein", "Venous Intervention"


def _vascular_action_target(source_text: str, action: str) -> tuple[str, str, str, str]:
    if _has_any(source_text, r"\b(?:fistula|fistulagram|fistulogram|graftogram|avf|avg|dialysis access|dialysis graft|dialysis fistula|access declot)\b"):
        if action == "PTA":
            return "Dialysis Shunt Management", "Dialysis access PTA", "Dialysis access PTA", "Dialysis Access"
        if action == "stent":
            return "Dialysis Shunt Management", "Dialysis access stent", "Dialysis access stent", "Dialysis Access"
    if action != "stent_graft" and _has_any(source_text, r"\b(?:venous|vein|venogram|venography|venoplasty|svc|ivc|brachiocephalic|subclavian vein|iliac vein|femoral vein|renal vein|portal vein|tips)\b"):
        return _venous_pta_or_stent_description(source_text, action)
    return _arterial_pta_or_stent_description(source_text, action)


def _thrombus_target(source_text: str, action: str) -> tuple[str, str, str, str]:
    is_lysis = action == "thrombolysis"
    if _has_any(source_text, r"\b(?:pulmonary artery|pulmonary embolism|pe thrombectomy|pe\b)\b"):
        desc = "Pulmonary artery thrombolysis" if is_lysis else "Pulmonary artery thrombectomy"
        return "Pulmonary Arterial Interventions", desc, desc, "Thrombolysis/Thrombectomy"
    if _has_any(source_text, r"\b(?:portal vein|tips)\b"):
        desc = "Portal vein thrombolysis - initial" if is_lysis else "Portal vein thrombectomy"
        return "Portal Interventions", desc, desc, "Thrombolysis/Thrombectomy"
    if _has_any(source_text, r"\b(?:fistula|fistulagram|fistulogram|graftogram|avf|avg|dialysis access|access declot)\b"):
        return "Dialysis Shunt Management", "Dialysis access thrombectomy/thrombolysis", "Dialysis access thrombectomy/thrombolysis", "Dialysis Access"
    if _has_any(source_text, r"\b(?:carotid|vertebral|intracranial|mca|aca|pca|basilar|cerebral|neuro)\b"):
        desc = "Thrombolysis neuro artery" if is_lysis else "Thrombectomy neuro artery"
        return "Neurovascular Interventions", desc, desc, "Neuro dx/intervention"
    if _has_any(source_text, r"\b(?:venous|vein|dvt|svc|ivc|brachiocephalic|subclavian vein|iliac vein|femoral vein|renal vein)\b"):
        if _has_any(source_text, r"\bsvc\b|\bsuperior vena cava\b"):
            desc = "Thrombolysis catheter SVC" if is_lysis else "Mechanical thrombectomy SVC"
        elif _has_any(source_text, r"\bivc\b|\binferior vena cava\b"):
            desc = "Thrombolysis catheter IVC" if is_lysis else "Mechanical thrombectomy IVC"
        elif _has_any(source_text, r"\b(?:renal vein|visceral vein)\b"):
            desc = "Thrombolysis catheter visceral/renal vein" if is_lysis else "Mechanical thrombectomy visceral/renal vein"
        elif _has_any(source_text, r"\b(?:subclavian|upper extremity|axillary|brachial)\b"):
            desc = "Thrombolysis catheter upper extremity vein" if is_lysis else "Mechanical thrombectomy upper extremity vein"
        else:
            desc = "Thrombolysis catheter lower extremity vein" if is_lysis else "Mechanical thrombectomy lower extremity vein"
        return "Venous Interventions", "Venous thrombolysis - initial" if is_lysis else "Venous mechanical thrombectomy", desc, "Thrombolysis/Thrombectomy"
    if _has_any(source_text, r"\b(?:aorta|aortic)\b"):
        desc = "Thrombolysis - aorta" if is_lysis else "Mechanical thrombectomy - aorta"
    elif _has_any(source_text, r"\b(?:subclavian artery|axillary artery|brachial artery|radial artery|ulnar artery|upper extremity artery)\b"):
        desc = "Thrombolysis - upper extremity artery (excluding neuro)" if is_lysis else "Mechanical thrombectomy - upper extremity artery"
    elif _has_any(source_text, r"\b(?:celiac|hepatic artery|splenic artery|sma|ima|renal artery|mesenteric|visceral)\b"):
        desc = "Thrombolysis - visceral artery" if is_lysis else "Mechanical thrombectomy - visceral artery"
    else:
        desc = "Thrombolysis - lower extremity artery" if is_lysis else "Mechanical thrombectomy - lower extremity artery"
    return "Arterial Interventions", "Arterial thrombolysis - initial" if is_lysis else "Arterial mechanical thrombectomy", desc, "Thrombolysis/Thrombectomy"


def procedure_events(source: dict[str, Any] | sqlite3.Row, phrases: list[str]) -> list[ProcedureEvent]:
    events: list[ProcedureEvent] = []
    source_text = _positive_source_text(source, phrases)

    biliary_dilation_phrases = _combine_event_phrases(phrases, ACTION_PATTERNS["biliary_stricture_dilation"])
    if biliary_dilation_phrases:
        _append_event(
            events,
            ProcedureEvent(
                key="biliary_stricture_dilation",
                label="Biliary stricture dilation",
                phrases=biliary_dilation_phrases,
                target_area="Biliary Interventions",
                target_type="Biliary stricture dilation",
                target_description="Biliary stricture dilation",
                target_def_category="GI/biliary Intervention; Other",
                reason="Explicit event: biliary stricture dilation/cholangioplasty.",
                score=0.96,
            ),
        )

    stent_graft_phrases = _combine_event_phrases(phrases, ACTION_PATTERNS["stent_graft"])
    if stent_graft_phrases:
        area, typ, description, def_cat = _vascular_action_target(source_text, "stent_graft")
        _append_event(
            events,
            ProcedureEvent(
                key=f"stent_graft_{canonical_key(description)}",
                label=description,
                phrases=stent_graft_phrases,
                target_area=area,
                target_type=typ,
                target_description=description,
                target_def_category=def_cat,
                reason="Explicit event: covered stent/stent graft.",
                score=0.96,
            ),
        )

    pta_phrases = tuple(
        phrase
        for phrase in _combine_event_phrases(phrases, ACTION_PATTERNS["pta"])
        if not re.search(r"\b(?:biliary|bile duct|cholangioplasty|bilioplasty|ureteroplasty|gastrointestinal|gi stricture)\b", normalize_match_text(phrase), re.I)
    )
    if pta_phrases:
        area, typ, description, def_cat = _vascular_action_target(source_text, "PTA")
        _append_event(
            events,
            ProcedureEvent(
                key=f"pta_{canonical_key(description)}",
                label=description,
                phrases=pta_phrases,
                target_area=area,
                target_type=typ,
                target_description=description,
                target_def_category=def_cat,
                reason="Explicit event: angioplasty/PTA.",
                score=0.95,
            ),
        )

    generic_stent_phrases = tuple(
        phrase
        for phrase in _combine_event_phrases(phrases, ACTION_PATTERNS["stent"])
        if phrase not in stent_graft_phrases
        and not re.search(r"\b(?:biliary|ureteral|ureter|nephroureteral|double j|jj stent|airway|tracheal|bronchial)\b", normalize_match_text(phrase), re.I)
    )
    if generic_stent_phrases:
        area, typ, description, def_cat = _vascular_action_target(source_text, "stent")
        _append_event(
            events,
            ProcedureEvent(
                key=f"stent_{canonical_key(description)}",
                label=description,
                phrases=generic_stent_phrases,
                target_area=area,
                target_type=typ,
                target_description=description,
                target_def_category=def_cat,
                reason="Explicit event: vascular stent placement.",
                score=0.95,
            ),
        )

    atherectomy_phrases = _combine_event_phrases(phrases, ACTION_PATTERNS["atherectomy"])
    if atherectomy_phrases:
        if _has_any(source_text, r"\b(?:aorta|aortic)\b"):
            description = "Atherectomy - aorta"
        elif _has_any(source_text, r"\b(?:subclavian artery|axillary artery|brachial artery|radial artery|ulnar artery|upper extremity artery)\b"):
            description = "Atherectomy - upper extremity artery"
        elif _has_any(source_text, r"\b(?:celiac|hepatic artery|splenic artery|sma|ima|renal artery|mesenteric|visceral)\b"):
            description = "Atherectomy - visceral artery"
        else:
            description = "Atherectomy - lower extremity artery"
        _append_event(
            events,
            ProcedureEvent(
                key=f"atherectomy_{canonical_key(description)}",
                label=description,
                phrases=atherectomy_phrases,
                target_area="Arterial Interventions",
                target_type="Arterial atherectomy",
                target_description=description,
                target_def_category="Arterial PTA or Stent",
                reason="Explicit event: atherectomy.",
                score=0.95,
            ),
        )

    for thrombus_action in ("thrombolysis", "thrombectomy"):
        thrombus_phrases = _combine_event_phrases(phrases, ACTION_PATTERNS[thrombus_action])
        if thrombus_phrases:
            area, typ, description, def_cat = _thrombus_target(source_text, thrombus_action)
            _append_event(
                events,
                ProcedureEvent(
                    key=f"{thrombus_action}_{canonical_key(description)}",
                    label=description,
                    phrases=thrombus_phrases,
                    target_area=area,
                    target_type=typ,
                    target_description=description,
                    target_def_category=def_cat,
                    reason=f"Explicit event: {thrombus_action}.",
                    score=0.95,
                ),
            )

    biliary_stent_pattern = (
        r"\bbiliary\b.*\bstent\b.*\b(?:placement|placed|conversion|internal|plastic|metal)\b|"
        r"\bconversion\b.*\bbiliary\b.*\bstent\b|"
        r"\binternal plastic stents?\b|"
        r"\bmetal cbd stent\b"
    )
    biliary_stent_phrases = _combine_event_phrases(phrases, biliary_stent_pattern)
    if biliary_stent_phrases:
        events.append(
            ProcedureEvent(
                key="biliary_stent_placement",
                label="Biliary stent placement",
                phrases=biliary_stent_phrases,
                target_area="Biliary Interventions",
                target_type="Biliary stent placement",
                target_description="Biliary stent placement (metal or plastic)",
                target_def_category="GI/biliary Intervention; Other",
                reason="Explicit event: biliary stent placement/conversion.",
                score=0.97,
            )
        )

    biliary_exchange_pattern = (
        r"\bbiliary\b.*\b(?:drain|tube|catheter)\b.*\b(?:exchange|exchanged|internalization|internalized)\b|"
        r"\b(?:exchange|exchanged|internalization|internalized)\b.*\bbiliary\b.*\b(?:drain|tube|catheter)\b"
    )
    biliary_exchange_phrases = _combine_event_phrases(phrases, biliary_exchange_pattern)
    if biliary_exchange_phrases:
        target_description = (
            "Biliary tube exchange w/internalization"
            if re.search(r"\binternaliz", " ".join(normalize_match_text(p) for p in biliary_exchange_phrases))
            else "Biliary tube exchange"
        )
        target_def = "GI/biliary Intervention; Other" if "internalization" in target_description.lower() else "Catheter exchange; Other"
        events.append(
            ProcedureEvent(
                key="biliary_tube_exchange",
                label="Biliary tube exchange/internalization",
                phrases=biliary_exchange_phrases,
                target_area="Biliary Interventions",
                target_type="Biliary tube maintenance",
                target_description=target_description,
                target_def_category=target_def,
                reason="Explicit event: biliary drain/tube exchange or internalization.",
                score=0.95,
            )
        )

    radioembolization_pattern = r"\b(?:hepatic\s+)?radioembolization\b|\by-?90\b|\byttrium\b|\bradioisotope administration\b|\btherasphere\b|\bsir-?spheres\b"
    radioembolization_phrases = _combine_event_phrases(phrases, radioembolization_pattern)
    if radioembolization_phrases:
        events.append(
            ProcedureEvent(
                key="tumor_radioembolization",
                label="Tumor radioembolization",
                phrases=radioembolization_phrases,
                target_area="Arterial Interventions",
                target_type="Arterial embolization",
                target_description="Embolization of tumor - radioembolization",
                target_def_category="Embolization",
                reason="Explicit event: radioembolization/Y90.",
                score=0.98,
            )
        )

    embolization_phrases = _combine_event_phrases(phrases, r"\b(?:embolization|embolized|embolize|transarterial embolization)\b")
    if embolization_phrases and not radioembolization_phrases:
        if re.search(r"\b(?:portal vein embolization|pve)\b", source_text):
            target_area = "Portal Interventions"
            target_type = "Portal vein embolization"
            target_description = "Portal vein embolization"
            target_def = "Embolization"
            reason = "Explicit event: portal vein embolization."
        elif re.search(r"\b(?:brto|parto|carto|varices|varix)\b", source_text):
            target_area = "Portal Interventions"
            target_type = "BRTO (balloon-occluded retrograde transvenous obliteration)"
            target_description = "BRTO PARTO CARTO"
            target_def = "Embolization"
            reason = "Explicit event: portal variceal embolization/obliteration."
        elif re.search(r"\b(?:pulmonary artery|pulmonary arterial)\b", source_text):
            target_area = "Pulmonary Arterial Interventions"
            target_type = "Pulmonary artery embolization"
            target_description = "Pulmonary artery embolization"
            target_def = "Embolization"
            reason = "Explicit event: pulmonary artery embolization."
        elif re.search(r"\b(?:gonadal vein|varicocele)\b", source_text):
            target_area = "Venous Interventions"
            target_type = "Venous embolization"
            target_description = "Venous embolization - gonadal vein"
            target_def = "Embolization"
            reason = "Explicit event: gonadal vein embolization."
        elif re.search(r"\b(?:hypogastric vein|internal iliac vein|pelvic congestion)\b", source_text):
            target_area = "Venous Interventions"
            target_type = "Venous embolization"
            target_description = "Venous embolization - hypogastric vein"
            target_def = "Embolization"
            reason = "Explicit event: hypogastric/internal iliac vein embolization."
        elif re.search(r"\b(?:venous malformation|low flow|sclerotherapy).*\b(?:embolization|embolized|sclerotherapy)|\b(?:embolization|embolized|sclerotherapy).*\b(?:venous malformation|low flow)\b", source_text):
            target_area = "Venous Interventions"
            target_type = "Vascular malformation embolization"
            target_description = "Low flow vascular malformation/venous embolization"
            target_def = "Embolization"
            reason = "Explicit event: low-flow vascular malformation embolization/sclerotherapy."
        elif re.search(r"\b(?:avm|arteriovenous malformation|high flow)\b", source_text):
            target_area = "Venous Interventions"
            target_type = "Vascular malformation embolization"
            target_description = "High flow vascular malformation/AVM embolization"
            target_def = "Embolization"
            reason = "Explicit event: high-flow vascular malformation/AVM embolization."
        elif re.search(r"\b(?:lymphatic malformation)\b", source_text):
            target_area = "Venous Interventions"
            target_type = "Vascular malformation embolization"
            target_description = "Lymphatic malformation embolization"
            target_def = "Embolization"
            reason = "Explicit event: lymphatic malformation embolization."
        elif re.search(r"\b(?:thoracic duct|lymphatic leak|chylous leak)\b", source_text):
            target_area = "Lymphatic Interventions and Other"
            target_type = "Lymphatic embolization thoracic duct"
            target_description = "Lymphatic embolization thoracic duct"
            target_def = "Lymphatic Intervention"
            reason = "Explicit event: thoracic duct/lymphatic embolization."
        elif re.search(r"\b(?:uterine|uterus|fibroid|uae)\b", source_text):
            target_area = "Arterial Interventions"
            target_type = "Arterial embolization"
            target_description = "Uterine artery embolization"
            target_def = "Embolization"
            reason = "Explicit event: uterine artery embolization."
        elif re.search(r"\b(?:prostate|prostatic|pae)\b", source_text):
            target_area = "Arterial Interventions"
            target_type = "Arterial embolization"
            target_description = "Prostate artery embolization"
            target_def = "Embolization"
            reason = "Explicit event: prostate artery embolization."
        elif re.search(r"\b(?:bronchial|hemoptysis)\b", source_text):
            target_area = "Arterial Interventions"
            target_type = "Arterial embolization"
            target_description = "Bronchial artery embolization"
            target_def = "Embolization"
            reason = "Explicit event: bronchial artery embolization."
        else:
            target_area = "Arterial Interventions"
            target_type = "Arterial embolization"
            target_description = "Other arterial embolization"
            target_def = "Embolization"
            reason = "Explicit event: arterial embolization."
        events.append(
            ProcedureEvent(
                key=canonical_key(target_description),
                label=target_description,
                phrases=embolization_phrases,
                target_area=target_area,
                target_type=target_type,
                target_description=target_description,
                target_def_category=target_def,
                reason=reason,
                score=0.92,
            )
        )

    drain_placement_pattern = (
        r"\b(?:transvaginal|pelvic|abscess|fluid collection|peritoneal|retroperitoneal|visceral|organ|superficial|extremity|pleural|chest)\b"
        r".*\b(?:drainage catheter|drain|tube)\b.*\b(?:placement|placed|insertion|inserted)\b|"
        r"\b(?:drainage catheter|drain|tube)\b.*\b(?:placement|placed|insertion|inserted)\b"
    )
    drain_phrases = _combine_event_phrases(phrases, drain_placement_pattern)
    if drain_phrases and re.search(r"\b(?:drainage|drain|abscess|fluid collection)\b", source_text):
        if re.search(r"\b(?:pelvic|transvaginal|intraperitoneal|peritoneal)\b", source_text):
            target_description = "Drainage - intraperitoneal tube"
            target_area = "Drainage Procedures"
            target_def = "Drain Placement; Image guided bx/drainage"
        elif re.search(r"\b(?:retroperitoneal)\b", source_text):
            target_description = "Drainage - retroperitoneal tube"
            target_area = "Body Procedures"
            target_def = "Drain Placement; Image guided bx/drainage"
        elif re.search(r"\b(?:chest|pleural)\b", source_text):
            target_description = "Drainage - chest tube"
            target_area = "Drainage Procedures"
            target_def = "Drain Placement; Image guided bx/drainage"
        elif re.search(r"\b(?:superficial|extremity)\b", source_text):
            target_description = "Drainage - superficial/extremity tube"
            target_area = "Body Procedures"
            target_def = "Drain Placement; Image guided bx/drainage"
        else:
            target_description = "Drainage - visceral/organ tube"
            target_area = "Drainage Procedures"
            target_def = "Drain Placement; Image guided bx/drainage"
        events.append(
            ProcedureEvent(
                key="drainage_catheter_placement",
                label="Drainage catheter placement",
                phrases=drain_phrases,
                target_area=target_area,
                target_type="Drainage tube placement",
                target_description=target_description,
                target_def_category=target_def,
                reason="Explicit event: drainage catheter placement.",
                score=0.94,
            )
        )

    biopsy_phrases = _combine_event_phrases(phrases, r"\b(?:biopsy|core needle|fine needle|fna)\b")
    if biopsy_phrases:
        biopsy_sites = [
            (r"\badrenal\b", "Biopsy", "Biopsy abdominal/retroperitoneal", "Biopsy - adrenal"),
            (r"\b(?:lymph node|nodal|node biopsy)\b", "Biopsy", "Biopsy lymph node", "Biopsy - lymph node"),
            (r"\b(?:cervical|neck).*\b(?:node|nodal|lymph)\b|\b(?:node|nodal|lymph).*\b(?:cervical|neck)\b", "Biopsy", "Biopsy cervical nodal", "Biopsy - cervical nodal"),
            (r"\bthyroid\b", "Biopsy", "Biopsy thyroid", "Biopsy - thyroid"),
            (r"\b(?:lung|pulmonary)\b", "Biopsy", "Biopsy thoracic", "Biopsy - lung"),
            (r"\bmediastin", "Biopsy", "Biopsy thoracic", "Biopsy - mediastinum"),
            (r"\b(?:soft tissue|subcutaneous|superficial|muscle|fat pad)\b", "Biopsy", "Biopsy soft tissue", "Biopsy - soft tissue"),
            (r"\b(?:bone marrow|marrow)\b", "Biopsy", "Biopsy musculoskeletal", "Biopsy - bone marrow"),
            (r"\bjoint\b", "Biopsy", "Biopsy musculoskeletal", "Biopsy - joint"),
            (r"\b(?:spleen|splenic)\b", "Biopsy", "Biopsy abdominal/retroperitoneal", "Biopsy - spleen"),
            (r"\b(?:renal|kidney|nephro|ureter|bladder|prostate|testicular|genitourinary|gu)\b", "Biopsy", "Biopsy abdominal/retroperitoneal", "Biopsy - genitourinary"),
            (r"\b(?:biliary|bile duct|gallbladder)\b", "Biopsy", "Biopsy abdominal/retroperitoneal", "Biopsy - biliary"),
        ]
        for pattern, area, typ, description in biopsy_sites:
            if re.search(pattern, source_text):
                events.append(
                    ProcedureEvent(
                        key=f"biopsy_{canonical_key(description)}",
                        label=description,
                        phrases=biopsy_phrases,
                        target_area=area,
                        target_type=typ,
                        target_description=description,
                        target_def_category="Biopsy; Image guided bx/drainage",
                        reason=f"Explicit event: {description.lower()} with site evidence.",
                        score=0.92,
                    )
                )
                break

    nephroureteral_pattern = (
        r"\bnephroureteral\b.*\b(?:stent|tube)\b.*\b(?:exchange|change|upsize|upsizing|exchanged|changed)\b|"
        r"\b(?:exchange|change|upsize|upsizing|exchanged|changed)\b.*\bnephroureteral\b.*\b(?:stent|tube)\b"
    )
    nephroureteral_phrases = _combine_event_phrases(phrases, nephroureteral_pattern)
    if nephroureteral_phrases:
        events.append(
            ProcedureEvent(
                key="nephroureteral_stent_change",
                label="Nephroureteral stent change",
                phrases=nephroureteral_phrases,
                target_area="GU Intervention",
                target_type="GU tube/stent exchange",
                target_description="Nephroureteral stent change",
                target_def_category="Catheter exchange",
                reason="Semantic alias: nephroureteral stent/tube exchange.",
                score=0.97,
            )
        )

    nephrostomy_placement_phrases = _combine_event_phrases(
        phrases,
        r"\b(?:nephrostomy|pcn)\b.*\b(?:placement|placed|insertion|inserted|creation|new access)\b|"
        r"\b(?:placement|placed|insertion|inserted|creation|new access)\b.*\b(?:nephrostomy|pcn)\b",
    )
    if nephrostomy_placement_phrases and not re.search(r"\bnephroureteral\b", " ".join(normalize_match_text(p) for p in nephrostomy_placement_phrases)):
        _append_event(
            events,
            ProcedureEvent(
                key="nephrostomy_tube_placement",
                label="Nephrostomy tube placement",
                phrases=nephrostomy_placement_phrases,
                target_area="GU Intervention",
                target_type="Nephrostomy tube placement",
                target_description="Nephrostomy tube placement",
                target_def_category="Primary Nephrostomy",
                reason="Explicit event: nephrostomy tube placement.",
                score=0.96,
            ),
        )

    nephrostomy_exchange_phrases = _combine_event_phrases(
        phrases,
        r"\b(?:nephrostomy|pcn)\b.*\b(?:exchange|change|replacement|replaced|upsiz|downsiz)\b|"
        r"\b(?:exchange|change|replacement|replaced|upsiz|downsiz)\b.*\b(?:nephrostomy|pcn)\b",
    )
    if nephrostomy_exchange_phrases and not nephroureteral_phrases:
        _append_event(
            events,
            ProcedureEvent(
                key="nephrostomy_change",
                label="Nephrostomy change",
                phrases=nephrostomy_exchange_phrases,
                target_area="GU Intervention",
                target_type="GU tube/stent exchange",
                target_description="Nephrostomy change",
                target_def_category="Catheter exchange",
                reason="Explicit event: nephrostomy exchange/change.",
                score=0.96,
            ),
        )

    double_j_exchange_phrases = _combine_event_phrases(
        phrases,
        r"\b(?:double j|double-j|jj|ureteral stent)\b.*\b(?:exchange|change|replacement|replaced)\b|"
        r"\b(?:exchange|change|replacement|replaced)\b.*\b(?:double j|double-j|jj|ureteral stent)\b",
    )
    if double_j_exchange_phrases:
        local_double_j = [
            normalize_match_text(phrase)
            for phrase in double_j_exchange_phrases
            if re.search(r"\b(?:double j|double-j|jj|ureteral stent)\b", normalize_match_text(phrase), re.I)
        ]
        transrenal = bool(local_double_j) and all(
            re.search(r"\b(?:antegrade|transrenal|nephrostomy)\b", phrase, re.I) for phrase in local_double_j
        )
        description = "Double J exchange transrenal" if transrenal else "Double J exchange transurethral"
        _append_event(
            events,
            ProcedureEvent(
                key=canonical_key(description),
                label=description,
                phrases=double_j_exchange_phrases,
                target_area="GU Intervention",
                target_type="GU tube/stent exchange",
                target_description=description,
                target_def_category="GU Intervention",
                reason="Explicit event: double-J/ureteral stent exchange.",
                score=0.95,
            ),
        )

    gu_removal_phrases = _combine_event_phrases(
        phrases,
        r"\b(?:nephrostomy|nephroureteral|double j|double-j|jj|ureteral stent)\b.*\b(?:removal|removed|retrieval|pulled)\b|"
        r"\b(?:removal|removed|retrieval|pulled)\b.*\b(?:nephrostomy|nephroureteral|double j|double-j|jj|ureteral stent)\b",
    )
    if gu_removal_phrases:
        if re.search(r"\bnephroureteral\b", source_text):
            description = "Nephroureteral stent removal"
        elif re.search(r"\b(?:double j|double-j|jj|ureteral stent)\b", source_text):
            description = "Ureter double J removal transrenal" if re.search(r"\b(?:antegrade|transrenal|nephrostomy)\b", source_text) else "Ureter double J removal transurethral"
        else:
            description = "Nephrostomy removal"
        _append_event(
            events,
            ProcedureEvent(
                key=canonical_key(description),
                label=description,
                phrases=gu_removal_phrases,
                target_area="GU Intervention",
                target_type="GU tube/stent removal",
                target_description=description,
                target_def_category="Removal",
                reason="Explicit event: GU tube/stent removal.",
                score=0.95,
            ),
        )

    ureteroplasty_pattern = r"\b(?:upj\s+)?ureteroplasty\b"
    ureteroplasty_phrases = _combine_event_phrases(phrases, ureteroplasty_pattern)
    if ureteroplasty_phrases:
        events.append(
            ProcedureEvent(
                key="gu_stricture_dilation",
                label="GU stricture dilation",
                phrases=ureteroplasty_phrases,
                target_area="GU Intervention",
                target_type="GU stricture dilation",
                reason="Semantic alias: ureteroplasty implies ureteral/UPJ stricture dilation.",
                score=0.93,
            )
        )

    if not nephroureteral_phrases and re.search(r"\birtubechgl\b", source_text) and re.search(
        r"\b(?:nephrostomy|nephroureteral|ureter|genitourinary|gu)\b", source_text
    ):
        matched = tuple(
            phrase
            for phrase in phrases
            if re.search(r"\b(?:genitourinary|nephrostomy|nephroureteral|ureter).*\b(?:exchange|change)\b", normalize_match_text(phrase), re.I)
        )
        events.append(
            ProcedureEvent(
                key="gu_tube_stent_exchange",
                label="GU tube/stent exchange",
                phrases=matched or tuple(phrases[:1]),
                target_area="GU Intervention",
                target_type="GU tube/stent exchange",
                reason="Rule match: GU tube/stent exchange.",
                score=0.88,
            )
        )

    return events


def candidate_from_event(event: ProcedureEvent, targets: list[CandidateTarget]) -> dict[str, Any]:
    target = _find_target(
        targets,
        event.target_area,
        event.target_type,
        event.target_description,
        event.target_def_category,
    )
    confidence = confidence_for_score(event.score)
    matched = list(event.phrases)
    return {
        "case_class": target.case_class,
        "acgme_code": target.acgme_code,
        "area": target.area,
        "type": target.type,
        "acgme_description": target.acgme_description,
        "acgme_def_category": target.acgme_def_category,
        "keyword": target.keyword,
        "component_label": event.key,
        "score": round(event.score, 4),
        "confidence": confidence,
        "default_checked": int(confidence == "high"),
        "match_reason": event.reason,
        "matched_phrases": matched,
        "evidence_snippet": " | ".join(matched[:3]),
        "event_key": event.key,
        "event_label": event.label,
        "source_kind": "event_alias",
        "algorithm_version": ALGORITHM_VERSION,
    }


def _prefer_specific_targets(candidates: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selected = dict(candidates)
    specific_type_keys = {
        (item["case_class"], item["area"], item["type"])
        for item in selected.values()
        if item.get("acgme_description")
    }
    event_specific_type_keys = {
        (item["case_class"], item["area"], item["type"])
        for item in selected.values()
        if item.get("source_kind") == "event_alias" and item.get("acgme_description")
    }
    for key, item in list(selected.items()):
        if not item.get("acgme_description") and (item["case_class"], item["area"], item["type"]) in specific_type_keys:
            del selected[key]
            continue
        if (
            item.get("source_kind") != "event_alias"
            and (item["case_class"], item["area"], item["type"]) in event_specific_type_keys
        ):
            del selected[key]
    return selected


def build_match_candidates(
    source: dict[str, Any] | sqlite3.Row,
    limit: int = 12,
    learning_signals: list[sqlite3.Row] | None = None,
) -> list[dict[str, Any]]:
    rules, cached_rule_targets = cached_rules_and_targets()
    phrases = evidence_phrases(source)
    positive_phrases = _positive_phrases(phrases)
    positive_domain_text = _positive_source_text(source, phrases)
    phrase_features = prepared_phrases(positive_phrases)
    source_token_union: set[str] = set()
    for _, tokens, _ in phrase_features:
        source_token_union.update(tokens)
    source_dict = dict(source) if isinstance(source, sqlite3.Row) else source
    candidates: dict[str, dict[str, Any]] = {}
    exact_rule_ids = {
        rule.rule_id
        for rule in rules
        if rule.active and rule.action == "generate" and rule_matches(rule, source_dict)
    }
    official = list(cached_official_targets())
    targets = list(cached_rule_targets) + official
    for event in procedure_events(source, phrases):
        candidate = candidate_from_event(event, official)
        candidates[target_key(_target_from_parts(
            candidate["area"],
            candidate["type"],
            candidate.get("acgme_description", ""),
            candidate.get("acgme_def_category", ""),
            candidate.get("acgme_code", ""),
        ))] = candidate
    for target in targets:
        exact_rule = target.source_kind == "rule" and target.rule_id in exact_rule_ids
        if not exact_rule and not (source_token_union & set(cached_target_tokens(target))):
            continue
        if not _has_domain_support(target, positive_domain_text):
            continue
        score, matched, reason = score_target(target, phrase_features, exact_rule)
        if score < 0.38:
            continue
        shared_tokens = match_tokens(" ".join(matched)) & set(cached_target_tokens(target))
        if not exact_rule and _is_generic_device_match(shared_tokens):
            continue
        if not _target_requires_specific_evidence(target, positive_domain_text):
            continue
        key = target_key(target)
        confidence = confidence_for_score(score)
        can_default_check = (
            exact_rule
            and confidence == "high"
            and _rule_can_default_check(target, positive_domain_text, shared_tokens)
        )
        candidate = {
            "case_class": target.case_class,
            "acgme_code": target.acgme_code,
            "area": target.area,
            "type": target.type,
            "acgme_description": target.acgme_description,
            "acgme_def_category": target.acgme_def_category,
            "keyword": target.keyword,
            "component_label": target.component_label,
            "score": round(score, 4),
            "confidence": confidence,
            "default_checked": int(can_default_check),
            "match_reason": reason if not exact_rule else f"Rule match: {target.rule_name or target.type}. {reason}",
            "matched_phrases": matched,
            "evidence_snippet": " | ".join(matched[:3]),
            "event_key": "",
            "event_label": "",
            "source_kind": target.source_kind,
            "algorithm_version": ALGORITHM_VERSION,
        }
        existing = candidates.get(key)
        if existing is None or candidate["score"] > existing["score"] or existing["source_kind"] == "dropdown":
            candidates[key] = candidate
    candidates = _prefer_specific_targets(candidates)
    ranked = sorted(candidates.values(), key=lambda item: item["score"], reverse=True)
    return apply_learning_signals(source, ranked, learning_signals)[:limit]


def candidate_key(source_case_id: int, candidate: dict[str, Any]) -> str:
    return "|".join(
        [
            str(source_case_id),
            candidate["case_class"],
            candidate.get("acgme_code", ""),
            candidate["area"],
            candidate["type"],
            candidate.get("acgme_description", ""),
            candidate.get("acgme_def_category", ""),
        ]
    )


def store_match_candidates(conn: sqlite3.Connection, source_case_id: int, candidates: list[dict[str, Any]]) -> int:
    now = utc_now()
    count = 0
    for candidate in candidates:
        params = {
            **candidate,
            "source_case_id": source_case_id,
            "candidate_key": candidate_key(source_case_id, candidate),
            "matched_phrases_json": json.dumps(candidate.get("matched_phrases", []), ensure_ascii=False),
            "acgme_code": candidate.get("acgme_code", ""),
            "event_key": candidate.get("event_key", ""),
            "event_label": candidate.get("event_label", ""),
            "created_at": now,
            "updated_at": now,
        }
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO source_match_candidates(
              source_case_id, candidate_key, case_class, acgme_code, area, type, acgme_description, acgme_def_category,
              keyword, component_label, score, confidence, default_checked, match_reason, matched_phrases_json,
              evidence_snippet, event_key, event_label, source_kind, algorithm_version, created_at, updated_at
            )
            VALUES (
              :source_case_id, :candidate_key, :case_class, :acgme_code, :area, :type, :acgme_description, :acgme_def_category,
              :keyword, :component_label, :score, :confidence, :default_checked, :match_reason, :matched_phrases_json,
              :evidence_snippet, :event_key, :event_label, :source_kind, :algorithm_version, :created_at, :updated_at
            )
            """,
            params,
        )
        count += int(cur.rowcount == 1)
    return count


def ensure_candidates_for_source(conn: sqlite3.Connection, source: sqlite3.Row) -> int:
    current = conn.execute(
        """
        SELECT COUNT(*)
        FROM source_match_candidates
        WHERE source_case_id = ?
          AND algorithm_version = ?
        """,
        (source["id"], ALGORITHM_VERSION),
    ).fetchone()[0]
    if current:
        return 0
    conn.execute(
        """
        DELETE FROM source_match_candidates
        WHERE source_case_id = ?
          AND user_status = 'pending'
        """,
        (source["id"],),
    )
    return store_match_candidates(conn, int(source["id"]), build_match_candidates(source, learning_signals=load_learning_signals(conn)))


def load_candidates(conn: sqlite3.Connection, source_case_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT *
        FROM source_match_candidates
        WHERE source_case_id = ?
          AND algorithm_version = ?
          AND user_status IN ('pending', 'accepted', 'rejected')
        ORDER BY COALESCE(user_checked, default_checked) DESC, score DESC, id
        """,
        (source_case_id, ALGORITHM_VERSION),
    ).fetchall()


def search_acgme_targets(query: str, limit: int = 20) -> list[dict[str, Any]]:
    query_tokens = match_tokens(query)
    if not query_tokens:
        return []
    results: list[dict[str, Any]] = []
    query_norm = normalize_match_text(query)
    for target in load_acgme_targets():
        tokens = match_tokens(target_text(target))
        shared = query_tokens & tokens
        target_norm = normalize_match_text(target_text(target))
        direct = bool(query_norm and query_norm in target_norm)
        if not shared and not direct:
            continue
        score = len(shared) / max(len(query_tokens | tokens), 1)
        if direct:
            score += 0.35
        results.append(
            {
                "label": " / ".join(
                    part
                    for part in [
                        target.acgme_code,
                        target.acgme_description,
                        f"{target.area} / {target.type}",
                        f"Def Cat: {target.acgme_def_category}" if target.acgme_def_category else "",
                    ]
                    if part
                ),
                "case_class": target.case_class,
                "acgme_code": target.acgme_code,
                "area": target.area,
                "type": target.type,
                "acgme_description": target.acgme_description,
                "acgme_def_category": target.acgme_def_category,
                "keyword": target.keyword,
                "component_label": "manual_search",
                "score": score,
            }
        )
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:limit]


def add_manual_candidate(conn: sqlite3.Connection, source_case_id: int, target: dict[str, Any]) -> int:
    candidate = {
        "case_class": target["case_class"],
        "acgme_code": target.get("acgme_code", ""),
        "area": target["area"],
        "type": target["type"],
        "acgme_description": target.get("acgme_description", ""),
        "acgme_def_category": target.get("acgme_def_category", ""),
        "keyword": target.get("keyword", ""),
        "component_label": target.get("component_label") or "manual_search",
        "score": 1.0,
        "confidence": "manual",
        "default_checked": 1,
        "match_reason": "Added from ACGME search.",
        "matched_phrases": [],
        "evidence_snippet": "",
        "event_key": "manual_search",
        "event_label": "Manual search",
        "source_kind": "manual_search",
        "algorithm_version": ALGORITHM_VERSION,
    }
    store_match_candidates(conn, source_case_id, [candidate])
    row = conn.execute(
        "SELECT id FROM source_match_candidates WHERE candidate_key = ?",
        (candidate_key(source_case_id, candidate),),
    ).fetchone()
    return int(row["id"])


def add_manual_candidates(conn: sqlite3.Connection, source_case_id: int, targets: list[dict[str, Any]]) -> set[int]:
    return {add_manual_candidate(conn, source_case_id, target) for target in targets}


def candidate_to_entry(source: sqlite3.Row, candidate: sqlite3.Row, mapping_rules_file_hash: str = "candidate_v1") -> dict[str, Any]:
    from .review_queue import source_row_to_mapping_source

    mapping_source = source_row_to_mapping_source(source)
    derived = mapping_source["derived"]
    component_label = candidate["component_label"] or "dominant_procedure"
    dedupe_key = "|".join(
        [
            source["accession_number"],
            derived["case_date"],
            candidate["case_class"],
            candidate["acgme_code"] or "" if "acgme_code" in candidate.keys() else "",
            candidate["area"],
            candidate["type"],
            candidate["acgme_description"] or "",
            component_label,
        ]
    )
    return {
        "dedupe_key": dedupe_key,
        "component_label": component_label,
        "case_id": derived["case_id"],
        "case_date": derived["case_date"],
        "case_year": derived["case_year"],
        "role": derived["role"],
        "site": derived["site"],
        "patient_type": derived["patient_type"],
        "case_class": candidate["case_class"],
        "acgme_code": candidate["acgme_code"] or "" if "acgme_code" in candidate.keys() else "",
        "area": candidate["area"],
        "type": candidate["type"],
        "acgme_description": candidate["acgme_description"] or "",
        "acgme_def_category": candidate["acgme_def_category"] or "",
        "keyword": candidate["keyword"] or "",
        "comments": candidate["match_reason"] or "",
        "mapping_rule_id": f"candidate:{candidate['id']}",
        "mapping_rule_version": "1",
        "mapping_rules_file_hash": mapping_rules_file_hash,
        "mapping_rule_name": f"Candidate: {candidate['source_kind']}",
        "mapping_confidence": "high" if candidate["confidence"] in {"high", "manual"} else candidate["confidence"],
        "role_confidence": derived["role_confidence"],
        "compound_flag": 0,
        "review_status": "approved",
    }


def approve_candidate_review(conn: sqlite3.Connection, source_case_id: int, checked_candidate_ids: set[int]) -> int:
    from .importer import insert_generated_entries

    source = conn.execute("SELECT * FROM source_cases WHERE id = ?", (source_case_id,)).fetchone()
    if not source:
        raise ValueError(f"Source case not found: {source_case_id}")
    candidates = load_candidates(conn, source_case_id)
    accepted = [candidate for candidate in candidates if int(candidate["id"]) in checked_candidate_ids]
    now = utc_now()
    for candidate in candidates:
        checked = int(candidate["id"]) in checked_candidate_ids
        weight = 0.3 if checked and candidate["source_kind"] != "manual_search" else (1.0 if checked else 0.0)
        conn.execute(
            """
            UPDATE source_match_candidates
            SET user_checked = ?, user_status = ?, learning_weight = ?, updated_at = ?
            WHERE id = ?
            """,
            (int(checked), "accepted" if checked else "rejected", weight, now, candidate["id"]),
        )
        signal_type = "accepted_guess" if checked and candidate["source_kind"] != "manual_search" else ("explicit_add" if checked else "explicit_reject")
        signal_weight = weight if checked else -0.2
        conn.execute(
            """
            INSERT INTO mapping_learning_signals(
              source_case_id, candidate_id, signal_type, learning_weight, exam_code, procedure_text,
              case_class, area, type, acgme_description, acgme_def_category, evidence_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_case_id,
                candidate["id"],
                signal_type,
                signal_weight,
                source["exam_code"] or "",
                source["procedure_text"] or "",
                candidate["case_class"],
                candidate["area"],
                candidate["type"],
                candidate["acgme_description"] or "",
                candidate["acgme_def_category"] or "",
                candidate["matched_phrases_json"] or "[]",
                now,
            ),
        )
    conn.execute(
        """
        UPDATE generated_entries
        SET review_status = 'skipped', updated_at = ?
        WHERE source_case_id = ?
          AND review_status IN ('new_high_confidence', 'needs_review')
        """,
        (now, source_case_id),
    )
    entries = [candidate_to_entry(source, candidate) for candidate in accepted]
    if len(entries) > 1:
        for entry in entries:
            entry["compound_flag"] = 1
    inserted = insert_generated_entries(conn, source_case_id, entries)
    for entry in entries:
        conn.execute(
            """
            UPDATE generated_entries
            SET review_status = 'approved', mapping_confidence = ?, compound_flag = ?, updated_at = ?
            WHERE dedupe_key = ?
            """,
            (entry["mapping_confidence"], entry["compound_flag"], now, entry["dedupe_key"]),
        )
    conn.execute(
        "UPDATE source_cases SET source_mapping_status = ? WHERE id = ?",
        ("candidate_reviewed" if entries else "candidate_reviewed_empty", source_case_id),
    )
    log_event(conn, None, "candidate_reviewed", None, None, "review_app", f"source_case_id={source_case_id}; accepted={len(entries)}")
    conn.commit()
    return inserted
