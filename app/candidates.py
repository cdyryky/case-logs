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

ALGORITHM_VERSION = "candidate_v3_acgme_csv"
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
            item["default_checked"] = int(item["confidence"] in {"high", "medium"})
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
        for phrase in phrases
        if re.search(pattern, normalize_match_text(phrase), re.I)
    ]
    return tuple(dict.fromkeys(matched))


def procedure_events(source: dict[str, Any] | sqlite3.Row, phrases: list[str]) -> list[ProcedureEvent]:
    events: list[ProcedureEvent] = []
    source_text = _source_domain_text(source, phrases)

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
        "default_checked": int(confidence in {"high", "medium"}),
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
    source_domain_text = _source_domain_text(source, phrases)
    phrase_features = prepared_phrases(phrases)
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
        if not exact_rule and not _has_domain_support(target, source_domain_text):
            continue
        score, matched, reason = score_target(target, phrase_features, exact_rule)
        if score < 0.38:
            continue
        shared_tokens = match_tokens(" ".join(matched)) & set(cached_target_tokens(target))
        if not exact_rule and _is_generic_device_match(shared_tokens):
            continue
        key = target_key(target)
        confidence = confidence_for_score(score)
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
            "default_checked": int(confidence in {"high", "medium"}),
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
          AND user_status IN ('pending', 'accepted', 'rejected')
        ORDER BY COALESCE(user_checked, default_checked) DESC, score DESC, id
        """,
        (source_case_id,),
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
