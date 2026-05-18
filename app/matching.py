from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from .utils import canonical_text


STOP_TOKENS = {
    "a",
    "an",
    "and",
    "by",
    "for",
    "from",
    "guided",
    "ir",
    "of",
    "or",
    "the",
    "to",
    "with",
    "without",
}

ALIAS_GROUPS = [
    {"ivc", "inferior", "vena", "cava"},
]

SYNONYM_GROUPS = [
    {"retrieval", "retrieve", "retrieved", "removal", "remove", "removed"},
    {"insertion", "insert", "inserted", "placement", "place", "placed", "creation", "new"},
    {"exchange", "exchanged", "change", "changed", "replacement", "replaced", "revision", "converted", "conversion"},
    {"catheter", "tube", "drain"},
    {"angioplasty", "pta", "plasty", "dilation", "dilatation", "venoplasty"},
    {"stent", "stenting", "relining"},
    {"stentgraft", "endograft", "endoprosthesis"},
    {"thrombolysis", "lysis", "tpa", "alteplase", "cdt", "ekos"},
    {"thrombectomy", "embolectomy", "aspiration", "suction"},
    {"embolization", "embolisation", "embolized", "embolize", "occlusion", "devascularization"},
    {"biopsy", "bx", "fna", "sampling"},
    {"ablation", "rfa", "mwa", "cryoablation", "cryotherapy", "sclerotherapy"},
    {"hypogastric", "internal", "iliac"},
    {"gastrostomy", "g", "peg"},
    {"gastrojejunostomy", "gj"},
    {"jejunostomy", "j"},
]

TOKEN_SYNONYMS: dict[str, set[str]] = {}
for group in SYNONYM_GROUPS:
    for token in group:
        TOKEN_SYNONYMS[token] = set(group)


@dataclass(frozen=True)
class MatchSuggestion:
    rule_id: str
    rule_name: str
    case_class: str
    area: str
    type: str
    acgme_description: str
    acgme_def_category: str
    keyword: str
    component_label: str
    score: float
    confidence: str
    match_kind: str
    reason: str

    def as_entry_values(self) -> dict[str, str]:
        return {
            "case_class": self.case_class,
            "area": self.area,
            "type": self.type,
            "acgme_description": self.acgme_description,
            "acgme_def_category": self.acgme_def_category,
            "keyword": self.keyword,
            "component_label": self.component_label,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.as_entry_values(),
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "score": round(self.score, 3),
            "confidence": self.confidence,
            "match_kind": self.match_kind,
            "reason": self.reason,
        }


def normalize_match_text(value: str | None) -> str:
    text = canonical_text(value)
    if not text:
        return ""
    text = re.sub(r"\(([^)]{1,30})\)", r" \1 ", text)
    text = re.sub(r"[/_+]", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _raw_tokens(value: str | None) -> list[str]:
    return [token for token in normalize_match_text(value).split() if token and token not in STOP_TOKENS]


def match_tokens(*values: str | None) -> set[str]:
    raw: list[str] = []
    for value in values:
        raw.extend(_raw_tokens(value))
    tokens = set(raw)
    if {"inferior", "vena", "cava"} <= tokens:
        tokens.add("ivc")
    if "ivc" in tokens:
        tokens.update({"inferior", "vena", "cava"})
    expanded = set(tokens)
    for token in list(tokens):
        expanded.update(TOKEN_SYNONYMS.get(token, set()))
    return {token for token in expanded if token not in STOP_TOKENS}


def _regex_to_text(pattern: str) -> str:
    text = pattern or ""
    text = text.replace("|", " ")
    text = re.sub(r"[\^\$\.\*\+\?\[\]\(\)\{\}\\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _important(tokens: set[str]) -> set[str]:
    return {token for token in tokens if len(token) > 2 and token not in STOP_TOKENS}


def _rule_tokens(rule: Any) -> set[str]:
    return match_tokens(
        getattr(rule, "rule_name", ""),
        getattr(rule, "match_exam_code", ""),
        _regex_to_text(getattr(rule, "match_study_description_regex", "")),
        _regex_to_text(getattr(rule, "match_procedure_regex", "")),
        getattr(rule, "area", ""),
        getattr(rule, "type", ""),
        getattr(rule, "acgme_description", ""),
        getattr(rule, "acgme_def_category", ""),
    )


def _source_tokens(source: dict[str, Any]) -> set[str]:
    return match_tokens(
        source.get("exam_code", ""),
        source.get("procedure_text", ""),
        source.get("study_description", ""),
    )


def _source_text(source: dict[str, Any]) -> str:
    return normalize_match_text(
        " ".join(
            str(source.get(key) or "")
            for key in ("exam_code", "procedure_text", "study_description")
        )
    )


def _rule_text(rule: Any) -> str:
    return normalize_match_text(
        " ".join(
            [
                getattr(rule, "rule_name", ""),
                getattr(rule, "match_exam_code", ""),
                _regex_to_text(getattr(rule, "match_study_description_regex", "")),
                _regex_to_text(getattr(rule, "match_procedure_regex", "")),
                getattr(rule, "area", ""),
                getattr(rule, "type", ""),
                getattr(rule, "acgme_description", ""),
            ]
        )
    )


def _score_candidate(source: dict[str, Any], rule: Any) -> tuple[float, str, str]:
    src_tokens = _important(_source_tokens(source))
    rule_tokens = _important(_rule_tokens(rule))
    if not src_tokens or not rule_tokens:
        return 0.0, "none", "No comparable tokens."

    shared = src_tokens & rule_tokens
    exact_exam = bool(
        getattr(rule, "match_exam_code", "")
        and canonical_text(getattr(rule, "match_exam_code", "")) == canonical_text(source.get("exam_code"))
    )
    coverage = len(shared) / max(len(rule_tokens), 1)
    overlap = len(shared) / max(len(src_tokens | rule_tokens), 1)
    phrase = SequenceMatcher(None, _source_text(source), _rule_text(rule)).ratio()
    score = (coverage * 0.55) + (overlap * 0.25) + (phrase * 0.20)

    if exact_exam:
        score += 0.18
    if len(shared) < 2 and not exact_exam:
        score *= 0.45

    kind = "fuzzy"
    if exact_exam or coverage >= 0.92:
        kind = "alias"
    reason = f"Matched {', '.join(sorted(shared)[:6]) or 'exam code'}."
    return min(score, 1.0), kind, reason


def suggest_mappings(
    source: dict[str, Any],
    rules: list[Any],
    limit: int = 3,
    min_score: float = 0.42,
) -> list[MatchSuggestion]:
    suggestions: list[MatchSuggestion] = []
    for rule in rules:
        if not getattr(rule, "active", False) or getattr(rule, "action", "") != "generate":
            continue
        score, kind, reason = _score_candidate(source, rule)
        if score < min_score:
            continue
        confidence = "medium" if score >= 0.68 else "low"
        suggestions.append(
            MatchSuggestion(
                rule_id=getattr(rule, "rule_id", ""),
                rule_name=getattr(rule, "rule_name", ""),
                case_class=getattr(rule, "case_class", ""),
                area=getattr(rule, "area", ""),
                type=getattr(rule, "type", ""),
                acgme_description=getattr(rule, "acgme_description", ""),
                acgme_def_category=getattr(rule, "acgme_def_category", ""),
                keyword=getattr(rule, "keyword", ""),
                component_label=getattr(rule, "component_label", "") or "dominant_procedure",
                score=score,
                confidence=confidence,
                match_kind=kind,
                reason=reason,
            )
        )
    suggestions.sort(key=lambda item: item.score, reverse=True)
    return suggestions[:limit]


def should_generate_from_suggestion(suggestion: MatchSuggestion) -> bool:
    return suggestion.match_kind == "alias" and suggestion.score >= 0.92
