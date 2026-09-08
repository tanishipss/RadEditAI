"""
RES (Radiology Edit Score) — local scorer.

Implements PRD.md §7 / TRD.md §4.11 exactly as written there. No `radiology_task_cleaned.md`
was present in this repo, so this is a literal implementation of the TRD §4.11 summary; two
genuine ambiguities in that spec are resolved below and documented at the point they occur:

  1. Substitution cost in the weighted edit distance. TRD says insert/delete/substitute costs
     equal "the weight of the token being inserted/deleted/substituted" but does not say which
     token's weight governs a substitution (the reference token or the hypothesis token). This
     implementation uses max(weight(ref_token), weight(hyp_token)) so that replacing a critical
     word (e.g. a negation) with a low-weight word is penalized as heavily as the reverse.
  2. "Unexpected/unlabelled content in submitted adds to the numerator" (no matching reference
     field). Implemented as: field_weight=3.0 times the edit distance of that content against an
     empty reference, added to the numerator only — the denominator (sum of field weights) is
     built purely from the reference report's own fields, so hallucinated fields are pure penalty
     with no offsetting weight.

Also note: `left-sided`, `right-sided`, and `mild-to-moderate` all occur in train.csv. A naive
"delete every hyphen between two letters" rule (as in the toy examples "air-space" -> "airspace")
would silently swallow the critical laterality/severity words inside those compounds. This
scorer instead resolves each hyphen-chain as a whole: if any dash-segment is itself a critical
word (negation/laterality/severity/acuity), the chain is split into separate words (spaces)
instead of merged, so "left-sided" -> "left sided" (keeps "left" weighted 4.0) while
"air-space" -> "airspace" (no critical segment, merges as in the spec's own example).
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import OrderedDict

import pandas as pd

# ---------------------------------------------------------------------------
# Word weight classes
# ---------------------------------------------------------------------------

NEGATION_WORDS = {
    "no", "not", "without", "absent", "absence", "denies", "denied", "negative", "none", "nor",
}
LATERALITY_WORDS = {"left", "right", "bilateral", "bilaterally"}
SEVERITY_WORDS = {
    "mild", "mildly", "moderate", "moderately", "severe", "severely",
    "minimal", "minimally", "trace", "marked", "markedly",
}
ACUITY_WORDS = {"acute", "acutely", "chronic", "chronically", "subacute"}

CRITICAL_WORDS = NEGATION_WORDS | LATERALITY_WORDS | SEVERITY_WORDS | ACUITY_WORDS

# "in" is deliberately excluded — in this corpus it is overwhelmingly the preposition
# ("fluid in the...") and inches are never used as a unit, so treating it as a unit token
# would corrupt scoring. "x" is included: it only ever appears here as a dimension separator
# ("6.2 x 5.3 cm"), carrying no clinical weight of its own.
_RAW_FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "of", "with", "in", "on", "at", "to", "for", "by", "as",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those",
    "it", "its", "there", "their", "from", "also", "again", "noted", "seen", "x",
}
FUNCTION_WORDS = _RAW_FUNCTION_WORDS - CRITICAL_WORDS
assert not (FUNCTION_WORDS & CRITICAL_WORDS), "function words must never overlap critical words"

# No spelled-out "millimeters"/"centimeters" etc. occur anywhere in train.csv (verified by
# inspection) — mm/cm abbreviations are used exclusively — but the standardization is kept
# generic per the spec so it still works if test.csv (unseen) uses a spelled-out variant.
UNIT_STANDARDIZATION = {
    "millimeters": "mm", "millimeter": "mm", "millimetres": "mm", "millimetre": "mm",
    "centimeters": "cm", "centimeter": "cm", "centimetres": "cm", "centimetre": "cm",
    "milliliters": "ml", "milliliter": "ml", "millilitres": "ml", "millilitre": "ml",
}
UNIT_TOKENS = {"mm", "cm", "ml", "cc", "hu"}

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_LIST_MARKER_RE = re.compile(r"^\s*(?:\d{1,2}[.)]|[-*•])\s+")
_HYPHEN_CHAIN_RE = re.compile(r"[a-z]+(?:-[a-z]+)+")
# Number token: a leading sign is only consumed when NOT immediately preceded by a digit,
# so "3-4" tokenizes as "3", "4" (a range) rather than "3", "-4" (three and negative four).
_SUBTOKEN_RE = re.compile(r"(?<!\d)[+-]?\d+(?:\.\d+)?|[a-z]+")
_NUMBER_TOKEN_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


def _resolve_hyphen_chain(match: "re.Match[str]") -> str:
    segments = match.group(0).split("-")
    if any(seg in CRITICAL_WORDS for seg in segments):
        return " ".join(segments)
    return "".join(segments)


def _split_word(word: str) -> list[str]:
    tokens = []
    for m in _SUBTOKEN_RE.finditer(word):
        tok = m.group(0)
        if tok.isalpha():
            tok = UNIT_STANDARDIZATION.get(tok, tok)
        tokens.append(tok)
    return tokens


def normalize_text(text) -> list[str]:
    """Tokenize `text` per the normalization rules in TRD.md §4.11."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return []

    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower().replace("×", "x")  # × (multiplication sign) -> x

    lines = text.split("\n")
    stripped_lines = [_LIST_MARKER_RE.sub("", line) for line in lines]
    text = " ".join(stripped_lines)

    text = _HYPHEN_CHAIN_RE.sub(_resolve_hyphen_chain, text)

    tokens: list[str] = []
    for raw_word in re.split(r"\s+", text):
        if raw_word:
            tokens.extend(_split_word(raw_word))
    return tokens


def token_weight(token: str) -> float:
    if token in CRITICAL_WORDS:
        return 4.0
    if _NUMBER_TOKEN_RE.match(token):
        return 4.0
    if token in UNIT_TOKENS:
        return 4.0
    if token in FUNCTION_WORDS:
        return 0.25
    return 2.0


# ---------------------------------------------------------------------------
# Weighted edit distance
# ---------------------------------------------------------------------------

def weighted_edit_distance(ref_tokens: list[str], hyp_tokens: list[str]) -> float:
    """Ordered word-level weighted Levenshtein distance, normalized to [0, 1].

    Normalized by max(sum(weight) for ref, sum(weight) for hyp), capped at 1.0.
    """
    n, m = len(ref_tokens), len(hyp_tokens)
    ref_w = [token_weight(t) for t in ref_tokens]
    hyp_w = [token_weight(t) for t in hyp_tokens]

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + ref_w[i - 1]
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + hyp_w[j - 1]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                sub_cost = 0.0
            else:
                sub_cost = max(ref_w[i - 1], hyp_w[j - 1])
            dp[i][j] = min(
                dp[i - 1][j] + ref_w[i - 1],
                dp[i][j - 1] + hyp_w[j - 1],
                dp[i - 1][j - 1] + sub_cost,
            )

    raw = dp[n][m]
    denom = max(sum(ref_w), sum(hyp_w))
    if denom == 0:
        return 0.0
    return min(raw / denom, 1.0)


# ---------------------------------------------------------------------------
# Report field parsing
# ---------------------------------------------------------------------------

_FIELD_LINE_RE = re.compile(r"^([A-Z][A-Z0-9 /()&-]{1,40}):\s*(.*)$")
_UNLABELLED_KEY = "__UNLABELLED__"


def parse_report_fields(report_text) -> tuple["OrderedDict[str, str]", str]:
    """Split a FINDINGS/IMPRESSION report into (field_dict, impression_text).

    Any text appearing before the first recognized field label is kept under the reserved key
    "__UNLABELLED__" so callers can penalize it as unexpected content; it is otherwise ignored.
    """
    text = "" if report_text is None else str(report_text)
    if isinstance(report_text, float) and math.isnan(report_text):
        text = ""

    m = re.search(r"^IMPRESSION:\s*", text, flags=re.MULTILINE)
    if m:
        findings_block = text[: m.start()]
        impression_text = text[m.end():].strip()
    else:
        findings_block = text
        impression_text = ""

    findings_block = re.sub(r"^\s*FINDINGS:\s*\n?", "", findings_block, count=1)

    fields: "OrderedDict[str, str]" = OrderedDict()
    current_label = _UNLABELLED_KEY
    current_lines: list[str] = []

    def _flush() -> None:
        text_val = "\n".join(current_lines).strip()
        if current_label == _UNLABELLED_KEY and not text_val:
            return
        if current_label in fields:
            fields[current_label] = (fields[current_label] + "\n" + text_val).strip()
        else:
            fields[current_label] = text_val

    for line in findings_block.split("\n"):
        matched = _FIELD_LINE_RE.match(line)
        if matched:
            _flush()
            current_label = matched.group(1).strip()
            current_lines = [matched.group(2)] if matched.group(2) else []
        else:
            current_lines.append(line)
    _flush()

    return fields, impression_text


# ---------------------------------------------------------------------------
# Case / dataset scoring
# ---------------------------------------------------------------------------

def score_case(reference_report: str, submitted_report: str, template_content: str) -> dict:
    ref_fields, ref_impression = parse_report_fields(reference_report)
    sub_fields, sub_impression = parse_report_fields(submitted_report)
    tmpl_fields, _ = parse_report_fields(template_content)

    field_scores: dict[str, dict] = {}
    numerator = 0.0
    denominator = 0.0

    for label, ref_text in ref_fields.items():
        tmpl_text = tmpl_fields.get(label, "")
        changed = normalize_text(ref_text) != normalize_text(tmpl_text)
        weight = 3.0 if changed else 1.0
        sub_text = sub_fields.get(label, "")
        edit = weighted_edit_distance(normalize_text(ref_text), normalize_text(sub_text))
        numerator += weight * edit
        denominator += weight
        field_scores[label] = {"weight": weight, "edit": edit, "changed": changed}

    for label, sub_text in sub_fields.items():
        if label in ref_fields:
            continue
        weight = 3.0
        edit = weighted_edit_distance([], normalize_text(sub_text))
        numerator += weight * edit  # per spec: numerator only, denominator untouched
        field_scores[label] = {"weight": weight, "edit": edit, "changed": True, "unexpected": True}

    F = (numerator / denominator) if denominator > 0 else 0.0
    I = weighted_edit_distance(normalize_text(ref_impression), normalize_text(sub_impression))
    RES_case = 0.65 * F + 0.35 * I

    return {"RES_case": RES_case, "F": F, "I": I, "field_scores": field_scores}


def score_dataset(df: pd.DataFrame, report_column: str) -> dict:
    per_case = []
    for _, row in df.iterrows():
        result = score_case(row["report"], row[report_column], row["template_content"])
        result["case_id"] = row.get("case_id")
        per_case.append(result)

    n = len(per_case)
    mean_res = sum(r["RES_case"] for r in per_case) / n if n else 0.0
    mean_f = sum(r["F"] for r in per_case) / n if n else 0.0
    mean_i = sum(r["I"] for r in per_case) / n if n else 0.0

    return {
        "mean_RES": mean_res,
        "mean_F": mean_f,
        "mean_I": mean_i,
        "n_cases": n,
        "per_case": per_case,
    }
