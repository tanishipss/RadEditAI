"""
Router — FR3 (TRD.md §4.4), field routing (prompts_v2.md §2).

Deterministic-first design: build_lexicon(train_df) mines word -> {field_label: count}
co-occurrence counts from every (report field text, field_label) pair in train.csv — not a
hand-written anatomy list. Field labels vary across templates/body parts (185 unique labels seen
across train.csv), so picking one GLOBAL winner per word up front badly fragments near-synonymous
labels: e.g. "lymph" splits across "LYMPH NODES" / "MEDIASTINAL/HILAR LYMPH NODES" /
"SUPRACLAVICULAR/AXILLARY/MEDIASTINAL/HILAR" (same concept, different templates' own vocabulary),
so no single label reaches a global dominance threshold even though the word is a perfectly good
routing signal for any ONE of those templates. Resolving each word AT ROUTING TIME, restricted to
the current case's own field_labels list, fixes this (verified empirically against the worked
example in prompts_v2.md §2: 12/13 anatomy-bearing terms resolve this way, vs. only 4/13 with a
naive global-dominance lexicon).

Only ONE batched Groq call is made per case, for whatever findings the deterministic pass
couldn't confidently place — per this prompt's explicit instruction, not "one call per anatomy
term" as prompts_v2.md §2's note more loosely suggests, to keep the per-case call budget bounded
(TRD.md §8: ≤4 Groq calls/case). Multi-organ findings (anatomy names >1 organ, comma-separated)
are still fully supported on the deterministic path alone: each comma-separated organ segment is
looked up independently and every confidently-resolved field is kept, producing multiple
FieldRouting entries for one finding_index — this is the "multi-field-per-finding edge case"
prompts_v2.md §2 calls out, achieved without any extra LLM calls.
"""

from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd
from pydantic import BaseModel

from groq_client import call_groq
from res_scorer import CRITICAL_WORDS, FUNCTION_WORDS, normalize_text
from schemas import ExtractionResult, FieldRouting
from template_parser import UNLABELLED_LABEL, parse_template

ALL_REMAINING_UNMENTIONED = "__ALL_REMAINING_UNMENTIONED__"

# Clinical-report boilerplate: generic enough to pass a frequency/dominance filter (e.g.
# "changes" statistically leans BONES simply because "degenerative changes" is common) but not
# an anatomy-specific signal. Severity/acuity/negation/laterality words (res_scorer.CRITICAL_WORDS)
# are excluded too, since ExtractedFinding already captures those as separate structured fields.
_BOILERPLATE_WORDS = {
    "changes", "change", "evidence", "identified", "present", "seen", "noted", "involving",
    "visualized", "demonstrates", "demonstrate", "consistent", "finding", "findings", "normal",
    "abnormal", "abnormality", "unremarkable", "remarkable", "within", "limits", "grossly",
    "otherwise", "rest",
}
# Generic anatomical-position/region modifiers: these qualify almost any organ in any body
# region, so even a high RAW count under one field is noise, not a real signal — e.g.
# "abdominal" (as in "abdominal aorta") spreads thinly across VASCULATURE/SOFT TISSUES/LYMPH
# NODES/UPPER ABDOMEN/... (dominance ~0.12) yet still out-competed "aorta" -> MAJOR VESSELS
# (also fragmented, dominance ~0.11, but the actually-specific word here) on raw count alone;
# similarly "distal" (~0.35 dominance toward BONES/JOINTS, from orthopedic reports generally)
# out-competed "bowel" -> STOMACH AND BOWEL (~0.53 dominance) in a "distal bowel" segment.
# Both were caught empirically (scripts/_verify_router.py) before this exclusion was added.
_GENERIC_POSITION_WORDS = {
    "abdominal", "distal", "proximal", "medial", "lateral", "anterior", "posterior",
    "superior", "inferior", "surrounding", "adjacent", "region", "area",
}
_MIN_WORD_LENGTH = 4
_MIN_MATCH_COUNT = 2

_BLANKET_NORMAL_RE = re.compile(r"\b(rest normal|otherwise unremarkable|otherwise normal)\b")


def build_lexicon(train_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Mine word -> {field_label: count} co-occurrence from every report in train_df."""
    co_occurrence: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for _, row in train_df.iterrows():
        parsed = parse_template(row["report"])
        for field in parsed["findings_fields"]:
            label, text = field["label"], field["content"]
            if label == UNLABELLED_LABEL:
                continue
            for tok in normalize_text(text):
                if (
                    tok in FUNCTION_WORDS
                    or tok in CRITICAL_WORDS
                    or tok in _BOILERPLATE_WORDS
                    or tok in _GENERIC_POSITION_WORDS
                ):
                    continue
                if len(tok) < _MIN_WORD_LENGTH or tok.replace(".", "", 1).isdigit():
                    continue
                co_occurrence[tok][label] += 1

    return {word: dict(field_counts) for word, field_counts in co_occurrence.items()}


def _best_field_for_tokens(
    tokens: list[str], field_labels: list[str], lexicon: dict[str, dict[str, int]]
) -> str | None:
    field_set = set(field_labels)
    best_field: str | None = None
    best_count = 0
    for tok in tokens:
        counts = lexicon.get(tok)
        if not counts:
            continue
        for field, count in counts.items():
            if field in field_set and count > best_count:
                best_field, best_count = field, count
    if best_field is not None and best_count >= _MIN_MATCH_COUNT:
        return best_field
    return None


def _is_blanket_normal(finding) -> bool:
    return finding.anatomy in (None, "") and bool(_BLANKET_NORMAL_RE.search(finding.finding.lower()))


def _route_one_deterministic(
    finding, field_labels: list[str], lexicon: dict[str, dict[str, int]]
) -> list[tuple[str, str]] | None:
    """Returns a list of (field_label, confidence) for this finding, or None if unresolved."""
    if _is_blanket_normal(finding):
        return [(ALL_REMAINING_UNMENTIONED, "high")]

    anatomy = finding.anatomy or ""
    if "," in anatomy:
        resolved_fields: list[str] = []
        for segment in anatomy.split(","):
            seg_tokens = normalize_text(segment)
            field = _best_field_for_tokens(seg_tokens, field_labels, lexicon)
            if field and field not in resolved_fields:
                resolved_fields.append(field)
        if resolved_fields:
            return [(f, "high") for f in resolved_fields]
        return None

    anatomy_tokens = normalize_text(anatomy)
    field = _best_field_for_tokens(anatomy_tokens, field_labels, lexicon)
    if field:
        return [(field, "high")]

    finding_tokens = normalize_text(finding.finding)
    field = _best_field_for_tokens(finding_tokens, field_labels, lexicon)
    if field:
        return [(field, "high")]

    return None


_UNIVERSAL_RULES = """UNIVERSAL RULES (apply regardless of task):
- Never invent, infer, or assume any finding, value, or qualifier not explicitly present in the
  text you are given. Silence is not a finding.
- Never use modality, body_part, study_description, patient_age_band, or patient_sex to create
  findings — they are context only, for disambiguating anatomy terms (e.g. "AC joint" in a
  shoulder case), never a source of clinical content.
- Negation words ("no", "without", "absent", "denies") must never be dropped, added, or flipped.
- Laterality words ("left", "right", "bilateral") must never be dropped, added, or flipped.
- Measurements and units must be copied character-for-character from the source text (e.g. if
  the source says "3.2 cm", never write "3 cm", "32 mm", or "approximately 3 cm").
- Output ONLY valid JSON matching the given schema. No markdown fences, no prose before or after,
  no trailing commentary, no explanation of your reasoning.
- If you are unsure whether something qualifies, leave it out rather than guess."""

_SYSTEM_PROMPT = f"""You are a radiology report field-routing engine. Assign each given finding to exactly one field
label from the EXACT list of field labels provided for this case's template.

TASK-SPECIFIC RULES:
1. Use ONLY field labels from the provided list, copied character-for-character (same spelling,
   spacing, casing). Never invent a label, never abbreviate or expand one.
2. If a finding matches no listed field well, and "OTHER FINDINGS" is in the list, route there.
   If "OTHER FINDINGS" is not in the list, pick the single closest anatomical/clinical field and
   set confidence="low" so it can be reviewed.
3. A finding with anatomy=null and finding text like "rest normal" / "otherwise unremarkable"
   (a blanket normal statement) should be routed with a special reserved label
   "__ALL_REMAINING_UNMENTIONED__" — this is a signal to the caller that all fields not otherwise
   touched by any other finding should be treated as explicitly confirmed-normal (still copied
   from template verbatim, since template text already says "normal" — but the case counts as
   template-confirmed rather than unaddressed). Do not route this to any real field label.
4. Route by clinical/anatomical meaning, not surface keyword overlap. Example: a finding
   describing a technical limitation of a specific organ's visualization ("inadequate
   visualization of the urinary bladder") routes to that organ's field label (e.g.
   "URINARY BLADDER"), not to a generic "TECHNIQUE" field even if one exists, unless that field is
   clearly meant for this purpose.
5. If two findings describe the same organ, route both to the same field label — do not split one
   organ's content across two fields.
6. Each finding_index appears exactly once in your output, in the same order given.

{_UNIVERSAL_RULES}"""


class _RoutingResponse(BaseModel):
    routings: list[FieldRouting]


def _llm_route(
    findings_with_indices: list[tuple[int, object]], field_labels: list[str]
) -> list[FieldRouting]:
    findings_json = [
        {
            "finding_index": idx,
            "finding": f.finding,
            "anatomy": f.anatomy,
            "laterality": f.laterality,
            "severity": f.severity,
            "measurement": f.measurement,
            "acuity": f.acuity,
            "polarity": f.polarity,
        }
        for idx, f in findings_with_indices
    ]

    user = f"""TEMPLATE FIELD LABELS AVAILABLE (use exactly these strings):
{field_labels}

FINDINGS TO ROUTE:
{findings_json}

Return field_label and confidence for every finding_index."""

    result = call_groq(system=_SYSTEM_PROMPT, user=user, response_model=_RoutingResponse)
    return result.routings


def route_findings(
    extraction_result: ExtractionResult,
    field_labels: list[str],
    deterministic_lexicon: dict[str, dict[str, int]],
) -> list[FieldRouting]:
    routings: list[FieldRouting] = []
    needs_llm: list[tuple[int, object]] = []

    for idx, finding in enumerate(extraction_result.findings):
        resolved = _route_one_deterministic(finding, field_labels, deterministic_lexicon)
        if resolved is None:
            needs_llm.append((idx, finding))
            continue
        for field_label, confidence in resolved:
            routings.append(FieldRouting(finding_index=idx, field_label=field_label, confidence=confidence))

    if needs_llm:
        routings.extend(_llm_route(needs_llm, field_labels))

    routings.sort(key=lambda r: r.finding_index)
    return routings
