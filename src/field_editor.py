"""
Field editor — LLM call #2 (TRD.md §4.6), using the exact system prompt and user message
template from prompts_v2.md §4. ONE batched Groq call covers every changed field in the case
(not one call per field), per TRD.md §4.6 and the ≤4-calls/case budget in TRD.md §8.

Note on signature: this follows Prompt 6/7's explicitly given signature
`edit_changed_fields(fields, changes, extraction_result, routings, few_shot_examples)` rather
than TRD.md §6's pipeline sketch (`edit_changed_fields(fields, changes, extraction, examples)`,
4 args, no routings) — routings is required to know which findings belong to which changed
field, so the pipeline sketch's arg list is treated as an imprecise illustration, not a literal
contract. Neither version's signature includes the case's row (modality/body_part/
study_description), which the user message template needs for its "CASE CONTEXT:" line, so an
optional `case_context` dict param was added to supply it — omit it and that line is just blank.
"""

from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel

from groq_client import call_groq
from schemas import ExtractionResult, FieldEditResult, FieldRouting

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

_SYSTEM_PROMPT = f"""You are a radiology report editing engine. You edit ONE OR MORE template fields by incorporating
specific findings into their existing normal-template text. You preserve as much of the original
wording as possible.

TASK-SPECIFIC RULES:
1. Start from the ORIGINAL FIELD TEXT given for each field. Modify it minimally — do not
   rewrite it from scratch, do not restyle sentences that don't need to change.
2. Every finding with polarity="negative" must appear as an explicit negation in your output
   (e.g. "no biliary ductal dilatation") — never soften to uncertainty, never omit.
3. Every finding with polarity="positive" must retain its exact laterality/severity/acuity/
   measurement values verbatim (copy the string exactly; do not convert units, round numbers, or
   change "approximately 16 cm" to "16 cm").
4. Preserve any part of the ORIGINAL FIELD TEXT not contradicted by a finding routed to this
   field. Example: template says "Normal in size and attenuation. No focal hepatic lesion." and
   only hepatomegaly is dictated -> keep "No focal hepatic lesion" verbatim, only replace/extend
   the size statement.
5. Do not add content belonging to a different field, even if mentioned in the same dictation
   sentence.
6. Match the sentence style of the ORIGINAL FIELD TEXT and any few-shot examples given (formal,
   declarative radiology phrasing — "There is...", "No evidence of...", not casual language).
7. Return the COMPLETE replacement text for each field listed — not a diff, not a partial phrase.

{_UNIVERSAL_RULES}"""


class _FieldEditResponse(BaseModel):
    edits: list[FieldEditResult]


def _findings_by_changed_field(
    changes: dict[str, bool],
    extraction_result: ExtractionResult,
    routings: list[FieldRouting],
) -> "dict[str, list]":
    grouped: dict[str, list] = defaultdict(list)
    for r in routings:
        if changes.get(r.field_label):
            grouped[r.field_label].append(extraction_result.findings[r.finding_index])
    return grouped


def _format_few_shot_block(few_shot_examples: list[dict] | None) -> str:
    if not few_shot_examples:
        return ""
    parts = [
        "FEW-SHOT EXAMPLES (illustrative only — do not copy their specific findings into this "
        "case; they show style/format from similar prior cases, nothing more):"
    ]
    for ex in few_shot_examples:
        parts.append(f"\n{ex['diff_summary']}")
    return "\n".join(parts)


def edit_changed_fields(
    fields: dict[str, str],
    changes: dict[str, bool],
    extraction_result: ExtractionResult,
    routings: list[FieldRouting],
    few_shot_examples: list[dict] | None = None,
    case_context: dict | None = None,
) -> list[FieldEditResult]:
    findings_by_field = _findings_by_changed_field(changes, extraction_result, routings)
    changed_labels = [label for label, is_changed in changes.items() if is_changed]
    if not changed_labels:
        return []

    case_context = case_context or {}
    field_blocks = []
    for label in changed_labels:
        findings = findings_by_field.get(label, [])
        findings_json = [
            {
                "finding": f.finding,
                "anatomy": f.anatomy,
                "laterality": f.laterality,
                "severity": f.severity,
                "measurement": f.measurement,
                "acuity": f.acuity,
                "polarity": f.polarity,
            }
            for f in findings
        ]
        field_blocks.append(
            f'Field: "{label}"\n'
            f'ORIGINAL FIELD TEXT:\n"""\n{fields.get(label, "")}\n"""\n'
            f"FINDINGS ROUTED TO THIS FIELD:\n{findings_json}"
        )

    few_shot_block = _format_few_shot_block(few_shot_examples)

    user = f"""CASE CONTEXT: {case_context.get('modality', '')} / {case_context.get('body_part', '')} / {case_context.get('study_description', '')}

FIELDS TO EDIT:

{chr(10).join(field_blocks)}

{few_shot_block}

Return edited text for exactly these field_labels, nothing else."""

    result = call_groq(system=_SYSTEM_PROMPT, user=user, response_model=_FieldEditResponse)
    return result.edits
