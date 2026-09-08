"""
Validator — FR8 (TRD.md §4.9), rule-based checks (no LLM) + one Groq call (prompts_v2.md §6,
exact system/user prompt) for the harder checks (unsupported, wrong_field, unnecessary_edit,
impression_error).

Signature note: TRD.md §6's pipeline sketch shows `validate(row, report)` (2 args), but the
rule-based checks this prompt requires — "every negative finding's routed field must contain a
negation marker", "every measurement/laterality string must appear verbatim in its field's final
text" — are impossible to implement without knowing which finding was routed to which field. This
implementation therefore takes `extraction_result` and `routings` as two additional required
params; the pipeline sketch's 2-arg signature is treated as an imprecise illustration, the same
judgment call already made in field_editor.py for an analogous signature mismatch.
"""

from __future__ import annotations

from groq_client import call_groq
from res_scorer import NEGATION_WORDS, normalize_text
from schemas import ExtractionResult, FieldRouting, ValidationIssue, ValidationResult
from template_parser import fields_as_dict, parse_template

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

_SYSTEM_PROMPT = f"""You are a strict radiology report QA auditor. Compare DICTATION, ORIGINAL TEMPLATE, and the
GENERATED REPORT. Flag every discrepancy using the issue kinds below. Be exhaustive — missing a
real issue is worse than over-flagging.

- "unsupported": REPORT contains any finding/qualifier/diagnosis not traceable to DICTATION or
  TEMPLATE.
- "missing": DICTATION contains a finding not reflected anywhere in REPORT.
- "wrong_field": a finding appears under a field label that doesn't anatomically/clinically match
  it (e.g. a bowel finding under LIVER).
- "negation": a dictated negative ("no X") appears positive in REPORT, or vice versa.
- "laterality": left/right/bilateral in REPORT doesn't match DICTATION.
- "measurement": a numeric value or unit in REPORT differs from DICTATION, or was dropped/added.
- "unnecessary_edit": a field's text in REPORT differs from the same field in ORIGINAL TEMPLATE,
  but nothing in DICTATION required a change to that field.
- "impression_error": IMPRESSION omits a significant abnormal FINDINGS item, or introduces content
  not present in FINDINGS.

If none apply, return passed=true, issues=[].

{_UNIVERSAL_RULES}"""


def _rule_based_checks(
    extraction_result: ExtractionResult,
    routings: list[FieldRouting],
    final_fields: dict[str, str],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    for r in routings:
        if r.field_label not in final_fields:
            continue  # e.g. the reserved ALL_REMAINING_UNMENTONED label, not a real field
        finding = extraction_result.findings[r.finding_index]
        final_text = final_fields[r.field_label]
        final_tokens = set(normalize_text(final_text))

        if finding.polarity == "negative" and not (final_tokens & NEGATION_WORDS):
            issues.append(
                ValidationIssue(
                    kind="negation",
                    field_label=r.field_label,
                    detail=f"Finding {finding.finding!r} is negative but no negation marker found in {r.field_label!r}: {final_text!r}",
                )
            )

        if finding.measurement and finding.measurement not in final_text:
            issues.append(
                ValidationIssue(
                    kind="measurement",
                    field_label=r.field_label,
                    detail=f"Measurement {finding.measurement!r} not found verbatim in {r.field_label!r}: {final_text!r}",
                )
            )

        if finding.laterality and finding.laterality not in final_tokens:
            issues.append(
                ValidationIssue(
                    kind="laterality",
                    field_label=r.field_label,
                    detail=f"Laterality {finding.laterality!r} not found in {r.field_label!r}: {final_text!r}",
                )
            )

    return issues


def _llm_checks(row, report: str) -> list[ValidationIssue]:
    user = f"""DICTATION:
\"\"\"
{row['dictation']}
\"\"\"

ORIGINAL TEMPLATE:
\"\"\"
{row['template_content']}
\"\"\"

GENERATED REPORT:
\"\"\"
{report}
\"\"\"

Audit and return all issues."""

    result = call_groq(system=_SYSTEM_PROMPT, user=user, response_model=ValidationResult)
    return result.issues


def validate(
    row,
    report: str,
    extraction_result: ExtractionResult,
    routings: list[FieldRouting],
) -> ValidationResult:
    final_fields = fields_as_dict(parse_template(report))
    issues = _rule_based_checks(extraction_result, routings, final_fields)
    issues.extend(_llm_checks(row, report))
    return ValidationResult(passed=len(issues) == 0, issues=issues)
