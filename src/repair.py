"""
Repair — LLM call #4 (TRD.md §4.10), field-scoped repair (prompts_v2.md §7). One Groq call per
DISTINCT flagged field/section — if a field has multiple planted issues, they're combined into
that field's single repair call rather than one call per issue. Re-assembles and re-runs
validate() once; if still failing, logs the case_id to data/repair_failures.log and returns the
best pre-repair version rather than blocking (a case must never come back empty).

Signature note: takes `extraction_result` and `routings` in addition to `(row, report, issues)`
so it can call validator.validate() for its internal re-check — the same signature-completeness
deviation already made in validator.py itself relative to TRD.md §6's pipeline sketch.
"""

from __future__ import annotations

import os

from pydantic import BaseModel

from groq_client import call_groq
from schemas import ExtractionResult, FieldRouting, ValidationIssue
from template_parser import fields_as_dict, fields_from_dict, parse_template, render
from validator import validate

_REPAIR_FAILURES_LOG = os.path.join(os.path.dirname(__file__), "..", "data", "repair_failures.log")

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

_SYSTEM_PROMPT = f"""You are correcting ONE field of a radiology report based on a single specific QA issue. Make the
SMALLEST edit that resolves the issue. Do not otherwise rewrite the field.

- If kind="unnecessary_edit": revert the field to ORIGINAL TEMPLATE TEXT exactly, verbatim.
- If kind="negation"/"laterality"/"measurement": correct only that value; leave surrounding text
  untouched.
- If kind="unsupported": remove only the unsupported span; keep the rest.
- If kind="missing": add only the missing finding, inserted minimally into existing text.
- If kind="wrong_field": remove the misplaced content from this field (it will be added to the
  correct field by a separate repair call).

{_UNIVERSAL_RULES}"""


class _FieldRepairResponse(BaseModel):
    field_label: str
    new_text: str


class _ImpressionRepairResponse(BaseModel):
    impression: str


def _issue_lines(issues: list[ValidationIssue]) -> str:
    return "\n".join(f"- {i.kind}: {i.detail}" for i in issues)


def _repair_field(field_label: str, current_text: str, template_text: str, dictation: str, issues: list[ValidationIssue]) -> str:
    user = f"""FIELD: "{field_label}"

CURRENT TEXT:
\"\"\"
{current_text}
\"\"\"

ORIGINAL TEMPLATE TEXT:
\"\"\"
{template_text}
\"\"\"

RELEVANT DICTATION:
\"\"\"
{dictation}
\"\"\"

ISSUE ({issues[0].kind}): {_issue_lines(issues)}

Return the corrected field text only."""

    result = call_groq(system=_SYSTEM_PROMPT, user=user, response_model=_FieldRepairResponse)
    return result.new_text


def _repair_impression(current_impression: str, dictation: str, issues: list[ValidationIssue]) -> str:
    user = f"""FIELD: "IMPRESSION"

CURRENT TEXT:
\"\"\"
{current_impression}
\"\"\"

ORIGINAL TEMPLATE TEXT:
\"\"\"
(not applicable — IMPRESSION is synthesized from FINDINGS, not templated)
\"\"\"

RELEVANT DICTATION:
\"\"\"
{dictation}
\"\"\"

ISSUE (impression_error): {_issue_lines(issues)}

Return the corrected impression text only."""

    result = call_groq(system=_SYSTEM_PROMPT, user=user, response_model=_ImpressionRepairResponse)
    return result.impression


def repair(
    row,
    report: str,
    issues: list[ValidationIssue],
    extraction_result: ExtractionResult,
    routings: list[FieldRouting],
) -> str:
    if not issues:
        return report

    template_fields = fields_as_dict(parse_template(row["template_content"]))
    report_parsed = parse_template(report)
    fields = fields_as_dict(report_parsed)
    impression = report_parsed["impression"]

    by_field: dict[str, list[ValidationIssue]] = {}
    impression_issues: list[ValidationIssue] = []
    for issue in issues:
        if issue.kind == "impression_error" or issue.field_label is None:
            impression_issues.append(issue)
        else:
            by_field.setdefault(issue.field_label, []).append(issue)

    for field_label, field_issues in by_field.items():
        if field_label not in fields:
            continue
        try:
            fields[field_label] = _repair_field(
                field_label,
                fields[field_label],
                template_fields.get(field_label, ""),
                row["dictation"],
                field_issues,
            )
        except Exception:
            continue  # best-effort: leave this field's pre-repair text if the call fails

    if impression_issues:
        try:
            impression = _repair_impression(impression, row["dictation"], impression_issues)
        except Exception:
            pass

    normalized_impression = impression if impression.startswith("\n") else "\n" + impression
    repaired_report = render(fields_from_dict(fields, normalized_impression))

    revalidation = validate(row, repaired_report, extraction_result, routings)
    if not revalidation.passed:
        os.makedirs(os.path.dirname(_REPAIR_FAILURES_LOG), exist_ok=True)
        with open(_REPAIR_FAILURES_LOG, "a", encoding="utf-8") as f:
            f.write(f"{row['case_id']}\n")
        return report  # best pre-repair version — never block, never return empty

    return repaired_report
