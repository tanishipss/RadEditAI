"""
Pipeline orchestration — TRD.md §6. Wires every module together in exactly the order specified:
parse_template -> get_few_shot_examples -> extract_findings -> route_findings -> detect_changes
-> edit_changed_fields -> assemble_findings -> generate_impression -> assemble_report ->
validate -> repair-if-needed.

Signature note: TRD.md §6 sketches `run_case(row) -> str`. This implementation instead returns a
dict with `report`, `extraction`, `routings`, and `validation` — the extra fields are needed by
scripts/run_eval.py's diagnostic error-analysis pass (which must call validate() again on the
final report without re-triggering repair) and cost nothing for scripts/run_submission.py, which
only reads `result["report"]`. This is the same signature-completeness deviation already made,
and documented, in field_editor.py/validator.py/repair.py.

Also introduces build_pipeline_context(train_df): the retrieval index and the router's
deterministic lexicon are each an expensive full pass over train_df and must be built ONCE per
run (or once per fold, in cross-validation), never per case.
"""

from __future__ import annotations

import pandas as pd

from assembler import assemble_findings, assemble_report
from change_detector import detect_changes
from extraction import extract_findings
from field_editor import edit_changed_fields
from impression_gen import generate_impression
from repair import repair
from retrieval import build_index, get_few_shot_examples
from router import build_lexicon, route_findings
from template_parser import UNLABELLED_LABEL, fields_as_dict, parse_template
from validator import validate


def build_pipeline_context(train_df: pd.DataFrame) -> dict:
    return {
        "train_df": train_df,
        "retrieval_index": build_index(train_df),
        "lexicon": build_lexicon(train_df),
    }


def run_case(row: pd.Series, context: dict) -> dict:
    parsed_template = parse_template(row["template_content"])
    fields = fields_as_dict(parsed_template)
    template_impression = parsed_template["impression"]
    field_labels = [label for label in fields if label != UNLABELLED_LABEL]

    examples = get_few_shot_examples(row, context["train_df"], context["retrieval_index"], k=2)

    extraction = extract_findings(row, examples)
    routings = route_findings(extraction, field_labels, context["lexicon"])
    changes = detect_changes(fields, routings)

    case_context = {
        "modality": row["modality"],
        "body_part": row["body_part"],
        "study_description": row["study_description"],
    }
    edits = edit_changed_fields(fields, changes, extraction, routings, examples, case_context=case_context)

    findings_text = assemble_findings(fields, changes, edits)
    impression = generate_impression(findings_text, row, template_impression, examples)
    report = assemble_report(findings_text, impression)

    validation = validate(row, report, extraction, routings)
    if not validation.passed:
        report = repair(row, report, validation.issues, extraction, routings)

    return {
        "report": report,
        "extraction": extraction,
        "routings": routings,
        "validation": validation,
    }
