import os

import pandas as pd
import pytest

from assembler import assemble_findings, assemble_report
from res_scorer import score_case
from schemas import FieldEditResult
from template_parser import fields_as_dict, parse_template

TRAIN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")

# Fields the router determined as changed for this case (test_router.py).
_CHANGED_LABELS = {"LIVER", "STOMACH AND BOWEL", "LYMPH NODES", "MAJOR VESSELS", "BONES",
                   "URINARY BLADDER", "REPRODUCTIVE"}


@pytest.fixture(scope="module")
def ct_abdomen_row():
    if not os.path.exists(TRAIN_PATH):
        pytest.skip("train.csv not found")
    df = pd.read_csv(TRAIN_PATH)
    return df[df["case_id"] == "4813b1de-0fc4-4fd5-8734-a3d67bd7f5bb"].iloc[0]


def test_assemble_report_reproduces_reference_closely(ct_abdomen_row):
    row = ct_abdomen_row
    template_fields = fields_as_dict(parse_template(row["template_content"]))
    reference_parsed = parse_template(row["report"])
    reference_fields = fields_as_dict(reference_parsed)
    reference_impression = reference_parsed["impression"]

    changes = {label: (label in _CHANGED_LABELS) for label in template_fields}
    # Use the REAL ground-truth edited text for each changed field, so this test exercises
    # assemble_findings/assemble_report's own reconstruction logic in isolation, not the quality
    # of upstream (network-dependent) extraction/routing/field_editor calls.
    edits = [
        FieldEditResult(field_label=label, new_text=reference_fields[label])
        for label in _CHANGED_LABELS
    ]

    findings_text = assemble_findings(template_fields, changes, edits)
    report = assemble_report(findings_text, reference_impression)

    print("\n--- assembled report ---")
    print(report)

    result = score_case(row["report"], report, row["template_content"])
    print(f"\nRES_case = {result['RES_case']:.4f}  (F={result['F']:.4f}, I={result['I']:.4f})")

    # Not an exact-match requirement (per the prompt) -- just report the number. A sanity
    # ceiling: reconstructing from the real ground-truth field texts should score very low,
    # since the only differences left are assembler's own formatting choices.
    assert result["RES_case"] < 0.1


def test_assemble_findings_preserves_unchanged_field_verbatim(ct_abdomen_row):
    row = ct_abdomen_row
    template_fields = fields_as_dict(parse_template(row["template_content"]))
    changes = {label: False for label in template_fields}  # nothing changed

    findings_text = assemble_findings(template_fields, changes, edits=[])

    for label, original_text in template_fields.items():
        if label == "__UNLABELLED__":
            continue
        assert f"{label}: {original_text.strip()}" in findings_text
