import os

import pandas as pd
import pytest

from schemas import ExtractedFinding, ExtractionResult, FieldRouting
from validator import _rule_based_checks, validate

TRAIN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")


def test_rule_based_negation_check_catches_flipped_negation():
    extraction = ExtractionResult(findings=[
        ExtractedFinding(finding="acute diverticulitis", polarity="negative", raw_span="x"),
    ])
    routings = [FieldRouting(finding_index=0, field_label="STOMACH AND BOWEL", confidence="high")]

    # Negation flipped: "acute diverticulitis is present" contains no negation marker.
    broken_fields = {"STOMACH AND BOWEL": "Acute diverticulitis is present."}
    issues = _rule_based_checks(extraction, routings, broken_fields)
    assert any(i.kind == "negation" and i.field_label == "STOMACH AND BOWEL" for i in issues)

    fixed_fields = {"STOMACH AND BOWEL": "No evidence of acute diverticulitis."}
    issues = _rule_based_checks(extraction, routings, fixed_fields)
    assert not any(i.kind == "negation" for i in issues)


def test_rule_based_measurement_check_catches_dropped_measurement():
    extraction = ExtractionResult(findings=[
        ExtractedFinding(finding="hepatomegaly", measurement="16 cm", polarity="positive", raw_span="x"),
    ])
    routings = [FieldRouting(finding_index=0, field_label="LIVER", confidence="high")]

    broken_fields = {"LIVER": "Mild hepatomegaly. No focal hepatic lesion."}  # measurement dropped
    issues = _rule_based_checks(extraction, routings, broken_fields)
    assert any(i.kind == "measurement" and i.field_label == "LIVER" for i in issues)

    fixed_fields = {"LIVER": "Mild hepatomegaly, measuring approximately 16 cm. No focal hepatic lesion."}
    issues = _rule_based_checks(extraction, routings, fixed_fields)
    assert not any(i.kind == "measurement" for i in issues)


def test_rule_based_laterality_check_catches_dropped_laterality():
    extraction = ExtractionResult(findings=[
        ExtractedFinding(finding="hip osteoarthrosis", laterality="right", polarity="positive", raw_span="x"),
    ])
    routings = [FieldRouting(finding_index=0, field_label="JOINTS", confidence="high")]

    broken_fields = {"JOINTS": "Mild hip osteoarthrosis."}  # laterality dropped
    issues = _rule_based_checks(extraction, routings, broken_fields)
    assert any(i.kind == "laterality" and i.field_label == "JOINTS" for i in issues)

    fixed_fields = {"JOINTS": "Mild right hip osteoarthrosis."}
    issues = _rule_based_checks(extraction, routings, fixed_fields)
    assert not any(i.kind == "laterality" for i in issues)


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
def test_validate_catches_planted_issues_end_to_end():
    """Deliberately construct a broken report with three planted issues -- a flipped negation,
    a dropped measurement, and a reverted (unnecessary-edit) field -- and confirm validate()
    catches all of them (rule-based catches the first two; the Groq call catches the third)."""
    if not os.path.exists(TRAIN_PATH):
        pytest.skip("train.csv not found")
    df = pd.read_csv(TRAIN_PATH)
    row = df[df["case_id"] == "4813b1de-0fc4-4fd5-8734-a3d67bd7f5bb"].iloc[0]

    extraction = ExtractionResult(findings=[
        ExtractedFinding(finding="hepatomegaly", anatomy="liver", severity="mild",
                          measurement="16 cm", polarity="positive", raw_span="x"),
        ExtractedFinding(finding="acute diverticulitis", anatomy="colon", acuity="acute",
                          polarity="negative", raw_span="x"),
    ])
    routings = [
        FieldRouting(finding_index=0, field_label="LIVER", confidence="high"),
        FieldRouting(finding_index=1, field_label="STOMACH AND BOWEL", confidence="high"),
    ]

    broken_report = (
        "FINDINGS:\n"
        "LIVER: Mild hepatomegaly. Normal attenuation. No focal hepatic lesion.\n"  # measurement dropped
        "GALLBLADDER AND BILIARY TREE: Normal. No biliary ductal dilatation.\n"
        "PANCREAS: Normal in size and contour.\n"
        "SPLEEN: Normal.\n"
        "ADRENAL GLANDS: Normal in size and morphology.\n"
        "KIDNEYS AND URETERS: Normal appearance without renal calculi or hydronephrosis.\n"
        "URINARY BLADDER: Normal. No calculus or mass lesion. Recently rescanned per protocol.\n"  # unnecessary edit, nothing dictated
        "REPRODUCTIVE: The uterus and ovaries/prostate are normal in appearance.\n"
        "MAJOR VESSELS: The abdominal aorta is normal in caliber.\n"
        "PERITONEUM: No ascites. No pneumoperitoneum.\n"
        "ABDOMINAL WALL: The abdominal wall is unremarkable. No hernia.\n"
        "LYMPH NODES: No significant mesenteric or retroperitoneal lymphadenopathy.\n"
        "STOMACH AND BOWEL: Acute diverticulitis is present.\n"  # negation flipped
        "BONES: No acute osseous abnormality.\n"
        "\n"
        "IMPRESSION:\n"
        "No acute intra-abdominal abnormality identified."
    )

    result = validate(row, broken_report, extraction, routings)

    print("\n--- validation issues ---")
    for issue in result.issues:
        print(f"{issue.kind} | {issue.field_label} | {issue.detail}")

    assert not result.passed
    kinds = {i.kind for i in result.issues}
    assert "negation" in kinds
    assert "measurement" in kinds
    assert "unnecessary_edit" in kinds
