import os

import pytest

from field_editor import edit_changed_fields
from schemas import ExtractedFinding, ExtractionResult, FieldRouting

# CT abdomen worked example from prompts_v2.md section 4 (case_id
# 4813b1de-0fc4-4fd5-8734-a3d67bd7f5bb). Original field text given directly in the doc's worked
# example, so this test doesn't depend on a live extraction/routing call.
_FIELDS = {
    "LIVER": "Normal in size and attenuation. No focal hepatic lesion.",
    "STOMACH AND BOWEL": "No evidence of bowel obstruction.",
}
_CHANGES = {"LIVER": True, "STOMACH AND BOWEL": True}

_EXTRACTION = ExtractionResult(
    findings=[
        ExtractedFinding(finding="hepatomegaly", anatomy="liver", severity="mild",
                          measurement="16 cm", polarity="positive", raw_span="x"),
        ExtractedFinding(finding="colonic diverticulosis", anatomy="colon", polarity="positive", raw_span="x"),
        ExtractedFinding(finding="acute diverticulitis", anatomy="colon", acuity="acute",
                          polarity="negative", raw_span="x"),
        ExtractedFinding(finding="limited field of view / inadequate visualization",
                          anatomy="distal bowel, urinary bladder, prostate",
                          polarity="positive", raw_span="x"),
    ]
)
_ROUTINGS = [
    FieldRouting(finding_index=0, field_label="LIVER", confidence="high"),
    FieldRouting(finding_index=1, field_label="STOMACH AND BOWEL", confidence="high"),
    FieldRouting(finding_index=2, field_label="STOMACH AND BOWEL", confidence="high"),
    FieldRouting(finding_index=3, field_label="STOMACH AND BOWEL", confidence="high"),
]
_CASE_CONTEXT = {"modality": "CT", "body_part": "Abdomen", "study_description": "CT A/P W-S"}


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
def test_liver_and_stomach_bowel_edits():
    edits = edit_changed_fields(_FIELDS, _CHANGES, _EXTRACTION, _ROUTINGS, case_context=_CASE_CONTEXT)

    print("\n--- field edits ---")
    for e in edits:
        print(f"{e.field_label}: {e.new_text!r}")

    by_label = {e.field_label: e.new_text for e in edits}
    assert set(by_label.keys()) == {"LIVER", "STOMACH AND BOWEL"}

    liver_text = by_label["LIVER"]
    assert "No focal hepatic lesion" in liver_text
    assert "16 cm" in liver_text

    bowel_text = by_label["STOMACH AND BOWEL"]
    assert "No evidence of bowel obstruction." in bowel_text
