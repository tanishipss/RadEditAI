import os

import pytest

from extraction import extract_findings

# Real row from train.csv (case_id=4813b1de-0fc4-4fd5-8734-a3d67bd7f5bb), the same worked
# example shown in prompts_v2.md section 1.
CT_ABDOMEN_ROW = {
    "modality": "CT",
    "body_part": "Abdomen",
    "study_description": "CT A/P W-S",
    "patient_age_band": "60-64",
    "patient_sex": "male",
    "dictation": (
        "Mild hepatomegaly, measuring approximately 16 cm.\n"
        "Colonic diverticulosis without evidence of acute diverticulitis.\n"
        "Few nonspecific subcentimeter mesenteric lymph nodes.\n"
        "Atherosclerotic changes involving the abdominal aorta.\n"
        "Degenerative changes of the visualized spine.\n"
        "Limited inferior field of view with inadequate visualization of the distal bowel loops, "
        "urinary bladder and prostate."
    ),
}


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
def test_ct_abdomen_worked_example_extraction():
    result = extract_findings(CT_ABDOMEN_ROW)

    print("\n--- extraction JSON ---")
    print(result.model_dump_json(indent=2))

    findings_text = " | ".join(f.finding.lower() for f in result.findings)

    assert any("hepatomegaly" in f.finding.lower() for f in result.findings), findings_text
    assert any("diverticulosis" in f.finding.lower() for f in result.findings), findings_text
    assert any(
        "diverticulitis" in f.finding.lower() and f.polarity == "negative"
        for f in result.findings
    ), findings_text
    assert any(
        "field of view" in f.finding.lower() or "visualization" in f.finding.lower()
        for f in result.findings
    ), findings_text
