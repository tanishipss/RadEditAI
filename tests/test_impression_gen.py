import os

import pandas as pd
import pytest

from impression_gen import generate_impression
from template_parser import parse_template, fields_as_dict

TRAIN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")


@pytest.fixture(scope="module")
def ct_abdomen_case():
    if not os.path.exists(TRAIN_PATH):
        pytest.skip("train.csv not found")
    df = pd.read_csv(TRAIN_PATH)
    row = df[df["case_id"] == "4813b1de-0fc4-4fd5-8734-a3d67bd7f5bb"].iloc[0]

    rep_fields = fields_as_dict(parse_template(row["report"]))
    final_findings_text = "FINDINGS:\n" + "\n".join(
        f"{label}: {text}" for label, text in rep_fields.items() if label != "__UNLABELLED__"
    )

    template_impression = parse_template(row["template_content"])["impression"]

    return row, final_findings_text, template_impression


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
def test_impression_is_numbered_and_mentions_key_findings(ct_abdomen_case):
    row, final_findings_text, template_impression = ct_abdomen_case

    impression = generate_impression(final_findings_text, row, template_impression)

    print("\n--- generated impression ---")
    print(impression)

    assert impression.strip().startswith("1.") or "\n1." in impression, (
        f"expected a numbered list (reference IMPRESSION for this case is numbered), got: {impression!r}"
    )

    lowered = impression.lower()
    assert "hepatomegaly" in lowered
    assert "diverticulosis" in lowered
    assert any(
        phrase in lowered
        for phrase in ("field of view", "field-of-view", "visualization", "pelvis")
    )
