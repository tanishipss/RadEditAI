import os

import pandas as pd
import pytest

from res_scorer import (
    normalize_text,
    score_case,
    score_dataset,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")


# ---------------------------------------------------------------------------
# normalize_text unit tests, one per normalization rule
# ---------------------------------------------------------------------------

def test_normalize_hyphen_removal_between_letters():
    assert normalize_text("air-space") == ["airspace"]
    assert normalize_text("follow-up") == ["followup"]


def test_normalize_preserves_critical_word_inside_hyphen_chain():
    # "left-sided" must NOT collapse into a single token that loses "left" (laterality, 4.0)
    assert normalize_text("left-sided") == ["left", "sided"]
    # "mild-to-moderate" must keep both severity words intact
    assert normalize_text("mild-to-moderate") == ["mild", "to", "moderate"]


def test_normalize_unit_standardization():
    assert normalize_text("3 millimeters") == ["3", "mm"]
    assert normalize_text("2.5 centimeters") == ["2.5", "cm"]
    assert normalize_text("3 mm") == ["3", "mm"]


def test_normalize_list_marker_stripping():
    assert normalize_text("1. Mild hepatomegaly.") == ["mild", "hepatomegaly"]
    assert normalize_text("2) Colonic diverticulosis") == ["colonic", "diverticulosis"]
    assert normalize_text("- No fracture") == ["no", "fracture"]


def test_normalize_letter_number_split():
    assert normalize_text("3cm nodule") == ["3", "cm", "nodule"]
    assert normalize_text("T2 hyperintensity") == ["t", "2", "hyperintensity"]


def test_normalize_number_range_not_treated_as_negative():
    # "3-4 mm" is a range (3 and 4), not "3" and "-4"
    assert normalize_text("3-4 mm") == ["3", "4", "mm"]


def test_normalize_signed_measurement_change_preserved():
    assert normalize_text("-2 mm change") == ["-2", "mm", "change"]


def test_normalize_multiplication_sign():
    assert normalize_text("9.2 × 5.9 cm") == ["9.2", "x", "5.9", "cm"]


def test_normalize_case_and_punctuation():
    assert normalize_text("No acute fracture.") == ["no", "acute", "fracture"]


# ---------------------------------------------------------------------------
# score_case sanity checks
# ---------------------------------------------------------------------------

TEMPLATE = "FINDINGS:\nLIVER: Normal in size and attenuation. No focal hepatic lesion.\n\nIMPRESSION:\nNo acute abnormality."
REFERENCE = "FINDINGS:\nLIVER: Mild hepatomegaly, measuring approximately 16 cm. No focal hepatic lesion.\n\nIMPRESSION:\n1. Mild hepatomegaly."


def test_identical_reference_and_submitted_scores_zero():
    result = score_case(REFERENCE, REFERENCE, TEMPLATE)
    assert result["RES_case"] == 0.0
    assert result["F"] == 0.0
    assert result["I"] == 0.0


def test_submitted_equals_template_scores_above_zero_when_reference_has_real_changes():
    result = score_case(REFERENCE, TEMPLATE, TEMPLATE)
    assert result["RES_case"] > 0.0
    assert result["F"] > 0.0


# ---------------------------------------------------------------------------
# Real data sanity run — 5 rows from train.csv
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="train.csv not found")
def test_real_rows_self_score_zero_and_template_score_positive():
    df = pd.read_csv(DATA_PATH).head(5)
    print()
    for _, row in df.iterrows():
        self_result = score_case(row["report"], row["report"], row["template_content"])
        template_result = score_case(row["report"], row["template_content"], row["template_content"])
        print(
            f"case_id={row['case_id']} "
            f"self_RES={self_result['RES_case']:.4f} "
            f"template_RES={template_result['RES_case']:.4f}"
        )
        assert self_result["RES_case"] == 0.0
        assert template_result["RES_case"] > 0.0


@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="train.csv not found")
def test_score_dataset_smoke():
    df = pd.read_csv(DATA_PATH).head(20).copy()
    df["submitted_report"] = df["report"]
    result = score_dataset(df, "submitted_report")
    assert result["n_cases"] == 20
    assert result["mean_RES"] == 0.0
