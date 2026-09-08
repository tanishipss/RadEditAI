import os

import pandas as pd
import pytest

from retrieval import build_index, get_few_shot_examples

TRAIN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")
TEST_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "test.csv")


@pytest.fixture(scope="module")
def train_df():
    if not os.path.exists(TRAIN_PATH):
        pytest.skip("train.csv not found")
    return pd.read_csv(TRAIN_PATH)


@pytest.fixture(scope="module")
def test_df():
    if not os.path.exists(TEST_PATH):
        pytest.skip("test.csv not found")
    return pd.read_csv(TEST_PATH)


def test_ten_real_test_rows_retrieval_quality(train_df, test_df):
    index = build_index(train_df)
    print()
    for _, case in test_df.head(10).iterrows():
        examples = get_few_shot_examples(case, train_df, index, k=2)
        tier = examples[0]["tier"] if examples else "NONE"
        case_ids = [e["case_id"] for e in examples]
        print(
            f"case_id={case['case_id']} modality={case['modality']} body_part={case['body_part']} "
            f"-> tier={tier} examples={case_ids}"
        )
        assert len(examples) <= 2
        for ex in examples:
            assert {"case_id", "template_content", "dictation", "report", "diff_summary", "tier"} <= ex.keys()


def test_retrieval_is_deterministic(train_df, test_df):
    index = build_index(train_df)
    case = test_df.iloc[0]
    first = get_few_shot_examples(case, train_df, index, k=2)
    second = get_few_shot_examples(case, train_df, index, k=2)
    assert [e["case_id"] for e in first] == [e["case_id"] for e in second]


def test_self_exclusion_when_evaluating_on_train_rows(train_df):
    index = build_index(train_df)
    case = train_df.iloc[0]
    examples = get_few_shot_examples(case, train_df, index, k=5)
    returned_ids = [e["case_id"] for e in examples]
    assert case["case_id"] not in returned_ids


def test_tier_fallback_with_synthetic_data():
    synthetic = pd.DataFrame(
        [
            {"case_id": "a1", "modality": "CT", "body_part": "Abdomen", "template_content": "T1", "dictation": "d1", "report": "T1"},
            {"case_id": "a2", "modality": "CT", "body_part": "Abdomen", "template_content": "T1", "dictation": "d2", "report": "T1"},
            {"case_id": "b1", "modality": "CT", "body_part": "Chest", "template_content": "T2", "dictation": "d3", "report": "T2"},
            {"case_id": "c1", "modality": "MRI", "body_part": "Knee", "template_content": "T3", "dictation": "d4", "report": "T3"},
        ]
    )
    index = build_index(synthetic)

    # exact template match tier
    case_exact = pd.Series({"case_id": "new1", "modality": "CT", "body_part": "Abdomen", "template_content": "T1"})
    ex = get_few_shot_examples(case_exact, synthetic, index, k=2)
    assert [e["case_id"] for e in ex] == ["a1", "a2"]
    assert ex[0]["tier"] == "exact_template"

    # falls back to modality+body_part (no exact template match, but same modality/body_part as b1)
    case_mbp = pd.Series({"case_id": "new2", "modality": "CT", "body_part": "Chest", "template_content": "UNSEEN"})
    ex = get_few_shot_examples(case_mbp, synthetic, index, k=2)
    assert [e["case_id"] for e in ex] == ["b1"]
    assert ex[0]["tier"] == "modality_body_part"

    # falls back to modality only (CT rows: a1, a2, b1 -- sorted by case_id, top 2)
    case_mod = pd.Series({"case_id": "new3", "modality": "CT", "body_part": "UNSEEN", "template_content": "UNSEEN"})
    ex = get_few_shot_examples(case_mod, synthetic, index, k=2)
    assert [e["case_id"] for e in ex] == ["a1", "a2"]
    assert ex[0]["tier"] == "modality"

    # no match at all
    case_none = pd.Series({"case_id": "new4", "modality": "XRAY", "body_part": "UNSEEN", "template_content": "UNSEEN"})
    ex = get_few_shot_examples(case_none, synthetic, index, k=2)
    assert ex == []


def test_diff_summary_abbreviates_unchanged_fields():
    from retrieval import _build_diff_summary

    template = "FINDINGS:\nLIVER: Normal in size. No focal lesion.\nSPLEEN: Normal.\n\nIMPRESSION:\nNormal."
    report = "FINDINGS:\nLIVER: Mild hepatomegaly. No focal lesion.\nSPLEEN: Normal.\n\nIMPRESSION:\n1. Mild hepatomegaly."

    summary = _build_diff_summary(template, report)
    assert "Mild hepatomegaly" in summary
    assert "SPLEEN: ...unchanged, copied verbatim..." in summary
