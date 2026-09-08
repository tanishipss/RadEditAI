import pandas as pd
import pytest

import profiling as prof


def test_dataset_overview_reports_shape_and_missing():
    df = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})
    overview = prof.dataset_overview(df)
    assert overview["n_rows"] == 3
    assert overview["n_cols"] == 2
    assert overview["missing_counts"] == {"a": 1, "b": 0}
    assert overview["missing_pct"]["a"] == pytest.approx(33.33, abs=0.01)


def test_categorical_summary_counts_and_ranks():
    df = pd.DataFrame({"modality": ["CT", "CT", "XRAY"]})
    summary = prof.categorical_summary(df, ["modality"])
    assert summary["modality"]["n_unique"] == 2
    assert summary["modality"]["value_counts"]["CT"] == 2
    assert summary["modality"]["value_counts"]["XRAY"] == 1


def test_distribution_stats_empty_and_nonempty():
    assert prof.distribution_stats([]) == {"n": 0}
    stats = prof.distribution_stats([1, 2, 3, 4])
    assert stats["n"] == 4
    assert stats["min"] == 1
    assert stats["max"] == 4
    assert stats["p50"] == pytest.approx(2.5)


def test_text_length_stats_counts_tokens_not_just_chars():
    series = pd.Series(["No acute fracture.", "Mild degenerative changes are noted in the spine."])
    stats = prof.text_length_stats(series)
    assert stats["char_length"]["n"] == 2
    assert stats["token_length"]["max"] > stats["token_length"]["min"]


def test_template_structure_summary_counts_fields_and_unlabelled():
    df = pd.DataFrame({
        "template_content": [
            "FINDINGS:\nBONES: normal.\nJOINTS: normal.\n\nIMPRESSION:\nNormal.",
            "FINDINGS:\nJust a flat paragraph with no labels at all.\n\nIMPRESSION:\nNormal.",
        ]
    })
    summary = prof.template_structure_summary(df)
    assert summary["n_unique_templates"] == 2
    assert summary["n_templates_with_no_field_labels"] == 1
    assert summary["fields_per_template"]["max"] == 2


def test_common_terms_excludes_function_and_critical_words():
    series = pd.Series(["There is no fracture but there is hepatomegaly and hepatomegaly again"])
    terms = dict(prof.common_terms(series))
    assert "hepatomegaly" in terms
    assert terms["hepatomegaly"] == 2
    assert "no" not in terms  # negation word, excluded as CRITICAL_WORDS
    assert "there" not in terms  # function word


def test_linguistic_flags_detect_laterality_measurement_negation():
    series = pd.Series([
        "No fracture identified.",
        "Left hip joint effusion measuring 3.2 cm.",
        "Unremarkable study.",
    ])
    flags = prof.linguistic_flags(series)
    assert flags["n_cases"] == 3
    assert flags["pct_with_negation"] == pytest.approx(100 * 1 / 3, abs=0.01)
    assert flags["pct_with_laterality"] == pytest.approx(100 * 1 / 3, abs=0.01)
    assert flags["pct_with_measurement"] == pytest.approx(100 * 1 / 3, abs=0.01)


def test_heuristic_abnormal_segment_count_skips_normal_openings():
    text = "No acute fracture. Mild degenerative changes are present. Unremarkable soft tissues."
    assert prof.heuristic_abnormal_segment_count(text) == 1


def test_compare_template_to_report_identifies_changed_and_unchanged_fields():
    template = "FINDINGS:\nBONES: No acute fracture.\nJOINTS: Normal alignment.\n\nIMPRESSION:\nNormal."
    report = "FINDINGS:\nBONES: Mild degenerative changes are present.\nJOINTS: Normal alignment.\n\nIMPRESSION:\nMild degenerative changes."
    diff = prof.compare_template_to_report("case1", template, report)
    assert diff["changed_fields"] == ["BONES"]
    assert diff["unchanged_fields"] == ["JOINTS"]
    assert diff["added_fields"] == []
    assert diff["removed_fields"] == []
    assert diff["impression_changed"] is True


def test_compare_template_to_report_matches_relabelled_fields_case_insensitively():
    template = "FINDINGS:\nBones: No acute fracture.\n\nIMPRESSION:\nNormal."
    report = "FINDINGS:\nBONES: No acute fracture.\n\nIMPRESSION:\nNormal."
    diff = prof.compare_template_to_report("case2", template, report)
    assert diff["n_relabelled_fields"] == 1
    assert diff["unchanged_fields"] == ["Bones"]
    assert diff["added_fields"] == []
    assert diff["removed_fields"] == []


def test_compare_template_to_report_detects_genuine_added_and_removed_fields():
    template = "FINDINGS:\nBONES: Normal.\nJOINTS: Normal.\n\nIMPRESSION:\nNormal."
    report = "FINDINGS:\nBONES: Normal.\nSOFT TISSUES: Normal.\n\nIMPRESSION:\nNormal."
    diff = prof.compare_template_to_report("case3", template, report)
    assert diff["removed_fields"] == ["JOINTS"]
    assert diff["added_fields"] == ["SOFT TISSUES"]
    assert diff["unchanged_fields"] == ["BONES"]


def test_field_change_frequency_aggregates_across_cases():
    diffs = [
        {"changed_fields": ["BONES"], "unchanged_fields": ["JOINTS"], "removed_fields": []},
        {"changed_fields": [], "unchanged_fields": ["BONES", "JOINTS"], "removed_fields": []},
    ]
    freq = prof.field_change_frequency(diffs)
    bones_row = freq[freq["field_label"] == "BONES"].iloc[0]
    assert bones_row["n_appearances"] == 2
    assert bones_row["n_changed"] == 1
    assert bones_row["change_rate"] == pytest.approx(0.5)


def test_res_scorer_compatibility_check_flags_mixed_case_templates():
    df = pd.DataFrame({
        "template_content": [
            "FINDINGS:\nBones: Normal.\nJoints: Normal.\n\nIMPRESSION:\nNormal.",
            "FINDINGS:\nBONES: Normal.\n\nIMPRESSION:\nNormal.",
        ]
    })
    result = prof.res_scorer_compatibility_check(df)
    assert result["n_templates_res_scorer_fails_to_parse_any_field"] == 1
    assert result["n_rows_affected"] == 1
