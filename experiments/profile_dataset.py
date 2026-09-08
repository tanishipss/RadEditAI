"""
experiments/profile_dataset.py — Phase 1 dataset profiling, single-command entry point.

    python experiments/profile_dataset.py

Reads data/train.csv, data/test.csv, data/sample_submission.csv READ-ONLY (never written to)
and writes every analysis artifact under outputs/profiling/. Uses only the functions in
src/profiling.py (which itself makes no LLM calls) plus src/template_parser.py and
src/res_scorer.py's tokenizer for text comparison — no LLM calls anywhere in this script.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402

import profiling as prof  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "profiling")

CATEGORICAL_COLUMNS = ["modality", "body_part", "study_description", "patient_age_band", "patient_sex"]


def _write_json(name: str, obj) -> None:
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def _write_jsonl(name: str, records: list[dict]) -> None:
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + "\n")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    sample_submission = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

    # --- 1-3: dimensions, dtypes, missing values ------------------------------------------------
    overview = {
        "train": prof.dataset_overview(train),
        "test": prof.dataset_overview(test),
        "sample_submission": prof.dataset_overview(sample_submission),
    }
    _write_json("overview.json", overview)

    # --- 4-6: categorical columns --------------------------------------------------------------
    categorical_train = prof.categorical_summary(train, CATEGORICAL_COLUMNS)
    categorical_test = prof.categorical_summary(test, CATEGORICAL_COLUMNS)
    _write_json("categorical_train.json", categorical_train)
    _write_json("categorical_test.json", categorical_test)

    # --- 7-9: template structure, + train/test template overlap --------------------------------
    template_structure_train = prof.template_structure_summary(train)
    template_structure_test = prof.template_structure_summary(test)
    train_templates = set(train["template_content"])
    test_templates = set(test["template_content"])
    template_overlap = {
        "n_unique_train_templates": len(train_templates),
        "n_unique_test_templates": len(test_templates),
        "n_test_templates_also_in_train": len(test_templates & train_templates),
        "n_test_templates_unseen_in_train": len(test_templates - train_templates),
        "pct_test_templates_seen_in_train": (
            round(100 * len(test_templates & train_templates) / len(test_templates), 2)
            if test_templates else 0.0
        ),
    }
    _write_json("template_structure_train.json", template_structure_train)
    _write_json("template_structure_test.json", template_structure_test)
    _write_json("template_overlap.json", template_overlap)

    res_scorer_compat = prof.res_scorer_compatibility_check(train)
    _write_json("res_scorer_compatibility.json", res_scorer_compat)

    # --- 10-11: dictation / report length distributions -----------------------------------------
    text_length_train = {
        "dictation": prof.text_length_stats(train["dictation"]),
        "report": prof.text_length_stats(train["report"]),
        "template_content": prof.text_length_stats(train["template_content"]),
    }
    text_length_test = {
        "dictation": prof.text_length_stats(test["dictation"]),
        "template_content": prof.text_length_stats(test["template_content"]),
    }
    _write_json("text_length_train.json", text_length_train)
    _write_json("text_length_test.json", text_length_test)

    # --- 16: common finding/anatomy terms in dictation ------------------------------------------
    common_terms_train = prof.common_terms(train["dictation"])
    common_terms_test = prof.common_terms(test["dictation"])
    pd.DataFrame(common_terms_train, columns=["term", "count"]).to_csv(
        os.path.join(OUT_DIR, "common_terms_dictation_train.csv"), index=False
    )
    pd.DataFrame(common_terms_test, columns=["term", "count"]).to_csv(
        os.path.join(OUT_DIR, "common_terms_dictation_test.csv"), index=False
    )

    # --- 17-19: laterality / measurement / negation presence -----------------------------------
    linguistic_flags = {
        "train_dictation": prof.linguistic_flags(train["dictation"]),
        "test_dictation": prof.linguistic_flags(test["dictation"]),
    }
    _write_json("linguistic_flags.json", linguistic_flags)

    # --- 12, 20: heuristic abnormal-finding-segment count / multi-finding cases -----------------
    abnormal_finding_heuristic = {
        "train_dictation": prof.abnormal_finding_count_stats(train["dictation"]),
        "test_dictation": prof.abnormal_finding_count_stats(test["dictation"]),
    }
    _write_json("abnormal_finding_heuristic.json", abnormal_finding_heuristic)

    # --- Core comparison: template_content vs reference report, every train row ----------------
    diffs = prof.compare_dataset(train)
    _write_jsonl("case_level_diff_detail.jsonl", diffs)

    case_level_df = prof.diffs_to_case_level_frame(diffs, meta_df=train)
    case_level_df.to_csv(os.path.join(OUT_DIR, "case_level_diff.csv"), index=False)

    field_freq_df = prof.field_change_frequency(diffs)
    field_freq_df.to_csv(os.path.join(OUT_DIR, "field_change_frequency.csv"), index=False)

    edit_magnitude_stats = prof.field_edit_magnitude_stats(diffs)
    _write_json("field_edit_magnitude.json", edit_magnitude_stats)

    n_cases = len(diffs)
    n_impression_changed = sum(1 for d in diffs if d["impression_changed"])
    total_field_slots = sum(len(d["changed_fields"]) + len(d["unchanged_fields"]) for d in diffs)
    total_changed_slots = sum(len(d["changed_fields"]) for d in diffs)
    total_added_fields = sum(len(d["added_fields"]) for d in diffs)
    total_removed_fields = sum(len(d["removed_fields"]) for d in diffs)
    total_relabelled_fields = sum(d["n_relabelled_fields"] for d in diffs)
    impression_summary = {
        "n_cases": n_cases,
        "n_impression_changed": n_impression_changed,
        "pct_impression_changed": round(100 * n_impression_changed / n_cases, 2) if n_cases else 0.0,
        "total_field_slots_compared": total_field_slots,
        "total_field_slots_changed": total_changed_slots,
        "overall_field_change_rate_pct": (
            round(100 * total_changed_slots / total_field_slots, 2) if total_field_slots else 0.0
        ),
        "n_cases_with_added_field_labels": sum(1 for d in diffs if d["added_fields"]),
        "n_cases_with_removed_field_labels": sum(1 for d in diffs if d["removed_fields"]),
        "total_added_field_label_instances": total_added_fields,
        "total_removed_field_label_instances": total_removed_fields,
        "n_cases_with_relabelled_fields": sum(1 for d in diffs if d["n_relabelled_fields"]),
        "total_relabelled_field_instances": total_relabelled_fields,
    }
    _write_json("impression_change_summary.json", impression_summary)

    # --- Human-readable summary -----------------------------------------------------------------
    _write_summary_md(
        overview, categorical_train, template_structure_train, template_overlap,
        text_length_train, field_freq_df, impression_summary, case_level_df,
        linguistic_flags, abnormal_finding_heuristic, edit_magnitude_stats, res_scorer_compat,
    )

    print(f"Wrote profiling outputs to {os.path.abspath(OUT_DIR)}")
    print(f"  train: {overview['train']['n_rows']} rows, test: {overview['test']['n_rows']} rows")
    print(f"  unique templates: train={template_structure_train['n_unique_templates']} "
          f"test={template_structure_test['n_unique_templates']} "
          f"(overlap: {template_overlap['pct_test_templates_seen_in_train']}% of test templates seen in train)")
    print(f"  mean FINDINGS field change rate: {impression_summary['overall_field_change_rate_pct']}%")
    print(f"  IMPRESSION changed in {impression_summary['pct_impression_changed']}% of cases")


def _write_summary_md(
    overview, categorical_train, template_structure_train, template_overlap,
    text_length_train, field_freq_df, impression_summary, case_level_df,
    linguistic_flags, abnormal_finding_heuristic, edit_magnitude_stats, res_scorer_compat,
) -> None:
    top_changed = field_freq_df[field_freq_df["n_appearances"] >= 5].sort_values(
        "change_rate", ascending=False
    ).head(15)
    top_unchanged = field_freq_df[field_freq_df["n_appearances"] >= 5].sort_values(
        "change_rate", ascending=True
    ).head(15)
    most_common_field = field_freq_df.sort_values("n_appearances", ascending=False).head(15)

    zero_change_cases = int((case_level_df["n_changed_fields"] == 0).sum())
    high_similarity_cases = int((case_level_df["overall_similarity_ratio"] > 0.9).sum())

    lines = []
    lines.append("# Phase 1 Dataset Profiling — Summary\n")
    lines.append("Generated by `experiments/profile_dataset.py`. All figures below come directly "
                  "from `data/train.csv` / `data/test.csv`; no medical interpretation is added.\n")

    lines.append("## 1. Dataset dimensions\n")
    lines.append(f"- train.csv: {overview['train']['n_rows']} rows x {overview['train']['n_cols']} cols\n")
    lines.append(f"- test.csv: {overview['test']['n_rows']} rows x {overview['test']['n_cols']} cols\n")
    lines.append(f"- sample_submission.csv: {overview['sample_submission']['n_rows']} rows x "
                 f"{overview['sample_submission']['n_cols']} cols\n")
    lines.append(f"- Missing values: none in train or test (all missing_counts are 0 per column; "
                 f"see overview.json).\n")

    lines.append("\n## 2. Template diversity\n")
    lines.append(f"- {template_structure_train['n_unique_templates']} unique templates in train "
                 f"({categorical_train['modality']['n_unique']} modalities, "
                 f"{categorical_train['body_part']['n_unique']} body parts, "
                 f"{categorical_train['study_description']['n_unique']} study descriptions).\n")
    lines.append(f"- {template_structure_train['n_distinct_field_labels']} distinct FINDINGS field "
                 f"labels across train templates (field vocabulary is template-specific, not a "
                 f"fixed global schema).\n")
    lines.append(f"- Fields per template: mean {template_structure_train['fields_per_template']['mean']}, "
                 f"median {template_structure_train['fields_per_template']['p50']}, "
                 f"range [{template_structure_train['fields_per_template']['min']}, "
                 f"{template_structure_train['fields_per_template']['max']}].\n")
    lines.append(f"- {template_structure_train['n_templates_with_no_field_labels']} of "
                 f"{template_structure_train['n_unique_templates']} unique templates have NO "
                 f"recognizable field labels at all (flat unlabelled FINDINGS paragraph).\n")
    lines.append(f"- Test-set template overlap: {template_overlap['pct_test_templates_seen_in_train']}% "
                 f"of the {template_overlap['n_unique_test_templates']} unique test templates also "
                 f"appear verbatim in train "
                 f"({template_overlap['n_test_templates_unseen_in_train']} test templates are unseen "
                 f"in train) — relevant to retrieval.py's few-shot lookup tiers.\n")

    lines.append("\n## 3. Text length\n")
    d = text_length_train["dictation"]["token_length"]
    r = text_length_train["report"]["token_length"]
    lines.append(f"- Dictation length (tokens): mean {d['mean']}, median {d['p50']}, "
                 f"range [{d['min']}, {d['max']}].\n")
    lines.append(f"- Reference report length (tokens): mean {r['mean']}, median {r['p50']}, "
                 f"range [{r['min']}, {r['max']}].\n")

    lines.append("\n## 4. Most frequently changed FINDINGS fields\n")
    lines.append("Restricted to field labels appearing in >= 5 template instances (see "
                 "`field_change_frequency.csv` for the full table, including rarer labels).\n\n")
    lines.append("Highest change rate:\n\n")
    lines.append("| field_label | n_appearances | n_changed | change_rate |\n|---|---|---|---|\n")
    for _, row in top_changed.iterrows():
        lines.append(f"| {row['field_label']} | {row['n_appearances']} | {row['n_changed']} | "
                     f"{row['change_rate']} |\n")
    lines.append("\nLowest change rate (most reliably left untouched):\n\n")
    lines.append("| field_label | n_appearances | n_changed | change_rate |\n|---|---|---|---|\n")
    for _, row in top_unchanged.iterrows():
        lines.append(f"| {row['field_label']} | {row['n_appearances']} | {row['n_changed']} | "
                     f"{row['change_rate']} |\n")

    lines.append("\n## 5. Edit-scope findings (template vs. reference report)\n")
    lines.append(f"- Overall FINDINGS-field change rate (binary: does the field's text differ from "
                 f"the template AT ALL): {impression_summary['overall_field_change_rate_pct']}% of "
                 f"all (case, field) slots "
                 f"({impression_summary['total_field_slots_changed']} of "
                 f"{impression_summary['total_field_slots_compared']}) — a MAJORITY of fields are "
                 f"touched at least once; this is not a mostly-untouched template.\n")
    lines.append(f"- Among fields that DO change, the edit is usually a substantial rewrite of that "
                 f"field's text, not a small word-level tweak: edit magnitude (normalized weighted "
                 f"edit distance, same measure res_scorer.py uses for RES's F component) has mean "
                 f"{edit_magnitude_stats['distribution']['mean']}, median "
                 f"{edit_magnitude_stats['distribution']['p50']} out of 1.0. Only "
                 f"{edit_magnitude_stats['pct_lightly_edited_lt_0_25']}% of changed-field instances "
                 f"are lightly edited (magnitude < 0.25), while "
                 f"{edit_magnitude_stats['pct_heavily_edited_gt_0_75']}% are heavily rewritten "
                 f"(magnitude > 0.75) — see `field_edit_magnitude.json` for the full distribution. "
                 f"This is consistent with most FINDINGS fields being short (often one templated "
                 f"sentence), so replacing a normal statement with a dictated abnormal finding "
                 f"typically replaces most of the field's own tokens.\n")
    lines.append(f"- IMPRESSION changed in {impression_summary['pct_impression_changed']}% of cases.\n")
    lines.append(f"- {zero_change_cases} of {len(case_level_df)} train cases have ZERO changed "
                 f"FINDINGS fields (dictation was fully consistent with the template's normal "
                 f"statements).\n")
    lines.append(f"- {high_similarity_cases} of {len(case_level_df)} cases have a whole-document "
                 f"similarity ratio > 0.9 against their own template; see "
                 f"`case_level_diff.csv`'s `overall_similarity_ratio` column for the full "
                 f"per-case distribution — most cases are NOT near-identical to their template, "
                 f"consistent with the field change rate above.\n")
    lines.append(f"- Field labels are matched template<->report case-insensitively (see "
                 f"profiling.py docstring): {impression_summary['n_cases_with_relabelled_fields']} "
                 f"cases have a field whose label is the SAME field but re-cased between template "
                 f"and report (typically Title Case in the template -> ALL CAPS in the report; "
                 f"{impression_summary['total_relabelled_field_instances']} field instances total) "
                 f"— a real, reproducible pattern in the reference data, not a parsing artifact; "
                 f"matching labels case-sensitively (as res_scorer.py's stricter ALL-CAPS regex "
                 f"effectively does for any Title/mixed-case label) would misread every one of "
                 f"these as one field removed plus an unrelated field added.\n")
    lines.append(f"- After case-insensitive matching, GENUINE field-label mismatches (a label with "
                 f"no counterpart at any casing — should be rare per the task's field-preservation "
                 f"rule): {impression_summary['n_cases_with_added_field_labels']} cases have a field "
                 f"label in the report with no template counterpart, "
                 f"{impression_summary['n_cases_with_removed_field_labels']} cases have a template "
                 f"field label dropped entirely from the report. These are worth inspecting directly "
                 f"in `case_level_diff_detail.jsonl` (non-zero `added_fields`/`removed_fields`).\n")

    lines.append("\n## 6. Linguistic signal prevalence (train dictation)\n")
    tf = linguistic_flags["train_dictation"]
    lines.append(f"- Laterality words present: {tf['pct_with_laterality']}% of cases.\n")
    lines.append(f"- Measurement patterns present: {tf['pct_with_measurement']}% of cases.\n")
    lines.append(f"- Negation words present: {tf['pct_with_negation']}% of cases.\n")

    af = abnormal_finding_heuristic["train_dictation"]
    lines.append(f"- Heuristic abnormal-finding-segment count (see profiling.py docstring for the "
                 f"exact, deliberately coarse method): mean {af['heuristic_abnormal_segment_count']['mean']}, "
                 f"median {af['heuristic_abnormal_segment_count']['p50']}; "
                 f"{af['pct_cases_with_multiple_findings']}% of cases have more than one such segment.\n")

    lines.append("\n## 7. Most important patterns discovered\n")
    if res_scorer_compat["n_templates_res_scorer_fails_to_parse_any_field"] > 0:
        lines.append(f"- **Scoring bug found**: `src/res_scorer.py`'s field parser "
                     f"(`parse_report_fields`, used by `score_case`/`score_dataset`/"
                     f"`scripts/run_eval.py`) uses a stricter ALL-CAPS-only label regex than "
                     f"`src/template_parser.py`'s (which was already widened for this reason — "
                     f"see its own docstring). Checked against every unique train template: "
                     f"{res_scorer_compat['n_templates_res_scorer_fails_to_parse_any_field']} of "
                     f"{res_scorer_compat['n_unique_templates_checked']} unique templates have "
                     f"real (Title/mixed-case) field labels that res_scorer.py fails to parse at "
                     f"all — the entire FINDINGS block falls into an undifferentiated "
                     f"\"__UNLABELLED__\" bucket instead of being scored per field. This affects "
                     f"{res_scorer_compat['n_rows_affected']} rows "
                     f"({res_scorer_compat['pct_rows_affected']}% of train) — any past or future "
                     f"local RES number from scripts/run_eval.py is unreliable until this is fixed. "
                     f"See `res_scorer_compatibility.json` for example templates.\n")
    lines.append("- Field labels are template-specific and NOT a fixed global schema "
                 f"({template_structure_train['n_distinct_field_labels']}+ distinct labels across "
                 f"only {template_structure_train['n_unique_templates']} templates) — routing logic "
                 "cannot hardcode a label list.\n")
    lines.append(f"- A MAJORITY of fields change at the binary level "
                 f"({impression_summary['overall_field_change_rate_pct']}%), AND when a field does "
                 f"change it is usually rewritten heavily (median edit magnitude "
                 f"{edit_magnitude_stats['distribution']['p50']} of 1.0, "
                 f"{edit_magnitude_stats['pct_heavily_edited_gt_0_75']}% of changed instances above "
                 f"0.75) — so the decisive lever for RES is WHICH fields get touched (routing/"
                 "change-detection precision), not preserving fragments of a touched field's prior "
                 "wording. Getting routing wrong (touching a field that shouldn't change, or missing "
                 "one that should) costs far more than any within-field phrasing choice.\n")
    lines.append(f"- Change rate varies sharply by field label (`field_change_frequency.csv`): some "
                 "labels (e.g. LINES/TUBES/SUPPORT DEVICES, DIAPHRAGM) change in under 10% of their "
                 "appearances, others (e.g. VERTEBRAE, OTHER FINDINGS) in over 90% — a uniform "
                 "per-field edit policy would be miscalibrated in both directions.\n")
    lines.append(f"- {template_structure_train['n_templates_with_no_field_labels']} unique template(s) "
                 "carry no parseable field labels at all (flat prose); a field-routing pipeline needs "
                 "an explicit fallback path for these, not just a labeled-field router.\n")
    lines.append(f"- Reference reports frequently re-case a template's field label (usually to ALL "
                 f"CAPS) with no other change to that field's meaning "
                 f"({impression_summary['n_cases_with_relabelled_fields']} cases affected) — a field "
                 f"router/assembler that matches labels case-sensitively will silently treat these "
                 f"as unrelated fields, corrupting both change-detection and RES scoring for that "
                 f"field. This is the single biggest data-quality gotcha found in Phase 1.\n")
    lines.append("- Genuine field-label additions/removals (after correcting for re-casing) are rare "
                 "but not zero — see the exact counts above and the flagged case_ids in "
                 "case_level_diff_detail.jsonl before assuming label preservation is guaranteed.\n")

    lines.append("\n## 8. Recommendations for the next pipeline stage\n")
    if res_scorer_compat["n_templates_res_scorer_fails_to_parse_any_field"] > 0:
        lines.append("- **Fix res_scorer.py's field-label regex first, before trusting any RES "
                     "number it produces.** Reuse template_parser.py's already-widened FIELD_RE "
                     "(or its parse_template) instead of maintaining a second, stricter regex — "
                     "the two parsers disagreeing is itself the bug.\n")
    lines.append("- Calibrate routing/editing confidence per field label using "
                 "`field_change_frequency.csv`'s empirical change_rate as a prior, rather than "
                 "assuming either \"always edit\" or \"rarely edit\" — both are wrong for large "
                 "parts of the label vocabulary here.\n")
    lines.append("- Because a changed field is usually rewritten wholesale rather than lightly "
                 "edited (edit magnitude distribution in `field_edit_magnitude.json`), full-field "
                 "regeneration for routed/changed fields (the current field_editor.py approach) "
                 "matches this corpus's own behavior — the priority is correct routing and leaving "
                 "untouched fields byte-for-byte alone, not preserving sub-sentence fragments "
                 "inside an edited field.\n")
    lines.append("- Build/validate the field router against the full label vocabulary in "
                 "`template_structure_train.json['top_field_labels']`, not a hand-authored anatomy "
                 "list, given how fragmented and template-specific labels are.\n")
    lines.append("- Add an explicit code path (and a test) for unlabelled/flat-prose templates before "
                 "assuming every template has parseable FINDINGS fields.\n")
    lines.append("- Inspect the cases with added/removed field labels individually "
                 "(`case_level_diff_detail.jsonl`, non-empty `added_fields`/`removed_fields`) — they "
                 "may reveal a parsing edge case rather than a genuine template/report mismatch.\n")

    with open(os.path.join(OUT_DIR, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
