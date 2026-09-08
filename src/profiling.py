"""
Phase 1 dataset profiling — pure, read-only analysis over train.csv / test.csv /
sample_submission.csv. No LLM calls anywhere in this module; nothing here mutates the source
CSVs. Every function returns plain dict/list/DataFrame values — this module never writes to
disk itself; that is left to the caller (experiments/profile_dataset.py), which keeps this
module reusable from a notebook, a test, or another script without side effects.

Field-level template<->report comparison reuses `template_parser.parse_template` (not
`res_scorer.parse_report_fields`): template_parser's widened label regex is the one verified
against every unique template_content in train.csv (see its own docstring), whereas
res_scorer's regex assumes ALL-CAPS labels and exists specifically for RES scoring, a separate
concern from this profiling pass. Text equality/diffing reuses `res_scorer.normalize_text` so
"unchanged" here means the same thing the scorer itself considers unchanged (case/whitespace/
list-marker/hyphen-chain insensitive), not a naive string `==`.
"""

from __future__ import annotations

import difflib
import math
import re
from collections import Counter
from typing import Any

import pandas as pd

from res_scorer import (
    CRITICAL_WORDS, FUNCTION_WORDS, NEGATION_WORDS, LATERALITY_WORDS,
    normalize_text, weighted_edit_distance,
)
from res_scorer import parse_report_fields as res_scorer_parse_fields
from template_parser import UNLABELLED_LABEL, fields_as_dict, parse_template

UNLABELLED_KEY = UNLABELLED_LABEL  # kept as an alias: this module's own public name for the sentinel

_MEASUREMENT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:mm|cm|ml|cc|hu)\b", flags=re.IGNORECASE)
# A segment counts as a "blanket normal / negative" statement (not a candidate abnormal finding)
# if, after light normalization, it opens with a negation word or a common normal/unremarkable
# phrase. This is a coarse heuristic over sentence-like segments, not a clinical NLP model — see
# `heuristic_abnormal_segment_count`'s docstring for exactly what it does and does not claim.
_NORMAL_OPENING_RE = re.compile(
    r"^(no|not|without|negative for|unremarkable|normal|nor|denies|absent|"
    r"no acute|no evidence)\b"
)
_SEGMENT_SPLIT_RE = re.compile(r"[.\n;]+")


# ---------------------------------------------------------------------------
# 1-3: dataset shape, dtypes, missing values
# ---------------------------------------------------------------------------

def dataset_overview(df: pd.DataFrame) -> dict[str, Any]:
    """Dimensions, column dtypes, and missing-value counts for one dataframe."""
    n_rows = len(df)
    missing_counts = df.isna().sum().to_dict()
    return {
        "n_rows": n_rows,
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": {col: int(n) for col, n in missing_counts.items()},
        "missing_pct": {
            col: round(100 * n / n_rows, 2) if n_rows else 0.0
            for col, n in missing_counts.items()
        },
    }


# ---------------------------------------------------------------------------
# 4-6: categorical columns (modality, body_part, study_description, age band, sex)
# ---------------------------------------------------------------------------

def categorical_summary(df: pd.DataFrame, columns: list[str], top_n: int = 50) -> dict[str, Any]:
    """Per-column {n_unique, value_counts (top_n, most frequent first)}."""
    out: dict[str, Any] = {}
    for col in columns:
        if col not in df.columns:
            continue
        counts = df[col].value_counts()
        out[col] = {
            "n_unique": int(df[col].nunique()),
            "value_counts": {str(k): int(v) for k, v in counts.head(top_n).items()},
        }
    return out


# ---------------------------------------------------------------------------
# Generic numeric-distribution helper (reused for lengths, finding counts, etc.)
# ---------------------------------------------------------------------------

def distribution_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    s = pd.Series(values, dtype="float64")
    return {
        "n": int(s.count()),
        "mean": round(float(s.mean()), 2),
        "std": round(float(s.std(ddof=0)), 2),
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "max": float(s.max()),
    }


# ---------------------------------------------------------------------------
# 10-11: dictation / report length distributions
# ---------------------------------------------------------------------------

def text_length_stats(series: pd.Series) -> dict[str, Any]:
    """Character-length and (normalized) token-length distributions for a text column."""
    texts = series.fillna("").astype(str)
    char_lengths = texts.str.len().tolist()
    token_lengths = [len(normalize_text(t)) for t in texts]
    return {
        "char_length": distribution_stats(char_lengths),
        "token_length": distribution_stats(token_lengths),
    }


# ---------------------------------------------------------------------------
# 7-9: template structure — unique templates, fields per template, field label frequency
# ---------------------------------------------------------------------------

def template_structure_summary(df: pd.DataFrame, template_col: str = "template_content") -> dict[str, Any]:
    unique_templates = df[template_col].drop_duplicates().tolist()

    n_fields_per_template = []
    label_frequency: Counter = Counter()  # across unique templates
    n_unlabelled_templates = 0

    for tmpl in unique_templates:
        fields = fields_as_dict(parse_template(tmpl))
        real_labels = [label for label in fields if label != UNLABELLED_KEY]
        n_fields_per_template.append(len(real_labels))
        if not real_labels:
            n_unlabelled_templates += 1
        for label in real_labels:
            label_frequency[label] += 1

    return {
        "n_unique_templates": len(unique_templates),
        "n_templates_with_no_field_labels": n_unlabelled_templates,
        "fields_per_template": distribution_stats(n_fields_per_template),
        "n_distinct_field_labels": len(label_frequency),
        "top_field_labels": label_frequency.most_common(40),
    }


# ---------------------------------------------------------------------------
# 16: common finding/anatomy terms (dictation vocabulary, function words excluded)
# ---------------------------------------------------------------------------

def common_terms(series: pd.Series, top_n: int = 60) -> list[tuple[str, int]]:
    counter: Counter = Counter()
    for text in series.fillna("").astype(str):
        for tok in normalize_text(text):
            if tok in FUNCTION_WORDS or tok in CRITICAL_WORDS:
                continue
            if len(tok) < 3 or tok.replace(".", "", 1).isdigit():
                continue
            counter[tok] += 1
    return counter.most_common(top_n)


# ---------------------------------------------------------------------------
# 17-19: laterality / measurement / negation presence per case
# ---------------------------------------------------------------------------

def linguistic_flags(series: pd.Series) -> dict[str, Any]:
    texts = series.fillna("").astype(str)
    n = len(texts)
    has_laterality = texts.apply(lambda t: any(w in LATERALITY_WORDS for w in normalize_text(t)))
    has_negation = texts.apply(lambda t: any(w in NEGATION_WORDS for w in normalize_text(t)))
    has_measurement = texts.apply(lambda t: bool(_MEASUREMENT_RE.search(t)))
    return {
        "n_cases": n,
        "pct_with_laterality": round(100 * has_laterality.sum() / n, 2) if n else 0.0,
        "pct_with_measurement": round(100 * has_measurement.sum() / n, 2) if n else 0.0,
        "pct_with_negation": round(100 * has_negation.sum() / n, 2) if n else 0.0,
    }


# ---------------------------------------------------------------------------
# 12, 20: heuristic abnormal-finding segment count per dictation
# ---------------------------------------------------------------------------

def heuristic_abnormal_segment_count(dictation: str) -> int:
    """Approximate count of dictated finding statements that read as an abnormal/positive
    finding rather than a blanket normal/negative statement.

    Method (deliberately simple, no clinical judgment): split the dictation on sentence-ish
    boundaries (. ; newline), and count non-empty segments whose normalized text does NOT open
    with a negation/normal-statement marker (`_NORMAL_OPENING_RE`). This is a lexical heuristic
    over sentence structure, not an extraction of actual clinical findings — it will overcount
    (e.g. a single sentence describing two organs counts once) and undercount (e.g. a negative
    finding embedded mid-sentence after a comma is not detected as "normal"). It exists only to
    give a rough, reproducible proxy for "how many things did the radiologist dictate", useful
    for corpus-level distribution shape, not for any single case's ground truth.
    """
    text = "" if dictation is None or (isinstance(dictation, float) and math.isnan(dictation)) else str(dictation)
    segments = [s.strip() for s in _SEGMENT_SPLIT_RE.split(text) if s.strip()]
    count = 0
    for seg in segments:
        if _NORMAL_OPENING_RE.match(seg.lower()):
            continue
        count += 1
    return count


def abnormal_finding_count_stats(series: pd.Series) -> dict[str, Any]:
    counts = series.fillna("").astype(str).apply(heuristic_abnormal_segment_count).tolist()
    multi_finding_pct = (
        round(100 * sum(1 for c in counts if c > 1) / len(counts), 2) if counts else 0.0
    )
    return {
        "heuristic_abnormal_segment_count": distribution_stats(counts),
        "pct_cases_with_multiple_findings": multi_finding_pct,
    }


# ---------------------------------------------------------------------------
# Template <-> report field-level comparison (the core of Phase 1)
# ---------------------------------------------------------------------------

def _tokens_equal(a: str, b: str) -> bool:
    return normalize_text(a) == normalize_text(b)


def _classify_field_diff(template_text: str, report_text: str) -> dict[str, Any]:
    old_tokens = normalize_text(template_text)
    new_tokens = normalize_text(report_text)
    sm = difflib.SequenceMatcher(a=old_tokens, b=new_tokens, autojunk=False)

    added: list[str] = []
    removed: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "insert":
            added.extend(new_tokens[j1:j2])
        elif tag == "delete":
            removed.extend(old_tokens[i1:i2])
        elif tag == "replace":
            removed.extend(old_tokens[i1:i2])
            added.extend(new_tokens[j1:j2])

    if added and removed:
        category = "modified"
    elif added:
        category = "added"
    elif removed:
        category = "removed"
    else:
        category = "modified"  # changed under normalize_text but opcodes found nothing (edge case)

    # Same weighted-edit-distance function res_scorer.py uses for RES's own F component (here
    # unweighted by field importance) — a normalized [0, 1] measure of how MUCH of the field's
    # content changed, distinct from `category`, which only says WHICH kind of change occurred.
    # A field can be "changed" (differs at all) while this magnitude is small (one word edited).
    edit_magnitude = weighted_edit_distance(old_tokens, new_tokens)

    return {
        "category": category, "added_tokens": added, "removed_tokens": removed,
        "edit_magnitude": round(edit_magnitude, 4),
    }


def compare_template_to_report(case_id: Any, template_content: str, report: str) -> dict[str, Any]:
    """Field-level diff of one case's template_content against its reference report.

    Field labels are matched case-insensitively: the reference reports in this corpus frequently
    re-case a template's own label (e.g. template "Bones:" -> report "BONES:") for the same
    field, with no other change intended. Matching by exact string would misclassify every one
    of those as one field "removed" plus a different field "added", hiding whether that field's
    CONTENT changed at all. A label with no match at any casing is genuinely added/removed.

    Returns changed_fields / unchanged_fields (labels present, at some casing, on both sides,
    split by whether their text changed), added_fields / removed_fields (labels genuinely present
    on only one side — expected to be rare, since the task requires preserving template field
    labels; a non-zero count here is a notable finding in its own right), a
    relabelled_fields count (same field, casing changed only), an impression_changed flag, and a
    field_diffs list giving each changed field's category (added/removed/modified content) plus
    the actual added/removed tokens.
    """
    tmpl_parsed = parse_template(template_content)
    rep_parsed = parse_template(report)
    tmpl_fields = fields_as_dict(tmpl_parsed)
    rep_fields = fields_as_dict(rep_parsed)
    tmpl_impression = tmpl_parsed["impression"]
    rep_impression = rep_parsed["impression"]

    tmpl_labels = [l for l in tmpl_fields if l != UNLABELLED_KEY]
    rep_labels = [l for l in rep_fields if l != UNLABELLED_KEY]

    # First-seen-wins: collisions where two labels differ only in case are not expected in
    # practice (verified: no template or report in train.csv has two distinct field labels
    # differing only by casing) and would be a genuine ambiguity, not something to silently drop.
    rep_by_lower: dict[str, str] = {}
    for label in rep_labels:
        rep_by_lower.setdefault(label.lower(), label)

    changed_fields: list[str] = []
    unchanged_fields: list[str] = []
    removed_fields: list[str] = []
    matched_rep_labels: set[str] = set()
    n_relabelled = 0
    field_diffs: list[dict[str, Any]] = []

    for label in tmpl_labels:
        rep_label = rep_by_lower.get(label.lower())
        if rep_label is None:
            removed_fields.append(label)
            continue
        matched_rep_labels.add(rep_label)
        if rep_label != label:
            n_relabelled += 1
        if _tokens_equal(tmpl_fields[label], rep_fields[rep_label]):
            unchanged_fields.append(label)
        else:
            changed_fields.append(label)
            diff = _classify_field_diff(tmpl_fields[label], rep_fields[rep_label])
            field_diffs.append({"field": label, **diff})

    added_fields = sorted(l for l in rep_labels if l not in matched_rep_labels)
    impression_changed = not _tokens_equal(tmpl_impression, rep_impression)

    overall_similarity = difflib.SequenceMatcher(
        a=str(template_content), b=str(report), autojunk=False
    ).ratio()

    return {
        "case_id": case_id,
        "n_template_fields": len(tmpl_labels),
        "n_report_fields": len(rep_labels),
        "changed_fields": changed_fields,
        "unchanged_fields": unchanged_fields,
        "added_fields": added_fields,
        "removed_fields": removed_fields,
        "n_relabelled_fields": n_relabelled,
        "impression_changed": impression_changed,
        "overall_similarity_ratio": round(overall_similarity, 4),
        "field_diffs": field_diffs,
    }


def compare_dataset(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Run compare_template_to_report over every row of a dataframe with a `report` column."""
    return [
        compare_template_to_report(row["case_id"], row["template_content"], row["report"])
        for _, row in df.iterrows()
    ]


def field_change_frequency(diffs: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate per-field-label change frequency across every case's diff record.

    A field's `n_changed` includes both "changed_fields" (text differs) and "removed_fields"
    (label dropped entirely from the report) — both represent the template's own field content
    not surviving unchanged. `n_appearances` is how many template instances carried that label.
    """
    appearances: Counter = Counter()
    changed: Counter = Counter()

    for d in diffs:
        for label in d["changed_fields"]:
            appearances[label] += 1
            changed[label] += 1
        for label in d["unchanged_fields"]:
            appearances[label] += 1
        for label in d["removed_fields"]:
            appearances[label] += 1
            changed[label] += 1

    rows = []
    for label, n_appear in appearances.items():
        n_changed = changed.get(label, 0)
        rows.append({
            "field_label": label,
            "n_appearances": n_appear,
            "n_changed": n_changed,
            "n_unchanged": n_appear - n_changed,
            "change_rate": round(n_changed / n_appear, 4) if n_appear else 0.0,
        })

    out = pd.DataFrame(rows).sort_values("n_appearances", ascending=False).reset_index(drop=True)
    return out


def field_edit_magnitude_stats(diffs: list[dict[str, Any]]) -> dict[str, Any]:
    """Distribution of `edit_magnitude` (weighted_edit_distance, in [0, 1]) across every CHANGED
    field only — i.e. conditional on a field having been touched at all, how much of it changed.
    Distinguishes "most fields that change are lightly edited" from "most fields that change are
    rewritten wholesale", which the changed/unchanged binary count alone cannot."""
    magnitudes = [fd["edit_magnitude"] for d in diffs for fd in d["field_diffs"]]
    return {
        "n_changed_field_instances": len(magnitudes),
        "distribution": distribution_stats(magnitudes),
        "pct_lightly_edited_lt_0_25": (
            round(100 * sum(1 for m in magnitudes if m < 0.25) / len(magnitudes), 2)
            if magnitudes else 0.0
        ),
        "pct_heavily_edited_gt_0_75": (
            round(100 * sum(1 for m in magnitudes if m > 0.75) / len(magnitudes), 2)
            if magnitudes else 0.0
        ),
    }


def res_scorer_compatibility_check(df: pd.DataFrame, template_col: str = "template_content") -> dict[str, Any]:
    """Cross-checks template_parser.parse_template (this module's, and the rest of the
    pipeline's, field parser) against res_scorer.parse_report_fields (a SEPARATE, stricter
    ALL-CAPS-only regex used only for RES scoring) on every unique template_content.

    template_parser.py's own docstring documents that its regex was deliberately widened because
    many real templates use Title/mixed-case field labels; res_scorer.py's regex was not updated
    to match. This function quantifies the consequence: for a template whose labels aren't
    ALL-CAPS, res_scorer.parse_report_fields matches zero of them and files the entire FINDINGS
    block under "__UNLABELLED__" — meaning score_case()/score_dataset() (used by
    scripts/run_eval.py for local RES cross-validation) silently mis-scores every field of every
    such case as one giant unexpected/unlabelled blob rather than per-field.
    """
    unique_templates = df[template_col].drop_duplicates().tolist()
    n_mismatched = 0
    n_rows_affected = 0
    row_counts = df[template_col].value_counts()
    example_templates: list[str] = []

    for tmpl in unique_templates:
        tp_fields = fields_as_dict(parse_template(tmpl))
        tp_n_real = len([l for l in tp_fields if l != UNLABELLED_KEY])
        if tp_n_real == 0:
            continue  # genuinely unlabelled template, not a res_scorer regex mismatch

        rs_fields, _ = res_scorer_parse_fields(tmpl)
        rs_n_real = len([l for l in rs_fields if l != "__UNLABELLED__"])
        if rs_n_real == 0:
            n_mismatched += 1
            n_rows_affected += int(row_counts.get(tmpl, 0))
            if len(example_templates) < 5:
                example_templates.append(tmpl[:200])

    return {
        "n_unique_templates_checked": len(unique_templates),
        "n_templates_res_scorer_fails_to_parse_any_field": n_mismatched,
        "n_rows_affected": n_rows_affected,
        "pct_rows_affected": round(100 * n_rows_affected / len(df), 2) if len(df) else 0.0,
        "example_mismatched_template_prefixes": example_templates,
    }


def diffs_to_case_level_frame(diffs: list[dict[str, Any]], meta_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Flatten compare_dataset()'s per-case dicts into one scalar-columned row per case_id,
    suitable for a CSV. Full field-level detail (added/removed tokens) is NOT included here —
    see the companion JSONL detail file for that."""
    rows = []
    for d in diffs:
        n_modified = sum(1 for fd in d["field_diffs"] if fd["category"] == "modified")
        n_content_added = sum(1 for fd in d["field_diffs"] if fd["category"] == "added")
        n_content_removed = sum(1 for fd in d["field_diffs"] if fd["category"] == "removed")
        rows.append({
            "case_id": d["case_id"],
            "n_template_fields": d["n_template_fields"],
            "n_report_fields": d["n_report_fields"],
            "n_changed_fields": len(d["changed_fields"]),
            "n_unchanged_fields": len(d["unchanged_fields"]),
            "n_added_fields": len(d["added_fields"]),
            "n_removed_fields": len(d["removed_fields"]),
            "n_content_modified": n_modified,
            "n_content_added": n_content_added,
            "n_content_removed": n_content_removed,
            "n_relabelled_fields": d["n_relabelled_fields"],
            "impression_changed": d["impression_changed"],
            "overall_similarity_ratio": d["overall_similarity_ratio"],
        })
    out = pd.DataFrame(rows)
    if meta_df is not None:
        out = out.merge(
            meta_df[["case_id", "modality", "body_part"]], on="case_id", how="left"
        )
    return out
