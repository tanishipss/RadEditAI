"""
Retrieval module — FR10 (TRD.md §4.2).

Three-tier lookup for a case's few-shot examples: exact template_content match, else same
(modality, body_part), else same modality. Whichever tier has at least one candidate (after
excluding the case's own case_id, so evaluating on a held-out train.csv fold never leaks a case
as its own example) is used exclusively — ties within a tier are broken by sorting on case_id
(TRD.md §8: no randomness), and up to `k` examples are returned from that tier only.

Each returned example also carries a `diff_summary` (TRD.md §5.2): a FINDINGS block where fields
whose text is unchanged from that example's own template are abbreviated to
"...unchanged, copied verbatim..." so long templates don't blow up prompt size in the modules
that consume these examples.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from res_scorer import normalize_text
from template_parser import UNLABELLED_LABEL, fields_as_dict, parse_template


def build_index(train_df: pd.DataFrame) -> dict:
    by_template: dict[str, list] = defaultdict(list)
    by_modality_body_part: dict[tuple, list] = defaultdict(list)
    by_modality: dict[str, list] = defaultdict(list)

    for idx, row in train_df.iterrows():
        by_template[row["template_content"]].append(idx)
        by_modality_body_part[(row["modality"], row["body_part"])].append(idx)
        by_modality[row["modality"]].append(idx)

    return {
        "by_template": dict(by_template),
        "by_modality_body_part": dict(by_modality_body_part),
        "by_modality": dict(by_modality),
    }


def _build_diff_summary(template_content: str, report: str) -> str:
    tmpl_fields = fields_as_dict(parse_template(template_content))
    rep_parsed = parse_template(report)

    lines = ["FINDINGS:"]
    for field in rep_parsed["findings_fields"]:
        label, rep_text = field["label"], field["content"]
        tmpl_text = tmpl_fields.get(label, "")
        changed = normalize_text(rep_text) != normalize_text(tmpl_text)
        if label == UNLABELLED_LABEL:
            lines.append(rep_text if changed else "...unchanged, copied verbatim...")
        else:
            lines.append(f"{label}: {rep_text if changed else '...unchanged, copied verbatim...'}")

    lines.append("")
    lines.append("IMPRESSION:")
    lines.append(rep_parsed["impression"].strip())
    return "\n".join(lines)


def _sorted_candidates(idx_list: list, train_df: pd.DataFrame, exclude_case_id) -> list:
    pairs = [
        (train_df.loc[i, "case_id"], i)
        for i in idx_list
        if train_df.loc[i, "case_id"] != exclude_case_id
    ]
    pairs.sort(key=lambda p: p[0])
    return [i for _, i in pairs]


def get_few_shot_examples(case: pd.Series, train_df: pd.DataFrame, index: dict, k: int = 2) -> list[dict]:
    exclude_case_id = case.get("case_id")

    tier = None
    candidate_idxs: list = []

    exact = _sorted_candidates(index["by_template"].get(case["template_content"], []), train_df, exclude_case_id)
    if exact:
        tier, candidate_idxs = "exact_template", exact[:k]
    else:
        mbp = _sorted_candidates(
            index["by_modality_body_part"].get((case["modality"], case["body_part"]), []),
            train_df,
            exclude_case_id,
        )
        if mbp:
            tier, candidate_idxs = "modality_body_part", mbp[:k]
        else:
            mod = _sorted_candidates(index["by_modality"].get(case["modality"], []), train_df, exclude_case_id)
            if mod:
                tier, candidate_idxs = "modality", mod[:k]

    examples = []
    for idx in candidate_idxs:
        row = train_df.loc[idx]
        examples.append(
            {
                "case_id": row["case_id"],
                "template_content": row["template_content"],
                "dictation": row["dictation"],
                "report": row["report"],
                "diff_summary": _build_diff_summary(row["template_content"], row["report"]),
                "tier": tier,
            }
        )
    return examples
