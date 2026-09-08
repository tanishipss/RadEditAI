"""
scripts/run_submission.py — data/test.csv -> data/submission.csv (TRD.md §6).

WARNING: makes real Groq API calls for every row processed (~2-5 calls/case). Use --limit for a
cheap smoke test before running the full 132-row test set.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from groq_client import print_usage_summary
from pipeline import build_pipeline_context, run_case


def main(limit: int | None = None) -> None:
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    if limit:
        test_df = test_df.head(limit)

    context = build_pipeline_context(train_df)

    rows_out = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        print(f"[{i + 1}/{len(test_df)}] {row['case_id']}")
        result = run_case(row, context)
        rows_out.append({"case_id": row["case_id"], "report": result["report"]})

    out_df = pd.DataFrame(rows_out, columns=["case_id", "report"])
    out_path = os.path.join("data", "submission.csv")
    # pandas' default QUOTE_MINIMAL already quotes any field containing a newline, comma, or
    # quote character — correct per-spec CSV quoting for multi-line report text; no need to
    # force QUOTE_ALL.
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(out_df)} rows to {out_path}")

    print()
    print_usage_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None,
        help="process only the first N rows of test.csv (for a quick, cheap smoke test)",
    )
    args = parser.parse_args()
    main(limit=args.limit)
