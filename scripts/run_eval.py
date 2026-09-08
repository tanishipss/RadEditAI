"""
scripts/run_eval.py — 5-fold cross-validation on data/train.csv (TRD.md §6, PRD.md §7).

Manual stratified-by-modality k-fold split rather than scikit-learn's StratifiedKFold: TRD.md §8
explicitly caps dependencies at "pandas, pydantic, groq" (for Kaggle-notebook portability), so
adding scikit-learn just for this split isn't warranted. Each modality's rows are shuffled with a
fixed random_state and distributed round-robin across folds — reproducible, no randomness leak
(TRD.md §8).

WARNING: a full run makes real Groq API calls for every one of the 636 rows in train.csv (each
case costs ~2-5 calls: extraction, batched field-edit, impression, validate, +1 diagnostic
validate here, +repair if needed) — potentially 1500-3000+ calls total, and there's no retry/
backoff yet (that's Prompt 11). Use --limit for a cheap smoke test before committing to a full
run.
"""

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from groq_client import print_usage_summary
from pipeline import build_pipeline_context, run_case
from res_scorer import score_case
from validator import validate

N_FOLDS = 5
RANDOM_STATE = 42
ISSUE_KINDS = [
    "unsupported", "missing", "wrong_field", "negation", "laterality",
    "measurement", "unnecessary_edit", "impression_error",
]


def stratified_kfold_indices(df: pd.DataFrame, n_folds: int, random_state: int) -> list[list[int]]:
    rng = np.random.default_rng(random_state)
    fold_indices: list[list[int]] = [[] for _ in range(n_folds)]
    for _, group in df.groupby("modality"):
        idxs = group.index.to_numpy().copy()
        rng.shuffle(idxs)
        for i, idx in enumerate(idxs):
            fold_indices[i % n_folds].append(idx)
    return fold_indices


def main(limit: int | None = None) -> None:
    df = pd.read_csv("data/train.csv")
    if limit:
        df = df.head(limit).reset_index(drop=True)

    folds = stratified_kfold_indices(df, N_FOLDS, RANDOM_STATE)

    issue_counter: Counter = Counter()
    fold_mean_res = []
    all_res_scores = []

    for fold_idx in range(N_FOLDS):
        held_out_idx = folds[fold_idx]
        if not held_out_idx:
            continue
        train_idx = [i for f in range(N_FOLDS) if f != fold_idx for i in folds[f]]

        fold_train_df = df.loc[train_idx].reset_index(drop=True)
        fold_held_out_df = df.loc[held_out_idx]

        context = build_pipeline_context(fold_train_df)

        fold_scores = []
        for _, row in fold_held_out_df.iterrows():
            result = run_case(row, context)
            report = result["report"]

            score = score_case(row["report"], report, row["template_content"])
            fold_scores.append(score["RES_case"])
            all_res_scores.append(score["RES_case"])

            # Diagnostic-only validation over the FINAL report, purely for the error-analysis
            # table below. Independent of whatever validation/repair already ran inside
            # run_case; must never trigger another repair loop here.
            diag = validate(row, report, result["extraction"], result["routings"])
            for issue in diag.issues:
                issue_counter[issue.kind] += 1

        fold_mean = sum(fold_scores) / len(fold_scores) if fold_scores else 0.0
        fold_mean_res.append(fold_mean)
        print(f"Fold {fold_idx}: n={len(fold_scores)}  mean RES = {fold_mean:.4f}")

    overall_mean = sum(all_res_scores) / len(all_res_scores) if all_res_scores else 0.0
    print(f"\nOverall mean RES ({len(all_res_scores)} cases): {overall_mean:.4f}")

    print("\n--- Error-analysis table ---")
    print(f"{'issue_kind':<20}{'count':>10}")
    for kind in ISSUE_KINDS:
        print(f"{kind:<20}{issue_counter.get(kind, 0):>10}")

    print()
    print_usage_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None,
        help="use only the first N rows of train.csv (for a quick, cheap smoke test)",
    )
    args = parser.parse_args()
    main(limit=args.limit)
