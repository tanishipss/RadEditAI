# RadEdit AI

A template-editing pipeline for the Kaggle Radiology Reporting Harness challenge.

This is **not** image-to-report generation. Every case already supplies a `template_content`
(the starting report) and a `dictation` (the radiologist's spoken findings). The task is to
produce a `FINDINGS:` / `IMPRESSION:` report that:

- routes each dictated finding to the correct template field,
- edits only the fields a dictated finding actually affects,
- preserves every untouched template field's wording exactly,
- never hallucinates a finding that wasn't dictated,
- keeps the template's field labels and order intact,
- and rewrites `IMPRESSION:` to summarize the dictated abnormalities.

Success is measured by the **Radiology Edit Score (RES)** — lower is better — reimplemented
locally in `src/res_scorer.py` so the pipeline can be evaluated before submission.

## Project structure

```
radiology_pipeline/
    data/            competition CSVs (train/test/sample_submission) — gitignored, see below
    src/             the pipeline itself, one module per pipeline stage
    tests/           pytest suite, one test file per src module
    configs/          reserved for model/threshold config as the pipeline grows
    experiments/      analysis scripts (currently: profile_dataset.py, Phase 1 dataset profiling)
    notebooks/        reserved for the Kaggle-notebook version of the pipeline
    outputs/          generated run artifacts (outputs/profiling/, etc.) — gitignored, see below
    scripts/          CLI entry points: cross-validation eval and test-set submission
    prompts/          reserved for prompt-library reference material
    pytest.ini        adds src/ to sys.path so tests/ can import pipeline modules by name
    requirements.txt
    .env.example
    .gitignore
```

## Module responsibilities (`src/`)

| Module | Responsibility |
|---|---|
| `schemas.py` | Pydantic models shared across the pipeline: `ExtractedFinding`, `ExtractionResult`, `FieldRouting`, `FieldChangeDecision`, `FieldEditResult`, `ValidationIssue`/`ValidationResult`. |
| `template_parser.py` | Splits `template_content` into an ordered list of `(field_label, field_text)` pairs (plus the template's own `IMPRESSION:` text), tolerant of mixed-case and comma-containing labels seen in the real templates. |
| `retrieval.py` | Builds a same-modality/body-part few-shot index over `train.csv` (three-tier fallback: exact template match -> same modality+body_part -> same modality) so every LLM call gets 1-2 relevant worked examples, with unchanged fields abbreviated to keep prompts small. |
| `extraction.py` | LLM call #1 — extracts every dictated finding out of `dictation` into structured `ExtractedFinding` objects (anatomy, laterality, severity, measurement, acuity, polarity), without deciding which field it belongs to. |
| `router.py` | Routes each extracted finding to one of the template's own field labels. Deterministic-first: mines a word -> field-label co-occurrence lexicon from `train.csv` and resolves each word at routing time, restricted to the current case's own field labels (falls back to an LLM call only when the lexicon can't resolve a finding). |
| `change_detector.py` | Pure logic, no LLM: a field counts as "changed" iff at least one routing points at it (excluding the reserved "unmentioned/normal" label). |
| `field_editor.py` | LLM call #2 — one batched call edits the text of every changed field for the case, given the original field text and the findings routed to it. |
| `assembler.py` | Purely deterministic reassembly: stitches changed + unchanged fields back into a `FINDINGS:` block in the template's original order/labels, then combines with the impression into the final report string. The LLM never controls structure — only field/impression text. |
| `impression_gen.py` | LLM call #3 — rewrites `IMPRESSION:` from the final `FINDINGS:` text, using the template's original impression as a fallback anchor for normal cases. |
| `validator.py` | Rule-based checks (negation, laterality, measurement fidelity — no LLM) plus one LLM call for the harder checks (unsupported content, wrong-field routing, unnecessary edits, impression mismatch). Produces a `ValidationResult`. |
| `repair.py` | LLM call #4 — only runs if validation fails; re-edits just the flagged field(s), one call per distinct flagged field, then re-validates once. Falls back to the best pre-repair report (logged to `data/repair_failures.log`) rather than ever returning nothing. |
| `pipeline.py` | Wires all of the above together in order and exposes `build_pipeline_context(train_df)` (build retrieval index + lexicon once per run) and `run_case(row, context) -> dict`. |
| `groq_client.py` | Shared Groq API wrapper: lazy client construction, JSON-schema-constrained structured output validated against a Pydantic model, retry-with-error-feedback on schema failures, and backoff/retry on rate-limit/transient errors. |
| `res_scorer.py` | Local reimplementation of the competition's Radiology Edit Score, used by `scripts/run_eval.py` for offline cross-validation. **Known issue**: its field-label regex is ALL-CAPS-only and disagrees with `template_parser.py`'s (deliberately widened) one — see Phase 1 profiling below. |
| `profiling.py` | Phase 1, read-only dataset profiling: dataset shape/dtype/missing-value summaries, categorical/template/text-length distributions, and — most importantly — a field-level diff of every train row's `template_content` against its reference `report` (changed/unchanged/added/removed/re-cased fields, plus a token-level added/removed/modified classification per changed field). No LLM calls. |

`scripts/run_eval.py` runs 5-fold cross-validation over `data/train.csv` and prints per-fold and
overall mean RES plus a validation-issue breakdown. `scripts/run_submission.py` runs the full
pipeline over `data/test.csv` and writes `data/submission.csv` with exactly `case_id,report`
columns.

## Phase 1: dataset profiling

```
python experiments/profile_dataset.py
```

Reads `data/train.csv` / `data/test.csv` / `data/sample_submission.csv` read-only and writes
every analysis artifact (JSON/CSV/JSONL plus a human-readable `SUMMARY.md`) to
`outputs/profiling/` — see `outputs/profiling/SUMMARY.md` for the current findings, template
diversity, most-frequently-changed fields, and recommendations for the next pipeline stage.
`outputs/` is gitignored, so re-run this to regenerate it locally.

## Setup

```
pip install -r requirements.txt
cp .env.example .env   # then fill in GROQ_API_KEY, or export it directly in your shell
```

## Running

```
pytest                                 # full test suite (no API key needed — LLM calls are
                                        # exercised through synthetic/deterministic paths only)
python scripts/run_eval.py --limit 20  # cheap smoke test against a slice of train.csv
python scripts/run_submission.py       # full test.csv -> data/submission.csv
```

## Notes

- Competition data files (`data/*.csv`, `data/*.log`) are gitignored — they are not
  redistributed through this repo.
- `configs/` and `notebooks/` are still placeholders (tracked via `.gitkeep`) — model/threshold
  configuration is still inline in `groq_client.py`, and the Kaggle-notebook version of the
  pipeline hasn't been assembled.
- `res_scorer.py`'s field parser fails to parse any field on templates with Title/mixed-case
  labels (falls back to one `__UNLABELLED__` blob), unlike `template_parser.py`'s widened regex.
  See `outputs/profiling/SUMMARY.md` §7/§8 (generated by `experiments/profile_dataset.py`) for
  the exact count of affected templates/rows before trusting any `scripts/run_eval.py` RES
  number.
