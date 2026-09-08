"""
Assembler — FR7 (TRD.md §4.8). Purely deterministic, NO LLM call anywhere in this file: the LLM
never controls final report structure (field order, field labels, headers) — only the text of
individual changed fields (field_editor.py) and the impression (impression_gen.py). This module
is the single source of truth for output formatting.
"""

from __future__ import annotations

from schemas import FieldEditResult


def assemble_findings(fields: dict[str, str], changes: dict[str, bool], edits: list[FieldEditResult]) -> str:
    """Reproduce the template's field order and labels exactly, substituting edited text for
    changed fields and template text (verbatim) for unchanged fields."""
    edits_by_label = {e.field_label: e.new_text for e in edits}

    lines = []
    for label, original_text in fields.items():
        text = edits_by_label.get(label, original_text) if changes.get(label) else original_text
        text = text.strip()
        if label == "__UNLABELLED__":
            lines.append(text)
            continue
        sep = "" if text == "" else " "
        lines.append(f"{label}:{sep}{text}")

    return "FINDINGS:\n" + "\n".join(lines)


def assemble_report(findings_text: str, impression: str) -> str:
    return findings_text.rstrip() + "\n\nIMPRESSION:\n" + impression.strip()
