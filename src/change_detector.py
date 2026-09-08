"""
Change detector — FR2 (TRD.md §4.5), field-change detection (prompts_v2.md §3). Pure Python,
NO LLM call: a field is changed iff at least one routing (other than the reserved
ALL_REMAINING_UNMENTIONED label) points at it.

This is exactly the `detect_changes` function from prompts_v2.md §3, adapted only to accept the
real FieldRouting pydantic objects route_findings() actually returns (`r.field_label`) rather
than the toy `dict` shown there (`r["field_label"]`) — same logic, same semantics.
"""

from __future__ import annotations

from router import ALL_REMAINING_UNMENTIONED
from schemas import FieldRouting


def detect_changes(fields: dict[str, str], routings: list[FieldRouting]) -> dict[str, bool]:
    touched = {r.field_label for r in routings if r.field_label != ALL_REMAINING_UNMENTIONED}
    return {label: (label in touched) for label in fields}
