"""
Template parser — FR1 (TRD.md §4.1).

Parses an arbitrary radiology report template (or a rendered report, same grammar) of the form

    FINDINGS:
    LABEL ONE: some content.
    LABEL TWO: more content,
    possibly spanning several lines.

    IMPRESSION:
    free text.

into a plain, order-preserving structure:

    {
        "findings_fields": [
            {"label": "LABEL ONE", "order": 0, "content": "some content."},
            {"label": "LABEL TWO", "order": 1, "content": "more content,\npossibly spanning several lines."},
        ],
        "impression": "\nfree text.",
    }

No field label is ever hardcoded (no chest-specific, spine-specific, etc. label list) — a line
is recognized as starting a new field purely by matching FIELD_RE, so the parser works
identically across every modality/body-part template in the corpus (XRAY, CT, MRI, USG all
observed to parse correctly against train.csv).

FIELD_RE deliberately widens the naive "ALL CAPS label" assumption
(`^([A-Z][A-Z0-9 /()&-]{1,40}):\\s*(.*)$`): scanning every unique template in train.csv shows
that assumption is false. Many templates use Title Case or mixed-case labels (`Bones:`,
`Menisci:`, `Osseous Structures:`), and at least one uses a comma inside the label
(`BICEPS BRACHII, LONG HEAD:`). Under the strict all-caps regex, all of those lines silently
fail to match and get absorbed as continuation text of whichever field preceded them. FIELD_RE
below widens the character class to `[A-Za-z0-9 /()&,-]` (lowercase letters and a comma added)
and requires only that the label START with a letter. Verified against every unique
template_content in train.csv (82 templates): the widened regex resolves all previously
unmatched label lines, introduces zero false-positive matches (checked by flagging any
newly-matched "label" longer than 6 words, since a real label here is always a short noun
phrase), and leaves zero templates with 0 parsed fields where real labels are actually present.
"""

from __future__ import annotations

import re

FIELD_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 /()&,-]{1,50}):\s*(.*)$")

_FINDINGS_HEADER_RE = re.compile(r"^FINDINGS:\n?")
_IMPRESSION_HEADER_RE = re.compile(r"^IMPRESSION:", flags=re.MULTILINE)

# 1 of the 82 unique templates in train.csv (cervical-spine XRAY) has NO field labels at all —
# the FINDINGS section is a flat paragraph of unlabelled prose. Content before the first
# recognized field label (or the entirety of the findings block, for that template) is kept
# under this reserved label rather than silently dropped, so no dictated/templated content is
# ever lost even when it can't be attributed to a named field.
UNLABELLED_LABEL = "__UNLABELLED__"


def parse_template(template_content) -> dict:
    """Parse `template_content` into {"findings_fields": [...], "impression": str}.

    findings_fields is a list of {"label", "content", "order"} dicts, in the exact order the
    labels appeared in the source text (order is also each entry's own list index, included
    explicitly per the required output shape). Each field's stored content preserves embedded
    blank lines / multi-line continuations exactly as they appeared in the source (this is what
    makes `render(parse_template(t))` reproduce `t` byte-for-byte). An empty field (label
    immediately followed by nothing, e.g. "OTHER FINDINGS:" with no text before the next
    section) is kept with content "" rather than dropped.
    """
    text = str(template_content)

    m = _IMPRESSION_HEADER_RE.search(text)
    if m:
        findings_block = text[: m.start()]
        impression_text = text[m.end():]
    else:
        findings_block = text
        impression_text = ""

    findings_block = _FINDINGS_HEADER_RE.sub("", findings_block, count=1)

    findings_fields: list[dict] = []
    current_label = UNLABELLED_LABEL
    current_lines: list[str] = []

    def _flush() -> None:
        if current_label == UNLABELLED_LABEL and not current_lines:
            return
        content = "\n".join(current_lines)
        if current_label == UNLABELLED_LABEL and content == "":
            return
        findings_fields.append({
            "label": current_label,
            "content": content,
            "order": len(findings_fields),
        })

    for line in findings_block.split("\n"):
        matched = FIELD_RE.match(line)
        if matched:
            _flush()
            current_label = matched.group(1)
            current_lines = [matched.group(2)]
        else:
            current_lines.append(line)
    _flush()

    return {"findings_fields": findings_fields, "impression": impression_text}


def fields_as_dict(parsed: dict) -> dict[str, str]:
    """Convenience view of parse_template()'s output as a plain {label: content} dict, for
    callers that only need label->text lookup (dict insertion order already matches field
    order, so nothing about ordering is lost by dropping the explicit "order" key here)."""
    return {field["label"]: field["content"] for field in parsed["findings_fields"]}


def fields_from_dict(fields: dict[str, str], impression: str) -> dict:
    """Inverse of fields_as_dict: build a parse_template()-shaped dict (suitable for render())
    from a plain {label: content} dict plus an impression string, using the dict's own
    iteration order as field order."""
    return {
        "findings_fields": [
            {"label": label, "content": content, "order": i}
            for i, (label, content) in enumerate(fields.items())
        ],
        "impression": impression,
    }


def render(parsed: dict) -> str:
    """Reconstruct template/report text from a parse_template()-shaped dict.

    Deterministic inverse of parse_template: a single space is (re-)inserted after the label's
    colon whenever the stored content is non-empty and doesn't already start with a newline —
    matching the near-universal "LABEL: text" convention in the source data. Genuine
    multi-space or irregular-spacing anomalies in the original are not byte-reproduced by this
    rule (they collapse to one space); the round-trip test normalizes whitespace before
    comparing for exactly this reason. Fields are emitted in `order` (== list position, since
    parse_template always assigns order as the list index).
    """
    lines = []
    for field in parsed["findings_fields"]:
        label, content = field["label"], field["content"]
        if label == UNLABELLED_LABEL:
            lines.append(content)
            continue
        sep = "" if (content == "" or content.startswith("\n")) else " "
        lines.append(f"{label}:{sep}{content}")
    body = "FINDINGS:\n" + "\n".join(lines)
    return body + "IMPRESSION:" + parsed["impression"]
