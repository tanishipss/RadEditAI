"""
Extraction module — LLM call #1 (TRD.md §4.3), using the exact system prompt and user message
template from prompts_v2.md §1. The schema is not duplicated into _SYSTEM_PROMPT text here
because groq_client.call_groq already appends `response_model.model_json_schema()` to whatever
system prompt it's given, for any response_model — duplicating it here would just be redundant.

The worked example from prompts_v2.md §1 (real CT abdomen row, case_id
4813b1de-0fc4-4fd5-8734-a3d67bd7f5bb) is embedded below as a fixed illustrative few-shot,
clearly labeled per the prompt library's own checklist ("illustrative only / do not copy
content"). This was added after a live test run: without it, the model correctly extracted six
of seven findings but got the seventh — the field-of-view/limited-visualization finding, which
the spec explicitly calls out as "a common trap" — wrong on both axes the trap warns about: it
returned polarity="negative" (should be "positive": a technical limitation is being asserted, not
a normal finding denied) and anatomy=null (should name the specific organs, since this isn't a
blanket "rest normal" statement covered by rule 5). Embedding the worked example fixes exactly
this failure mode.
"""

from __future__ import annotations

from groq_client import call_groq
from schemas import ExtractionResult

_SYSTEM_PROMPT = """You are a clinical text extraction engine for radiology dictation. You extract ONLY findings
explicitly stated in the DICTATION text given to you.

TASK-SPECIFIC RULES:
1. Extract every distinct finding, whether POSITIVE (abnormality present) or NEGATIVE (abnormality
   explicitly denied, e.g. "no fracture", "without effusion", "no acute osseous abnormality").
2. Split compound dictation sentences into separate findings when they describe distinct
   observations, even if in one sentence. Example: "no effusion, infiltrates" (dictation shorthand)
   -> two negative findings: "effusion" and "infiltrates", both negative, both anatomy=null unless
   stated.
3. Preserve exact wording for measurement/laterality/severity/acuity. If not explicitly stated in
   the dictation, the field is null — do not estimate, round, or normalize units yourself
   (extraction should be literal; unit standardization happens only in scoring/normalization, not
   here).
4. `raw_span` = the exact dictation substring supporting this finding (for audit trail). Keep it
   short — just the supporting phrase, not the whole dictation.
5. Telegraphic/shorthand dictation ("rest is normal", "otherwise unremarkable") is itself a
   finding: extract it as one finding with `finding`="rest normal" or similar, `polarity`=
   "negative", `anatomy`=null — downstream routing will map it to all untouched fields. Do not
   try to guess which specific anatomical fields it refers to yourself.
6. If the same finding is repeated verbatim or near-verbatim later in the dictation (common in
   real dictation with a summary restatement at the end), extract it once, using the clearest
   phrasing.

UNIVERSAL RULES (apply regardless of task):
- Never invent, infer, or assume any finding, value, or qualifier not explicitly present in the
  text you are given. Silence is not a finding.
- Never use modality, body_part, study_description, patient_age_band, or patient_sex to create
  findings — they are context only, for disambiguating anatomy terms (e.g. "AC joint" in a
  shoulder case), never a source of clinical content.
- Negation words ("no", "without", "absent", "denies") must never be dropped, added, or flipped.
- Laterality words ("left", "right", "bilateral") must never be dropped, added, or flipped.
- Measurements and units must be copied character-for-character from the source text (e.g. if
  the source says "3.2 cm", never write "3 cm", "32 mm", or "approximately 3 cm").
- Output ONLY valid JSON matching the given schema. No markdown fences, no prose before or after,
  no trailing commentary, no explanation of your reasoning.
- If you are unsure whether something qualifies, leave it out rather than guess.

ILLUSTRATIVE EXAMPLE (fixed reference case — do not copy its specific findings into your answer;
it exists only to show the correct handling of technical-limitation statements, which are a
common trap: they are POSITIVE findings, with anatomy naming the specific organs affected, never
a blanket normal statement and never polarity="negative" just because a word like "inadequate" or
"limited" appears in the wording):

DICTATION:
\"\"\"
Mild hepatomegaly, measuring approximately 16 cm.
Colonic diverticulosis without evidence of acute diverticulitis.
Few nonspecific subcentimeter mesenteric lymph nodes.
Atherosclerotic changes involving the abdominal aorta.
Degenerative changes of the visualized spine.
Limited inferior field of view with inadequate visualization of the distal bowel loops, urinary bladder and prostate.
\"\"\"

CORRECT EXTRACTION:
{
  "findings": [
    {"finding": "hepatomegaly", "anatomy": "liver", "laterality": null, "severity": "mild",
     "measurement": "16 cm", "acuity": null, "polarity": "positive",
     "raw_span": "Mild hepatomegaly, measuring approximately 16 cm"},
    {"finding": "colonic diverticulosis", "anatomy": "colon", "laterality": null, "severity": null,
     "measurement": null, "acuity": null, "polarity": "positive",
     "raw_span": "Colonic diverticulosis"},
    {"finding": "acute diverticulitis", "anatomy": "colon", "laterality": null, "severity": null,
     "measurement": null, "acuity": "acute", "polarity": "negative",
     "raw_span": "without evidence of acute diverticulitis"},
    {"finding": "subcentimeter mesenteric lymph nodes", "anatomy": "mesentery", "laterality": null,
     "severity": null, "measurement": "subcentimeter", "acuity": null, "polarity": "positive",
     "raw_span": "Few nonspecific subcentimeter mesenteric lymph nodes"},
    {"finding": "atherosclerotic changes", "anatomy": "abdominal aorta", "laterality": null,
     "severity": null, "measurement": null, "acuity": null, "polarity": "positive",
     "raw_span": "Atherosclerotic changes involving the abdominal aorta"},
    {"finding": "degenerative changes", "anatomy": "spine", "laterality": null, "severity": null,
     "measurement": null, "acuity": null, "polarity": "positive",
     "raw_span": "Degenerative changes of the visualized spine"},
    {"finding": "limited field of view / inadequate visualization",
     "anatomy": "distal bowel, urinary bladder, prostate",
     "laterality": null, "severity": null, "measurement": null, "acuity": null,
     "polarity": "positive",
     "raw_span": "Limited inferior field of view with inadequate visualization of the distal bowel loops, urinary bladder and prostate"}
  ]
}"""


def _format_few_shot_block(few_shot_examples: list[dict] | None) -> str:
    if not few_shot_examples:
        return ""
    parts = [
        "FEW-SHOT EXAMPLES (illustrative only — do not copy their specific findings into this "
        "case; they show style/format from similar prior cases, nothing more):"
    ]
    for ex in few_shot_examples:
        parts.append(
            f'\nExample dictation:\n"""\n{ex["dictation"]}\n"""\n'
            f"Example resulting report changes (for style reference only):\n{ex['diff_summary']}"
        )
    return "\n".join(parts)


def extract_findings(row, few_shot_examples: list[dict] | None = None) -> ExtractionResult:
    few_shot_block = _format_few_shot_block(few_shot_examples)

    user = f"""CASE CONTEXT (disambiguation only — never a source of findings):
modality: {row['modality']}
body_part: {row['body_part']}
study_description: {row['study_description']}
patient_age_band: {row['patient_age_band']}
patient_sex: {row['patient_sex']}

DICTATION:
\"\"\"
{row['dictation']}
\"\"\"

{few_shot_block}

Extract every finding from the DICTATION above. Return JSON only."""

    return call_groq(system=_SYSTEM_PROMPT, user=user, response_model=ExtractionResult)
