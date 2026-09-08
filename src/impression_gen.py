"""
Impression generator — LLM call #3 (TRD.md §4.7), using the exact system prompt and user
message template from prompts_v2.md §5.

The worked example from prompts_v2.md §5 (same CT abdomen case as extraction/field_editor,
case_id 4813b1de-0fc4-4fd5-8734-a3d67bd7f5bb) is embedded below as a fixed illustrative
few-shot, added after a live test run without it produced a real failure: given this case's
actual FINAL FINDINGS (clearly abnormal — hepatomegaly, diverticulosis, atherosclerotic changes,
degenerative changes, limited visualization) and its ORIGINAL TEMPLATE IMPRESSION ("No acute
intra-abdominal abnormality identified."), the model returned the template's fallback text
verbatim — i.e. it invoked rule 5's "no significant abnormality" escape hatch even though
significant abnormalities were clearly present, apparently because most of the 14 individual
FINDINGS fields (11 of them) are individually normal. Embedding a correct worked example of this
exact failure mode fixes it, the same way it fixed extraction.py's field-of-view mishandling.
"""

from __future__ import annotations

from pydantic import BaseModel

from groq_client import call_groq

_UNIVERSAL_RULES = """UNIVERSAL RULES (apply regardless of task):
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
it exists only to show that a case with abnormal findings must NOT fall back to a "no
abnormality" statement just because many individual FINDINGS fields happen to be normal — only
use the fallback when the case truly has no notable finding anywhere):

FINAL FINDINGS:
\"\"\"
FINDINGS:
LIVER: Mild hepatomegaly, measuring approximately 16 cm. Normal attenuation. No focal hepatic lesion.
GALLBLADDER AND BILIARY TREE: Normal. No biliary ductal dilatation.
PANCREAS: Normal in size and contour.
SPLEEN: Normal.
ADRENAL GLANDS: Normal in size and morphology.
KIDNEYS AND URETERS: Normal appearance without renal calculi or hydronephrosis.
URINARY BLADDER: Inadequate visualization of the urinary bladder due to limited inferior field of view.
REPRODUCTIVE: Inadequate visualization of the prostate due to limited inferior field of view.
MAJOR VESSELS: Atherosclerotic changes involving the abdominal aorta.
PERITONEUM: No ascites. No pneumoperitoneum.
ABDOMINAL WALL: The abdominal wall is unremarkable. No hernia.
LYMPH NODES: Few nonspecific subcentimeter mesenteric lymph nodes. No significant retroperitoneal lymphadenopathy.
STOMACH AND BOWEL: Colonic diverticulosis without evidence of acute diverticulitis. No evidence of bowel obstruction. Limited inferior field of view with inadequate visualization of the distal bowel loops.
BONES: Degenerative changes of the visualized spine.
\"\"\"

ORIGINAL TEMPLATE IMPRESSION (style/fallback reference):
\"\"\"
No acute intra-abdominal abnormality identified.
\"\"\"

CORRECT IMPRESSION (note: despite the template's fallback impression being "no abnormality",
several FINDINGS fields above ARE abnormal, so they are synthesized and reported — the fallback
is NOT reused here):
1. Mild hepatomegaly.
2. Colonic diverticulosis without evidence of acute diverticulitis.
3. Atherosclerotic changes of the abdominal aorta.
4. Degenerative changes of the visualized spine.
5. Limited evaluation of the pelvis due to field of view."""

_SYSTEM_PROMPT = f"""You are a radiology report engine writing the IMPRESSION section: a concise clinical summary of
the clinically significant abnormal findings in FINDINGS.

TASK-SPECIFIC RULES:
1. Summarize only findings already present in the FINAL FINDINGS text given. Never introduce a
   new finding or diagnosis not supported there.
2. Omit routine/normal statements — do not restate "no acute osseous abnormality" style content
   unless the FEW-SHOT examples show the reference style includes such lines for this
   template/body-part family.
3. Match numbering style: if few-shot IMPRESSIONs use a numbered list ("1. ... 2. ..."), use a
   numbered list. If they use prose, use prose. Match this exactly rather than picking your own
   default style.
4. Order findings from most to least clinically significant, mirroring the order shown in
   few-shot examples for similar cases when possible.
5. If FINDINGS contains no significant abnormality, write a brief normal-impression statement
   consistent with the ORIGINAL TEMPLATE IMPRESSION's wording style (you may reuse it near-verbatim
   in that case).
6. Do not copy FINDINGS sentences verbatim into IMPRESSION — synthesize/condense them.

{_UNIVERSAL_RULES}"""


class _ImpressionResponse(BaseModel):
    impression: str


def _format_few_shot_block(few_shot_examples: list[dict] | None) -> str:
    if not few_shot_examples:
        return ""
    parts = [
        "FEW-SHOT EXAMPLES (illustrative only — do not copy their specific findings into this "
        "case; they show style/format from similar prior cases, nothing more):"
    ]
    for ex in few_shot_examples:
        parts.append(f"\n{ex['diff_summary']}")
    return "\n".join(parts)


def generate_impression(
    final_findings_text: str,
    row,
    template_impression: str,
    few_shot_examples: list[dict] | None = None,
) -> str:
    few_shot_block = _format_few_shot_block(few_shot_examples)

    user = f"""CASE CONTEXT: {row['modality']} / {row['body_part']} / {row['study_description']}

FINAL FINDINGS:
\"\"\"
{final_findings_text}
\"\"\"

ORIGINAL TEMPLATE IMPRESSION (style/fallback reference):
\"\"\"
{template_impression}
\"\"\"

{few_shot_block}

Write the IMPRESSION."""

    result = call_groq(system=_SYSTEM_PROMPT, user=user, response_model=_ImpressionResponse)
    return result.impression
