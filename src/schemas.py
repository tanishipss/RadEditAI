from pydantic import BaseModel, Field
from typing import Optional, Literal


class ExtractedFinding(BaseModel):
    finding: str
    anatomy: Optional[str] = None
    laterality: Optional[Literal["left", "right", "bilateral"]] = None
    severity: Optional[Literal["mild", "moderate", "severe"]] = None
    measurement: Optional[str] = None     # e.g. "3.2 cm" — verbatim, not re-derived
    acuity: Optional[Literal["acute", "chronic", "subacute"]] = None
    polarity: Literal["positive", "negative"]   # "no fracture" -> negative
    raw_span: str                                # exact dictation substring, for traceability


class ExtractionResult(BaseModel):
    findings: list[ExtractedFinding]


class FieldRouting(BaseModel):
    finding_index: int          # index into ExtractionResult.findings
    field_label: str            # must be one of the template's own field labels, or "OTHER FINDINGS"
    confidence: Literal["high", "low"]


class FieldChangeDecision(BaseModel):
    field_label: str
    changed: bool
    supporting_finding_indices: list[int]


class FieldEditResult(BaseModel):
    field_label: str
    new_text: str


class ValidationIssue(BaseModel):
    kind: Literal["unsupported", "missing", "wrong_field", "negation",
                  "laterality", "measurement", "unnecessary_edit", "impression_error"]
    field_label: Optional[str] = None
    detail: str


class ValidationResult(BaseModel):
    passed: bool
    issues: list[ValidationIssue] = []
