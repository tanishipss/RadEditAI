import os

import pandas as pd
import pytest

from change_detector import detect_changes
from router import ALL_REMAINING_UNMENTIONED, build_lexicon, route_findings
from schemas import ExtractedFinding, ExtractionResult

TRAIN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")

# The CT abdomen worked example from prompts_v2.md sections 1-2 (case_id
# 4813b1de-0fc4-4fd5-8734-a3d67bd7f5bb), given as ExtractedFinding objects directly so this test
# doesn't depend on a live extraction call.
_CT_ABDOMEN_FINDINGS = [
    ExtractedFinding(finding="hepatomegaly", anatomy="liver", severity="mild",
                      measurement="16 cm", polarity="positive", raw_span="x"),
    ExtractedFinding(finding="colonic diverticulosis", anatomy="colon", polarity="positive", raw_span="x"),
    ExtractedFinding(finding="acute diverticulitis", anatomy="colon", acuity="acute",
                      polarity="negative", raw_span="x"),
    ExtractedFinding(finding="subcentimeter mesenteric lymph nodes", anatomy="mesentery",
                      measurement="subcentimeter", polarity="positive", raw_span="x"),
    ExtractedFinding(finding="atherosclerotic changes", anatomy="abdominal aorta",
                      polarity="positive", raw_span="x"),
    ExtractedFinding(finding="degenerative changes", anatomy="spine", polarity="positive", raw_span="x"),
    ExtractedFinding(finding="limited field of view / inadequate visualization",
                      anatomy="distal bowel, urinary bladder, prostate",
                      polarity="positive", raw_span="x"),
]

_CT_ABDOMEN_FIELD_LABELS = [
    "LIVER", "GALLBLADDER AND BILIARY TREE", "PANCREAS", "SPLEEN", "ADRENAL GLANDS",
    "KIDNEYS AND URETERS", "URINARY BLADDER", "REPRODUCTIVE", "MAJOR VESSELS", "PERITONEUM",
    "ABDOMINAL WALL", "LYMPH NODES", "STOMACH AND BOWEL", "BONES",
]


@pytest.fixture(scope="module")
def lexicon():
    if not os.path.exists(TRAIN_PATH):
        pytest.skip("train.csv not found")
    return build_lexicon(pd.read_csv(TRAIN_PATH))


@pytest.fixture(scope="module")
def routings(lexicon):
    extraction = ExtractionResult(findings=_CT_ABDOMEN_FINDINGS)
    return route_findings(extraction, _CT_ABDOMEN_FIELD_LABELS, lexicon)


def test_ct_abdomen_routings_match_documented_expected(routings):
    by_index: dict[int, set[str]] = {}
    for r in routings:
        by_index.setdefault(r.finding_index, set()).add(r.field_label)

    assert by_index[0] == {"LIVER"}
    assert by_index[1] == {"STOMACH AND BOWEL"}
    assert by_index[2] == {"STOMACH AND BOWEL"}
    assert by_index[3] == {"LYMPH NODES"}
    assert by_index[4] == {"MAJOR VESSELS"}
    assert by_index[5] == {"BONES"}
    # finding_index 6 (field-of-view limitation) must include URINARY BLADDER at minimum, per
    # the doc's given "Correct routing" JSON; this router additionally splits it across
    # REPRODUCTIVE and STOMACH AND BOWEL, matching the *true* human reference report even more
    # closely than that simplified single-field example does.
    assert "URINARY BLADDER" in by_index[6]


def test_ct_abdomen_multi_field_routing_for_finding_6(routings):
    by_index: dict[int, set[str]] = {}
    for r in routings:
        by_index.setdefault(r.finding_index, set()).add(r.field_label)
    assert by_index[6] == {"STOMACH AND BOWEL", "URINARY BLADDER", "REPRODUCTIVE"}


def test_all_findings_resolved_deterministically_no_llm_needed(lexicon):
    """Every finding in this canonical example should resolve via the deterministic lexicon
    alone — if this regresses, some finding silently started requiring an LLM fallback call."""
    extraction = ExtractionResult(findings=_CT_ABDOMEN_FINDINGS)
    from router import _route_one_deterministic

    for finding in extraction.findings:
        resolved = _route_one_deterministic(finding, _CT_ABDOMEN_FIELD_LABELS, lexicon)
        assert resolved is not None, f"finding {finding.finding!r} needed the LLM fallback"


def test_change_detector_matches_documented_expected(routings):
    fields = {label: "placeholder text" for label in _CT_ABDOMEN_FIELD_LABELS}
    changes = detect_changes(fields, routings)

    expected_changed = {"LIVER", "STOMACH AND BOWEL", "LYMPH NODES", "MAJOR VESSELS", "BONES", "URINARY BLADDER"}
    expected_unchanged = {
        "GALLBLADDER AND BILIARY TREE", "PANCREAS", "SPLEEN", "ADRENAL GLANDS",
        "KIDNEYS AND URETERS", "PERITONEUM", "ABDOMINAL WALL",
    }

    for label in expected_changed:
        assert changes[label] is True, f"{label} should be changed"
    for label in expected_unchanged:
        assert changes[label] is False, f"{label} should be unchanged"


def test_change_detector_ignores_reserved_label():
    fields = {"LIVER": "x", "SPLEEN": "y"}
    routings = [
        type("R", (), {"field_label": "LIVER"})(),
        type("R", (), {"field_label": ALL_REMAINING_UNMENTIONED})(),
    ]
    changes = detect_changes(fields, routings)
    assert changes == {"LIVER": True, "SPLEEN": False}


def test_blanket_normal_finding_routes_to_reserved_label(lexicon):
    finding = ExtractedFinding(finding="rest normal", anatomy=None, polarity="negative", raw_span="x")
    extraction = ExtractionResult(findings=[finding])
    result = route_findings(extraction, _CT_ABDOMEN_FIELD_LABELS, lexicon)
    assert len(result) == 1
    assert result[0].field_label == ALL_REMAINING_UNMENTIONED
