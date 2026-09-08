import os
import re

import pandas as pd
import pytest

from template_parser import (
    UNLABELLED_LABEL,
    fields_as_dict,
    fields_from_dict,
    parse_template,
    render,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")


def _normalize_whitespace(text: str) -> str:
    """Collapse whitespace-only differences for round-trip comparison purposes only
    (never used inside the parser itself)."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +\n", "\n", text)
    return text.strip()


@pytest.fixture(scope="module")
def train_df():
    if not os.path.exists(DATA_PATH):
        pytest.skip("train.csv not found")
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def one_row_per_modality(train_df):
    """One representative real template_content per modality, taken verbatim from train.csv
    (first occurrence, no fabricated/edited text) — used to prove the parser is not
    chest/XRAY-specific."""
    return {
        modality: group.iloc[0]
        for modality, group in train_df.groupby("modality")
    }


# A real CT Chest template from train.csv (case_id 83b6c57f-a5b9-41e4-9a27-47459944a63d):
# multi-word slash-containing labels, an embedded (unusual, verbatim) "consolidation,cavitation"
# typo, an explicitly empty OTHER FINDINGS field, and a SUPPORT DEVICES field.
_CT_CHEST_TEMPLATE = (
    "FINDINGS:\n"
    "LUNGS/AIRWAYS: The lungs are clear. No focal consolidation,cavitation, suspicious pulmonary "
    "nodule or other focal lesion is identified.\n"
    "PLEURA: No pleural effusion or pneumothorax.\n"
    "CARDIOVASCULAR: The heart and great vessels are normal in size and configuration. No thoracic "
    "aortic aneurysm. No pericardial effusion.\n"
    "MEDIASTINAL/HILAR LYMPH NODES: No lymphadenopathy.\n"
    "CHEST WALL/MUSCULOSKELETAL: No acute osseous abnormality. No aggressive osseous lesion.\n"
    "LINES/TUBES/SUPPORT DEVICES: None.\n"
    "OTHER FINDINGS:\n"
    "\n"
    "\n"
    "IMPRESSION:\n"
    "No acute intrathoracic abnormality."
)


# ---------------------------------------------------------------------------
# Basic shape / section detection
# ---------------------------------------------------------------------------

def test_parse_template_returns_required_shape():
    # BONES is followed by a second field (not the last field before IMPRESSION) so this
    # assertion isn't entangled with the blank-line-ownership behavior covered separately by
    # test_last_field_owns_trailing_blank_line_before_impression below.
    template = (
        "FINDINGS:\n"
        "BONES: No acute fracture.\n"
        "JOINTS: Normal.\n"
        "\n"
        "IMPRESSION:\n"
        "Normal."
    )
    parsed = parse_template(template)
    assert set(parsed.keys()) == {"findings_fields", "impression"}
    assert isinstance(parsed["findings_fields"], list)
    field = parsed["findings_fields"][0]
    assert set(field.keys()) == {"label", "content", "order"}
    assert field["label"] == "BONES"
    assert field["content"] == "No acute fracture."
    assert field["order"] == 0


def test_detects_findings_and_impression_sections():
    template = (
        "FINDINGS:\n"
        "BONES: Normal.\n"
        "JOINTS: Normal.\n"
        "\n"
        "IMPRESSION:\n"
        "No acute abnormality."
    )
    parsed = parse_template(template)
    assert parsed["findings_fields"][0]["content"] == "Normal."
    assert "No acute abnormality." in parsed["impression"]


def test_last_field_owns_trailing_blank_line_before_impression():
    """The blank line separating the FINDINGS section from IMPRESSION: is stored as part of the
    LAST field's own content (not stripped) — this is what lets render() reproduce that blank
    line exactly without render() needing any separator logic of its own. Round-trip fidelity
    depends on this; callers that want a "clean" field value should use .strip()."""
    template = "FINDINGS:\nBONES: No acute fracture.\n\nIMPRESSION:\nNormal."
    parsed = parse_template(template)
    assert parsed["findings_fields"][0]["content"] == "No acute fracture.\n\n"
    assert render(parsed) == template


def test_missing_impression_section_yields_empty_impression():
    template = "FINDINGS:\nBONES: Normal."
    parsed = parse_template(template)
    assert parsed["impression"] == ""
    assert parsed["findings_fields"][0]["content"] == "Normal."


# ---------------------------------------------------------------------------
# Arbitrary labels across modalities — NOT hardcoded to chest fields
# ---------------------------------------------------------------------------

def test_arbitrary_labels_not_hardcoded_chest_only():
    """A single label set (chest) must not be assumed anywhere in the parser: labels as varied
    as DEEP VEINS (vascular ultrasound) and TERES MINOR (shoulder MRI tendon) must parse
    identically to BONES/JOINTS."""
    template = (
        "FINDINGS:\n"
        "DEEP VEINS: Patent and compressible.\n"
        "TERES MINOR: The tendon is intact.\n"
        "SINUS TARSI: Unremarkable.\n"
        "\n"
        "IMPRESSION:\n"
        "Normal."
    )
    parsed = parse_template(template)
    labels = [f["label"] for f in parsed["findings_fields"]]
    assert labels == ["DEEP VEINS", "TERES MINOR", "SINUS TARSI"]


@pytest.mark.parametrize("modality", ["XRAY", "CT", "MRI", "USG"])
def test_parses_real_template_for_every_modality(one_row_per_modality, modality):
    row = one_row_per_modality[modality]
    parsed = parse_template(row["template_content"])
    assert len(parsed["findings_fields"]) >= 1
    # every field label must actually appear, verbatim, in the source template
    for field in parsed["findings_fields"]:
        if field["label"] != UNLABELLED_LABEL:
            assert field["label"] + ":" in row["template_content"]


def test_support_devices_field_recognized():
    parsed = parse_template(_CT_CHEST_TEMPLATE)
    labels = [f["label"] for f in parsed["findings_fields"]]
    assert "LINES/TUBES/SUPPORT DEVICES" in labels


def test_other_findings_field_recognized_and_may_be_empty():
    parsed = parse_template(_CT_CHEST_TEMPLATE)
    other = next(f for f in parsed["findings_fields"] if f["label"] == "OTHER FINDINGS")
    # OTHER FINDINGS is the last field before IMPRESSION here, so its raw content also carries
    # the trailing blank-line separator (see test_last_field_owns_trailing_blank_line_...);
    # .strip() is the right check for "this field has no real content".
    assert other["content"].strip() == ""


# ---------------------------------------------------------------------------
# Multi-line content, empty fields, varying field counts
# ---------------------------------------------------------------------------

def test_multiline_field_content_preserved():
    template = (
        "FINDINGS:\n"
        "BONES: No acute fracture.\n"
        "Mild degenerative changes are noted.\n"
        "JOINTS: Normal alignment.\n"
        "\n"
        "IMPRESSION:\n"
        "Normal."
    )
    parsed = parse_template(template)
    bones = next(f for f in parsed["findings_fields"] if f["label"] == "BONES")
    assert bones["content"] == "No acute fracture.\nMild degenerative changes are noted."


def test_empty_field_content_is_preserved_not_dropped():
    template = (
        "FINDINGS:\n"
        "BONES: Normal.\n"
        "OTHER FINDINGS:\n"
        "\n"
        "IMPRESSION:\n"
        "Normal."
    )
    parsed = parse_template(template)
    labels = [f["label"] for f in parsed["findings_fields"]]
    assert "OTHER FINDINGS" in labels
    other = next(f for f in parsed["findings_fields"] if f["label"] == "OTHER FINDINGS")
    assert other["content"].strip() == ""


@pytest.mark.parametrize("n_expected_fields,template", [
    (3, "FINDINGS:\nBONES: Normal.\nJOINTS: Normal.\nSOFT TISSUES: Normal.\n\nIMPRESSION:\nNormal."),
    (1, "FINDINGS:\nBONES: Normal.\n\nIMPRESSION:\nNormal."),
])
def test_varying_number_of_fields(n_expected_fields, template):
    parsed = parse_template(template)
    assert len(parsed["findings_fields"]) == n_expected_fields


def test_unlabelled_flat_prose_template_is_preserved_under_one_field():
    template = (
        "FINDINGS:\n"
        "The spinal alignment is maintained without evidence of spondylolisthesis.\n"
        "No acute fracture is identified.\n"
        "\n"
        "IMPRESSION:\n"
        "Unremarkable X-ray of the cervical spine."
    )
    parsed = parse_template(template)
    assert len(parsed["findings_fields"]) == 1
    assert parsed["findings_fields"][0]["label"] == UNLABELLED_LABEL
    assert "spondylolisthesis" in parsed["findings_fields"][0]["content"]


def test_mixed_case_label_parses_as_its_own_field():
    template = (
        "FINDINGS:\n"
        "Bones: No acute fracture.\n"
        "Menisci:\n"
        "Medial meniscus: Intact.\n"
        "\n"
        "IMPRESSION:\n"
        "Normal."
    )
    parsed = parse_template(template)
    labels = [f["label"] for f in parsed["findings_fields"]]
    assert labels == ["Bones", "Menisci", "Medial meniscus"]
    menisci = next(f for f in parsed["findings_fields"] if f["label"] == "Menisci")
    assert menisci["content"] == ""


def test_comma_in_label_parses_as_its_own_field():
    template = (
        "FINDINGS:\n"
        "TERES MINOR: The tendon is intact.\n"
        "BICEPS BRACHII, LONG HEAD: The tendon is intact.\n"
        "\n"
        "IMPRESSION:\n"
        "Normal."
    )
    parsed = parse_template(template)
    labels = [f["label"] for f in parsed["findings_fields"]]
    assert "BICEPS BRACHII, LONG HEAD" in labels


# ---------------------------------------------------------------------------
# Field order
# ---------------------------------------------------------------------------

def test_field_order_matches_source_position():
    template = (
        "FINDINGS:\n"
        "BONES: No acute fracture or focal osseous lesion.\n"
        "JOINTS: No dislocation. The joint spaces are normal.\n"
        "SOFT TISSUES: The soft tissues are unremarkable.\n"
        "\n"
        "IMPRESSION:\n"
        "No acute osseous abnormality."
    )
    parsed = parse_template(template)
    labels_in_order = [f["label"] for f in parsed["findings_fields"]]
    orders = [f["order"] for f in parsed["findings_fields"]]
    assert labels_in_order == ["BONES", "JOINTS", "SOFT TISSUES"]
    assert orders == [0, 1, 2]


def test_punctuation_capitalization_and_typo_preserved_verbatim():
    """No autocorrection anywhere: an unusual verbatim typo/spacing artifact in the source
    ("consolidation,cavitation", no space after the comma) must survive parsing unchanged."""
    parsed = parse_template(_CT_CHEST_TEMPLATE)
    lungs = next(f for f in parsed["findings_fields"] if f["label"] == "LUNGS/AIRWAYS")
    assert "consolidation,cavitation" in lungs["content"]


# ---------------------------------------------------------------------------
# render(): deterministic inverse of parse_template()
# ---------------------------------------------------------------------------

def test_render_round_trip_hand_crafted_template():
    template = (
        "FINDINGS:\n"
        "BONES: No acute fracture or focal osseous lesion.\n"
        "JOINTS: No dislocation. The joint spaces are normal.\n"
        "SOFT TISSUES: The soft tissues are unremarkable.\n"
        "\n"
        "IMPRESSION:\n"
        "No acute osseous abnormality."
    )
    parsed = parse_template(template)
    assert render(parsed) == template


def test_render_round_trip_ct_chest_template_with_empty_field():
    parsed = parse_template(_CT_CHEST_TEMPLATE)
    assert render(parsed) == _CT_CHEST_TEMPLATE


def test_render_round_trip_unlabelled_template():
    template = (
        "FINDINGS:\n"
        "The spinal alignment is maintained without evidence of spondylolisthesis.\n"
        "No acute fracture is identified.\n"
        "\n"
        "IMPRESSION:\n"
        "Unremarkable X-ray of the cervical spine."
    )
    parsed = parse_template(template)
    assert render(parsed) == template


@pytest.mark.parametrize("modality", ["XRAY", "CT", "MRI", "USG"])
def test_render_round_trip_real_template_per_modality(one_row_per_modality, modality):
    original = str(one_row_per_modality[modality]["template_content"])
    rendered = render(parse_template(original))
    assert _normalize_whitespace(rendered) == _normalize_whitespace(original)


def test_render_round_trip_across_every_unique_train_template(train_df):
    """Comprehensive sweep: parse+render every unique template_content in train.csv and
    confirm it reproduces the original modulo pure whitespace, and that no template silently
    produces zero fields."""
    unique_templates = train_df["template_content"].drop_duplicates().tolist()

    round_trip_failures = []
    zero_field_templates = []
    for original in unique_templates:
        original = str(original)
        parsed = parse_template(original)
        if len(parsed["findings_fields"]) == 0:
            zero_field_templates.append(original)
        rendered = render(parsed)
        if _normalize_whitespace(rendered) != _normalize_whitespace(original):
            round_trip_failures.append(original)

    assert not zero_field_templates, f"{len(zero_field_templates)} templates produced zero fields"
    assert not round_trip_failures, f"{len(round_trip_failures)} templates failed to round-trip"


def test_no_exceptions_on_any_row(train_df):
    for _, row in train_df.iterrows():
        parse_template(row["template_content"])


# ---------------------------------------------------------------------------
# fields_as_dict / fields_from_dict convenience helpers
# ---------------------------------------------------------------------------

def test_fields_as_dict_preserves_order_and_content():
    template = "FINDINGS:\nBONES: Normal.\nJOINTS: Normal.\n\nIMPRESSION:\nNormal."
    parsed = parse_template(template)
    as_dict = fields_as_dict(parsed)
    assert list(as_dict.keys()) == ["BONES", "JOINTS"]
    assert as_dict["BONES"] == "Normal."


def test_fields_from_dict_round_trips_through_render():
    template = "FINDINGS:\nBONES: Normal.\nJOINTS: Normal.\n\nIMPRESSION:\nAll normal."
    parsed = parse_template(template)
    as_dict = fields_as_dict(parsed)
    rebuilt = fields_from_dict(as_dict, parsed["impression"])
    assert render(rebuilt) == template
