from pixelrag_visual.tags import (
    parse_caption,
    caption_tags,
    build_tag_counts,
    matches_tags,
)


CAP_A = (
    "text:[providus ai, seed round] | subject:[startup pitch deck] | people:[male founder] | "
    "emotion:[confident] | action:[presenting] | environment:[conference stage] | "
    "location:[dubai] | objects:[projector screen] | appearance:[black suit] | "
    "colors:[blue, white] | topics:[artificial intelligence, fundraising] | "
    "search:[startup funding, investor presentation]"
)
CAP_B = (
    "text:[nike] | subject:[running shoe] | people:[] | emotion:[] | action:[product photography] | "
    "environment:[studio] | location:[] | objects:[athletic shoe] | appearance:[mesh upper] | "
    "colors:[black, red] | topics:[sportswear, footwear] | search:[nike running shoes]"
)


def test_parse_caption_fields_and_values():
    parsed = parse_caption(CAP_A)
    assert parsed["subject"] == ["startup pitch deck"]
    assert parsed["colors"] == ["blue", "white"]
    assert parsed["topics"] == ["artificial intelligence", "fundraising"]


def test_parse_caption_skips_empty_fields():
    parsed = parse_caption(CAP_B)
    assert "people" not in parsed  # people:[] dropped
    assert "location" not in parsed


def test_parse_caption_handles_garbage():
    assert parse_caption("") == {}
    assert parse_caption("no brackets here") == {}


def test_caption_tags_excludes_text_and_search_fields():
    tags = caption_tags(CAP_A)
    assert "startup pitch deck" in tags
    assert "dubai" in tags
    # `text:` / `search:` free-text fields are not faceted
    assert "providus ai" not in tags
    assert "investor presentation" not in tags


def test_build_tag_counts_aggregates():
    counts = build_tag_counts([CAP_A, CAP_B])
    assert counts["blue"] == 1
    assert counts["black"] == 1  # only CAP_B has black
    # both decks/shoes are distinct subjects -> each counted once
    assert counts["running shoe"] == 1


def test_matches_tags_empty_selection_passes():
    assert matches_tags(CAP_A, []) is True


def test_matches_tags_single():
    assert matches_tags(CAP_A, ["dubai"]) is True
    assert matches_tags(CAP_B, ["dubai"]) is False


def test_matches_tags_and_semantics():
    assert matches_tags(CAP_A, ["dubai", "confident"]) is True       # both present
    assert matches_tags(CAP_A, ["dubai", "studio"]) is False         # one missing -> fail


def test_matches_tags_case_insensitive():
    assert matches_tags(CAP_A, ["DUBAI"]) is True
