"""Parse tags out of the structured caption line and filter by them.

Captions are one line of `field:[v1, v2] | field:[] | ...` (see caption.py). This module
turns that into discrete tags for the UI's tag picker and AND-filter.
"""

import re
from collections import Counter

# Each `field:[...]` occurrence; body may be empty. Robust to odd spacing between fields.
_FIELD_RE = re.compile(r"(\w+):\[([^\]]*)\]")

# Fields that make good clickable tags. `text:` and `search:` are OCR/free-text phrases — kept
# out of the tag cloud (they're for semantic search, not faceting).
TAG_FIELDS = frozenset(
    {
        "subject",
        "people",
        "emotion",
        "action",
        "environment",
        "location",
        "objects",
        "appearance",
        "colors",
        "topics",
    }
)


def parse_caption(caption: str) -> dict[str, list[str]]:
    """Parse a caption line into {field: [values]}. Empty fields and junk are dropped."""
    result: dict[str, list[str]] = {}
    for field, body in _FIELD_RE.findall(caption or ""):
        values = [v.strip() for v in body.split(",") if v.strip()]
        if values:
            result[field] = values
    return result


def caption_tags(caption: str) -> set[str]:
    """Lowercased tag values for one caption, drawn only from TAG_FIELDS."""
    parsed = parse_caption(caption)
    return {v.lower() for field in TAG_FIELDS for v in parsed.get(field, [])}


def build_tag_counts(captions: list[str]) -> Counter:
    """Tag frequency across all captions — used to rank the tag-picker options."""
    counts: Counter = Counter()
    for c in captions:
        counts.update(caption_tags(c))
    return counts


def matches_tags(caption: str, selected: list[str]) -> bool:
    """True if every selected tag is present in the caption (AND). Empty selection passes."""
    if not selected:
        return True
    return {s.lower() for s in selected} <= caption_tags(caption)
