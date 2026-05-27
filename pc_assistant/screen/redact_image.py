"""Pixel-level OCR region redaction.

Regions are derived from OCR word boxes whose text overlaps regex PII spans.
Redaction uses solid black rectangles, following screenpipe's current image
redaction policy that avoids reversible blur.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from ..config import Config
from ..core import find_pii_spans


@dataclass(frozen=True)
class RedactionRegion:
    x: int
    y: int
    width: int
    height: int
    label: str


def regions_from_ocr(
    *,
    ocr_text: str,
    ocr_text_json: str | None,
    image_width: int,
    image_height: int,
    cfg: Config,
    padding_px: int = 2,
) -> list[RedactionRegion]:
    words = _load_words(ocr_text_json)
    if not words:
        return []

    full_text, spans = _word_spans(words)
    if not full_text and ocr_text:
        full_text = ocr_text
    pii_spans = find_pii_spans(full_text, cfg.redact.rules)
    if not pii_spans:
        return []

    regions: list[RedactionRegion] = []
    for pii in pii_spans:
        for word, start, end in spans:
            if start < pii.end and end > pii.start:
                region = _word_region(
                    word,
                    image_width=image_width,
                    image_height=image_height,
                    label=pii.rule,
                    padding_px=padding_px,
                )
                if region is not None:
                    regions.append(region)
    return _merge_regions(regions)


def redact_image_bytes(image_path: Path, regions: list[RedactionRegion], *, quality: int = 85) -> bytes:
    with Image.open(image_path) as img:
        redacted = img.convert("RGB")
    if regions:
        draw = ImageDraw.Draw(redacted)
        for region in regions:
            x1 = max(0, region.x)
            y1 = max(0, region.y)
            x2 = min(redacted.width, region.x + region.width)
            y2 = min(redacted.height, region.y + region.height)
            if x1 < x2 and y1 < y2:
                draw.rectangle((x1, y1, x2, y2), fill=(0, 0, 0))
    out = io.BytesIO()
    redacted.save(out, format="JPEG", quality=quality)
    return out.getvalue()


def _load_words(value: str | None) -> list[dict[str, Any]]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [w for w in parsed if isinstance(w, dict) and str(w.get("text") or "").strip()]


def _word_spans(words: list[dict[str, Any]]) -> tuple[str, list[tuple[dict[str, Any], int, int]]]:
    parts: list[str] = []
    spans: list[tuple[dict[str, Any], int, int]] = []
    offset = 0
    for word in words:
        if parts:
            parts.append(" ")
            offset += 1
        text = str(word.get("text") or "").strip()
        start = offset
        parts.append(text)
        offset += len(text)
        spans.append((word, start, offset))
    return "".join(parts), spans


def _word_region(
    word: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
    label: str,
    padding_px: int,
) -> RedactionRegion | None:
    try:
        left = float(word.get("left", 0))
        top = float(word.get("top", 0))
        width = float(word.get("width", 0))
        height = float(word.get("height", 0))
    except (TypeError, ValueError):
        return None

    # OCR output stores normalized coordinates, but accept pixel values too.
    if 0.0 <= left <= 1.0 and 0.0 <= width <= 1.0:
        x = int(left * image_width)
        w = int(width * image_width)
    else:
        x = int(left)
        w = int(width)
    if 0.0 <= top <= 1.0 and 0.0 <= height <= 1.0:
        y = int(top * image_height)
        h = int(height * image_height)
    else:
        y = int(top)
        h = int(height)

    x -= padding_px
    y -= padding_px
    w += padding_px * 2
    h += padding_px * 2
    if w <= 0 or h <= 0:
        return None
    return RedactionRegion(x=x, y=y, width=w, height=h, label=label)


def _merge_regions(regions: list[RedactionRegion]) -> list[RedactionRegion]:
    # Keep merging conservative: only coalesce exact/overlapping boxes from
    # adjacent PII words to reduce draw calls without covering whole lines.
    merged: list[RedactionRegion] = []
    for region in regions:
        current = region
        next_merged: list[RedactionRegion] = []
        for existing in merged:
            if _overlaps(existing, current):
                current = _union(existing, current)
            else:
                next_merged.append(existing)
        next_merged.append(current)
        merged = next_merged
    return merged


def _overlaps(a: RedactionRegion, b: RedactionRegion) -> bool:
    return (
        a.x <= b.x + b.width
        and b.x <= a.x + a.width
        and a.y <= b.y + b.height
        and b.y <= a.y + a.height
    )


def _union(a: RedactionRegion, b: RedactionRegion) -> RedactionRegion:
    x1 = min(a.x, b.x)
    y1 = min(a.y, b.y)
    x2 = max(a.x + a.width, b.x + b.width)
    y2 = max(a.y + a.height, b.y + b.height)
    return RedactionRegion(x=x1, y=y1, width=x2 - x1, height=y2 - y1, label=a.label)
