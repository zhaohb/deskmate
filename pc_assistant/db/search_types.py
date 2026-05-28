"""Search content types for unified /search API."""

from __future__ import annotations

from enum import Enum


class ContentType(str, Enum):
    ALL = "all"
    OCR = "ocr"
    AUDIO = "audio"
    INPUT = "input"
    ACCESSIBILITY = "accessibility"
    MEMORY = "memory"


def normalize_content_type(raw: str) -> ContentType:
    value = (raw or "all").strip().lower()
    aliases = {
        "ui": ContentType.INPUT,
        "frames": ContentType.OCR,
        "frame": ContentType.OCR,
        "transcript": ContentType.AUDIO,
        "transcripts": ContentType.AUDIO,
    }
    if value in aliases:
        return aliases[value]
    try:
        return ContentType(value)
    except ValueError:
        return ContentType.ALL
