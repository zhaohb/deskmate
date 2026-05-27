"""Regex-based PII removal.

Optional ONNX detectors can run as a heavier second pass; this module provides
the fast regex baseline."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "phone_cn": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "phone_us": re.compile(r"(?<!\d)(?:\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}(?!\d)"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "secret_token": re.compile(r"(?i)(?:api[_-]?key|token|secret)[\"':=\s]+[A-Za-z0-9+/_\-]{12,}"),
}


@dataclass(frozen=True)
class PiiSpan:
    rule: str
    start: int
    end: int
    text: str


def find_pii_spans(text: str, rules: Iterable[str] | None = None) -> list[PiiSpan]:
    if not text:
        return []
    chosen = list(rules) if rules else list(_PATTERNS.keys())
    spans: list[PiiSpan] = []
    for rule in chosen:
        pat = _PATTERNS.get(rule)
        if pat is None:
            continue
        for match in pat.finditer(text):
            spans.append(PiiSpan(rule=rule, start=match.start(), end=match.end(), text=match.group(0)))
    spans.sort(key=lambda s: (s.start, s.end))
    return spans


def remove_pii(text: str, rules: Iterable[str] | None = None, replacement: str = "[REDACTED]") -> str:
    if not text:
        return text
    chosen = list(rules) if rules else list(_PATTERNS.keys())
    out = text
    for rule in chosen:
        pat = _PATTERNS.get(rule)
        if pat is None:
            continue
        out = pat.sub(replacement, out)
    return out
