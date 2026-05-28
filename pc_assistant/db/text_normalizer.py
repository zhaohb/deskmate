"""FTS query normalization for safe MATCH clauses."""

from __future__ import annotations

import re

_CAMEL_CASE = re.compile(r"([a-z])([A-Z])")
_NUM_TO_LETTER = re.compile(r"([0-9])([a-zA-Z])")
_LETTER_TO_NUM = re.compile(r"([a-zA-Z])([0-9])")


def _split_compound(text: str) -> str:
    if not any(c.isupper() or c.isdigit() for c in text):
        return text
    result = _CAMEL_CASE.sub(r"\1 \2", text)
    result = _NUM_TO_LETTER.sub(r"\1 \2", result)
    result = _LETTER_TO_NUM.sub(r"\1 \2", result)
    return result


def sanitize_fts5_query(query: str) -> str:
    """Wrap each token in double quotes for safe FTS5 MATCH."""
    tokens: list[str] = []
    for token in query.split():
        cleaned = token.replace("\\", "").replace('"', "")
        if cleaned:
            tokens.append(f'"{cleaned}"')
    return " ".join(tokens)


def expand_search_query(query: str) -> str:
    """Prefix matching + compound split for grouped FTS queries."""
    query = query.strip()
    if not query:
        return ""

    expanded_terms: list[str] = []
    for word in query.split():
        cleaned = word.replace("\\", "").replace('"', "")
        split = _split_compound(cleaned)
        parts = split.split()
        if len(parts) > 1:
            terms = [f'"{cleaned}"*']
            for part in parts:
                if len(part) >= 2:
                    terms.append(f'"{part}"*')
            expanded_terms.append(f"({' OR '.join(terms)})")
        else:
            expanded_terms.append(f'"{cleaned}"*')

    if len(expanded_terms) == 1:
        return expanded_terms[0]
    return f"({' OR '.join(expanded_terms)})"


def value_to_fts5_column_query(column: str, value: str) -> str:
    return " ".join(f"{column}:{token}" for token in sanitize_fts5_query(value).split())
