from __future__ import annotations

import re
import unicodedata


_NUMBER_WORDS = {
    "un": 1,
    "une": 1,
    "deux": 2,
    "trois": 3,
    "quatre": 4,
    "cinq": 5,
    "six": 6,
    "sept": 7,
    "huit": 8,
    "neuf": 9,
    "dix": 10,
    "onze": 11,
    "douze": 12,
}


def _fold(text: str) -> str:
    value = unicodedata.normalize("NFKD", text or "")
    value = "".join(char for char in value if not unicodedata.combining(char))
    return value.casefold()


def latest_party_size(utterances: list[str]) -> int | None:
    found: int | None = None
    pattern = re.compile(r"\b(\d+|un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze)\b")
    for text in utterances:
        matches = pattern.findall(_fold(text))
        if not matches:
            continue
        raw = matches[-1]
        found = int(raw) if raw.isdigit() else _NUMBER_WORDS.get(raw)
    return found
