from __future__ import annotations

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:])\s+|(?<=[.!?])\s+")
_TERMINAL = re.compile(r"[.!?…]$")


def split_speech_chunks(text: str, strategy: str = "sentence") -> list[str]:
    cleaned = " ".join((text or "").split()).strip()
    if not cleaned:
        return []
    mode = (strategy or "sentence").strip().lower()
    if mode == "word":
        return [cleaned]
    pattern = _CLAUSE_SPLIT if mode == "clause" else _SENTENCE_SPLIT
    chunks = [part.strip() for part in pattern.split(cleaned) if part.strip()]
    return chunks or [cleaned]


def flush_speech_chunks(buffer: str, strategy: str = "sentence") -> tuple[list[str], str]:
    cleaned = " ".join((buffer or "").split()).strip()
    if not cleaned:
        return [], ""
    chunks = split_speech_chunks(cleaned, strategy=strategy)
    if not chunks:
        return [], ""
    if _TERMINAL.search(chunks[-1]):
        return chunks, ""
    return chunks[:-1], chunks[-1]
