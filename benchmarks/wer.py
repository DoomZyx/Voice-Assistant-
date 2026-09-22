from __future__ import annotations

import re
import unicodedata


_APOSTROPHES = str.maketrans({"’": "'", "´": "'", "`": "'"})


def normalize_transcript(text: str) -> str:
    value = unicodedata.normalize("NFKD", text or "")
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.translate(_APOSTROPHES).casefold()
    value = value.replace("'", "")
    value = re.sub(r"[^a-z0-9\s-]", " ", value)
    return " ".join(value.split())


def _levenshtein(left: list[str], right: list[str]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_word in enumerate(left, start=1):
        current = [i]
        for j, right_word in enumerate(right, start=1):
            cost = 0 if left_word == right_word else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def summarize_conditions(details: list[dict]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for row in details:
        groups.setdefault(str(row.get("condition") or "unknown"), []).append(row)
    summary: dict[str, dict] = {}
    for condition, rows in groups.items():
        wers = [float(row.get("wer") or 0.0) for row in rows]
        latencies = [float(row.get("latency_ms") or 0.0) for row in rows]
        ordered = sorted(latencies)
        mid = ordered[len(ordered) // 2] if ordered else 0.0
        if ordered and len(ordered) % 2 == 0:
            mid = (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2
        summary[condition] = {
            "samples": len(rows),
            "wer_mean": round(sum(wers) / len(wers), 4) if wers else None,
            "latency_p50_ms": round(mid, 1) if ordered else None,
            "empty_hypotheses": sum(1 for row in rows if not str(row.get("hypothesis") or "").strip()),
        }
    return summary


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = normalize_transcript(reference).split()
    hyp_words = normalize_transcript(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _levenshtein(ref_words, hyp_words) / len(ref_words)
