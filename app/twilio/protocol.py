from __future__ import annotations

import json
from typing import Any


def media_out(stream_sid: str, mulaw_b64: str) -> str:
    return json.dumps(
        {
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": mulaw_b64},
        }
    )


def clear_stream(stream_sid: str) -> str:
    return json.dumps({"event": "clear", "streamSid": stream_sid})


def parse_event(raw: str | bytes) -> dict[str, Any]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Message Twilio invalide")
    return data
