"""Helpers messages LLM / hangup. Importer: app.session.call."""

from __future__ import annotations

import json
from typing import Any

from app.llm import LLMResult
from app.session.bootstrap import apply_forced_caller_number

HANGUP_MAX_TURNS = 40
HANGUP_MAX_TURN_CHARS = 2000


def stt_turns(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for message in history:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        if message.get("tool_calls"):
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        turns.append(
            {
                "role": role,
                "content": content.strip()[:HANGUP_MAX_TURN_CHARS],
            }
        )
        if len(turns) >= HANGUP_MAX_TURNS:
            break
    return turns


def apply_forced_caller_to_result(result: LLMResult, caller_number: str | None) -> None:
    for call in result.tool_calls:
        if call.name == "create_appointment":
            call.arguments = apply_forced_caller_number(call.arguments, caller_number)


def assistant_tool_message(result: LLMResult) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": result.content or None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in result.tool_calls
        ],
    }
