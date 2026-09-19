from __future__ import annotations

from typing import Any

import httpx

from livekit_agent.config import LiveKitSettings, resolve_llm
from livekit_agent.llm.config import model_id_error
from livekit_agent.observability.logging import emit_event


class LLMModelMismatch(RuntimeError):
    pass


def _parse_model_ids(payload: Any) -> list[str]:
    ids: list[str] = []
    if not isinstance(payload, dict):
        return ids
    for item in payload.get("data") or []:
        if isinstance(item, dict) and item.get("id"):
            ids.append(str(item["id"]))
    return ids


async def list_llm_models(settings: LiveKitSettings, client: httpx.AsyncClient | None = None) -> list[str]:
    resolved = resolve_llm(settings)
    owns = client is None
    headers = {"Content-Type": "application/json"}
    if resolved.api_key:
        headers["Authorization"] = f"Bearer {resolved.api_key}"
    http = client or httpx.AsyncClient(
        base_url=resolved.base_url,
        headers=headers,
        timeout=resolved.timeout,
    )
    try:
        response = await http.get("/models")
        response.raise_for_status()
        return _parse_model_ids(response.json())
    finally:
        if owns:
            await http.aclose()


async def assert_llm_model(
    settings: LiveKitSettings,
    client: httpx.AsyncClient | None = None,
) -> list[str]:
    resolved = resolve_llm(settings)
    served = await list_llm_models(settings, client=client)
    error = model_id_error(resolved.model, served)
    if error:
        emit_event(
            "vllm.model_mismatch",
            model=resolved.model,
            served=",".join(served) or "-",
        )
        raise LLMModelMismatch(error)
    emit_event("vllm.model_ok", model=resolved.model, served=",".join(served))
    return served
