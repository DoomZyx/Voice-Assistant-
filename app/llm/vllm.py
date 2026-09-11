"""Backend OpenAI-compatible : vLLM en V1, interchangeable ensuite."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import httpx

from app.config import Settings
from app.llm import LLMResult, ToolCall

logger = logging.getLogger("mysmartfood-voice.llm")


class VLLMProvider:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._owns = client is None
        headers = {"Content-Type": "application/json"}
        if settings.vllm_api_key:
            headers["Authorization"] = f"Bearer {settings.vllm_api_key}"
        self._client = client or httpx.AsyncClient(
            base_url=settings.vllm_base_url.rstrip("/"),
            headers=headers,
            timeout=60.0,
        )

    async def aclose(self) -> None:
        if self._owns:
            await self._client.aclose()

    async def list_models(self) -> list[str]:
        response = await self._client.get("/models")
        if response.status_code >= 400:
            logger.error("step=llm.models.error provider=vllm status=%s", response.status_code)
            response.raise_for_status()
        data = response.json()
        ids = []
        for item in data.get("data") or []:
            if isinstance(item, dict) and item.get("id"):
                ids.append(str(item["id"]))
        return ids

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResult:
        payload: dict[str, Any] = {
            "model": self._settings.vllm_model,
            "messages": messages,
            "temperature": self._settings.llm_temperature,
            "max_tokens": self._settings.llm_max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        response = await self._client.post("/chat/completions", json=payload)
        if response.status_code >= 400:
            logger.error("step=llm.completion.error provider=vllm status=%s", response.status_code)
            response.raise_for_status()
        data = response.json()
        message = (data.get("choices") or [{}])[0].get("message") or {}
        return LLMResult(
            content=(message.get("content") or "").strip(),
            tool_calls=_parse_tool_calls(message.get("tool_calls") or []),
        )


def model_id_error(configured: str, served: list[str]) -> str | None:
    """None si VLLM_MODEL est exactement un id servi par GET /v1/models."""
    requested = (configured or "").strip()
    if not requested:
        return (
            "VLLM_MODEL est vide. Copier l'id exact renvoye par GET /v1/models "
            "(ex. meta-llama/Llama-3.1-8B-Instruct), pas un alias local."
        )
    if not served:
        return "vLLM n'expose aucun modele (GET /v1/models vide). Le serveur n'a peut-etre pas fini de charger."
    if requested in served:
        return None
    return (
        f"VLLM_MODEL={requested!r} ne correspond a aucun modele charge. "
        f"Modeles servis: {served}. "
        "L'id doit matcher exactement (souvent l'identifiant Hugging Face, "
        "ex. meta-llama/Llama-3.1-8B-Instruct)."
    )


def build_llm(settings: Settings) -> VLLMProvider:
    provider = (settings.llm_provider or "vllm").strip().lower()
    if provider in {"vllm", "openai", "openai-compat"}:
        return VLLMProvider(settings)
    raise ValueError(f"LLM_PROVIDER inconnu: {settings.llm_provider}")


def _parse_tool_calls(raw: list[Any]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") or {}
        arguments = fn.get("arguments") or "{}"
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                parsed = {}
        elif isinstance(arguments, dict):
            parsed = arguments
        else:
            parsed = {}
        calls.append(
            ToolCall(
                id=str(item.get("id") or uuid.uuid4()),
                name=str(fn.get("name") or ""),
                arguments=parsed,
            )
        )
    return calls
