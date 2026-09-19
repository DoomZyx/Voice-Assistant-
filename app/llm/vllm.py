"""Backend OpenAI-compatible : vLLM en V1, interchangeable ensuite."""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

import httpx

from app.config import Settings
from app.llm import LLMResult, ToolCall

logger = logging.getLogger("mysmartfood-voice.llm")

ALLOWED_TOOL_NAMES = {"create_appointment", "check_availability"}
_HERMES_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


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
        usage = data.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or 0) or input_tokens + output_tokens
        content = (message.get("content") or "").strip()
        tool_calls = _parse_tool_calls(message.get("tool_calls") or [])
        if not tool_calls:
            hermes_calls, remainder = extract_hermes_tool_calls(content)
            if hermes_calls:
                tool_calls = hermes_calls
                content = remainder
                logger.info("step=llm.tool_calls source=hermes count=%s", len(tool_calls))
        return LLMResult(
            content=content,
            tool_calls=tool_calls,
            input_tokens=max(0, input_tokens),
            output_tokens=max(0, output_tokens),
            total_tokens=max(0, total_tokens),
            model=str(data.get("model") or self._settings.vllm_model or ""),
        )

    async def complete_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ):
        payload: dict[str, Any] = {
            "model": self._settings.vllm_model,
            "messages": messages,
            "temperature": self._settings.llm_temperature,
            "max_tokens": self._settings.llm_max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        async with self._client.stream("POST", "/chat/completions", json=payload) as response:
            if response.status_code >= 400:
                logger.error("step=llm.stream.error provider=vllm status=%s", response.status_code)
                response.raise_for_status()
            async for line in response.aiter_lines():
                for chunk in iter_chat_completion_chunks([line]):
                    yield chunk


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


def extract_hermes_tool_calls(content: str) -> tuple[list[ToolCall], str]:
    """Recupere les tool_calls Hermes si vLLM les a laisses dans le texte."""
    if not content:
        return [], content
    calls: list[ToolCall] = []
    for match in _HERMES_TOOL_CALL.finditer(content):
        parsed = _parse_json_object(match.group(1))
        if not parsed:
            continue
        name = str(parsed.get("name") or "")
        if name not in ALLOWED_TOOL_NAMES:
            continue
        arguments = parsed.get("arguments") or {}
        if isinstance(arguments, str):
            arguments = _parse_json_object(arguments) or {}
        if not isinstance(arguments, dict):
            arguments = {}
        calls.append(ToolCall(id=str(uuid.uuid4()), name=name, arguments=arguments))
    remainder = _HERMES_TOOL_CALL.sub("", content).strip() if calls else content
    return calls, remainder


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def iter_chat_completion_chunks(lines: list[str]) -> list[LLMResult]:
    chunks: list[LLMResult] = []
    for raw in lines:
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        delta = ((data.get("choices") or [{}])[0].get("delta") or {})
        usage = data.get("usage") or {}
        chunks.append(
            LLMResult(
                content=str(delta.get("content") or ""),
                tool_calls=_parse_tool_calls(delta.get("tool_calls") or []),
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
                total_tokens=int(usage.get("total_tokens") or 0),
                model=str(data.get("model") or ""),
            )
        )
    return chunks


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
        name = str(fn.get("name") or "")
        if name not in ALLOWED_TOOL_NAMES:
            continue
        calls.append(
            ToolCall(
                id=str(item.get("id") or uuid.uuid4()),
                name=name,
                arguments=parsed if isinstance(parsed, dict) else {},
            )
        )
    return calls
