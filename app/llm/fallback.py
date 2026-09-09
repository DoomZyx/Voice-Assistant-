"""vLLM en priorite, OpenAI GPT en fallback. Les logs prefixent FALLBACK GPT."""

from __future__ import annotations

import logging
from typing import Any

from app.config import Settings
from app.llm import LLMResult
from app.llm.vllm import VLLMProvider, build_llm as build_vllm_client

logger = logging.getLogger("mysmartfood-voice.llm")


class FallbackLLM:
    def __init__(self, primary: VLLMProvider, fallback: VLLMProvider | None) -> None:
        self._primary = primary
        self._fallback = fallback
        self.active_provider = "vllm"

    async def aclose(self) -> None:
        await self._primary.aclose()
        if self._fallback is not None:
            await self._fallback.aclose()

    async def list_models(self) -> list[str]:
        try:
            return await self._primary.list_models()
        except Exception as exc:
            logger.warning("FALLBACK GPT: GET /models vLLM impossible reason=%s", exc)
            raise

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResult:
        try:
            result = await self._primary.complete(messages, tools)
            self.active_provider = "vllm"
            logger.info("step=llm.backend provider=vllm")
            return result
        except Exception as exc:
            if self._fallback is None:
                logger.error("FALLBACK GPT: impossible (OPENAI_API_KEY absent ou desactive) reason=%s", exc)
                raise
            logger.warning(
                "FALLBACK GPT: vLLM indisponible, bascule OpenAI Chat Completions. reason=%s",
                exc,
            )
            result = await self._fallback.complete(messages, tools)
            self.active_provider = "openai"
            logger.warning("FALLBACK GPT: reponse OpenAI recue provider=openai")
            return result


def build_llm(settings: Settings) -> VLLMProvider | FallbackLLM:
    primary = build_vllm_client(settings)
    if not settings.llm_fallback_gpt:
        logger.info("step=llm.fallback_gpt enabled=false")
        return primary
    key = (settings.openai_api_key or "").strip()
    if not key:
        logger.warning(
            "FALLBACK GPT: desactive cote LLM (OPENAI_API_KEY vide). "
            "Si vLLM est down, Fastify pourra encore basculer l appel vers OpenAI Realtime."
        )
        return primary
    fallback_settings = settings.model_copy(
        update={
            "vllm_base_url": settings.openai_base_url.rstrip("/"),
            "vllm_model": settings.openai_model,
            "vllm_api_key": key,
        }
    )
    fallback = VLLMProvider(fallback_settings)
    logger.info(
        "step=llm.fallback_gpt enabled=true openai_model=%s",
        settings.openai_model,
    )
    return FallbackLLM(primary, fallback)
