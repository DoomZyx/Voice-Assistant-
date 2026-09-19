from __future__ import annotations

import httpx

from livekit_agent.config import LiveKitSettings, resolve_llm


def build_llm(settings: LiveKitSettings):
    """OpenAI Chat Completions client pointed at self-hosted vLLM."""
    from livekit.plugins import openai

    resolved = resolve_llm(settings)
    timeout = httpx.Timeout(resolved.timeout)
    return openai.LLM(
        model=resolved.model,
        base_url=resolved.base_url,
        api_key=resolved.api_key or "not-needed",
        temperature=settings.llm_temperature,
        timeout=timeout,
    )
