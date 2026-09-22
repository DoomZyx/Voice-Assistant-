from __future__ import annotations

from dataclasses import dataclass, field

from livekit_agent.config import LiveKitSettings, resolve_llm


@dataclass(frozen=True)
class Phase4Report:
    ready: bool
    missing: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)


def missing_runtime_settings(settings: LiveKitSettings) -> list[str]:
    missing: list[str] = []
    if not settings.livekit_url.strip():
        missing.append("LIVEKIT_URL")
    if not settings.livekit_api_key.strip():
        missing.append("LIVEKIT_API_KEY")
    if not settings.livekit_api_secret.strip():
        missing.append("LIVEKIT_API_SECRET")
    resolved = resolve_llm(settings)
    if not resolved.base_url:
        missing.append("LLM_BASE_URL")
    if not resolved.model:
        missing.append("LLM_MODEL")
    return missing


def worker_exit_code(settings: LiveKitSettings) -> int:
    """Refuse le boot si l'infra LiveKit/LLM manque. Les tools P5+ ne dispensent pas ce check."""
    return 1 if missing_runtime_settings(settings) else 0


def phase4_readiness(settings: LiveKitSettings) -> Phase4Report:
    missing = missing_runtime_settings(settings)
    blockers: list[str] = []
    profile = (settings.voice_tools_profile or "").strip().lower()
    if settings.voice_tools_enabled and profile != "reservations":
        blockers.append(
            "VOICE_TOOLS_ENABLED=true : tools autorises seulement avec VOICE_TOOLS_PROFILE=reservations"
        )
    return Phase4Report(ready=not missing and not blockers, missing=missing, blockers=blockers)
