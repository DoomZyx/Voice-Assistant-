from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class ResolvedLLM:
    base_url: str
    model: str
    api_key: str
    timeout: float


class LiveKitSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    livekit_url: str = ""
    livekit_api_key: str = ""
    livekit_api_secret: str = ""
    agent_name: str = "restaurant-voice"

    node_base_url: str = "http://localhost:8080"
    node_api_key: str = ""
    http_timeout: float = 15.0

    llm_base_url: str = ""
    llm_model: str = ""
    llm_api_key: str = ""
    llm_timeout: float = 60.0
    llm_temperature: float = 0.4
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = ""
    vllm_api_key: str = ""

    whisper_model_size: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_language: str = "fr"
    whisper_initial_prompt: str = (
        "Reservation restaurant francais: table, couverts, menu, tacos, "
        "Schmitt, Nguyen, vingt heures, vingt et une heures."
    )
    whisper_beam_size: int = 5

    kokoro_voice: str = "ff_siwis"
    kokoro_lang: str = "fr-fr"
    livekit_kokoro_speed: float = 1.2
    kokoro_model_path: str = ""
    kokoro_voices_path: str = ""
    tts_sample_rate: int = 24000

    restaurant_id: str = ""
    restaurant_name: str = ""
    sip_did_tenant_map: str = ""
    voice_tools_enabled: bool = False
    greeting_text: str = "Bonjour, je vous ecoute."
    cache_dir: str = ""

    def resolve_cache_dir(self) -> Path:
        if self.cache_dir:
            return Path(self.cache_dir).expanduser()
        return Path.home() / ".cache" / "mysmartfood-voice"


def resolve_llm(settings: LiveKitSettings) -> ResolvedLLM:
    base_url = (settings.llm_base_url or settings.vllm_base_url).rstrip("/")
    model = (settings.llm_model or settings.vllm_model).strip()
    api_key = (settings.llm_api_key or settings.vllm_api_key).strip()
    return ResolvedLLM(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=float(settings.llm_timeout),
    )


def get_settings() -> LiveKitSettings:
    return LiveKitSettings()
