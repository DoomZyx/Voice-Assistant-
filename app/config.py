from pathlib import Path
import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    voice_host: str = "0.0.0.0"
    voice_port: int = 8090
    voice_eager_load: bool = True

    node_base_url: str = "http://localhost:8080"
    node_api_key: str = ""
    twilio_auth_token: str = ""

    llm_provider: str = "vllm"
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "llama-3.1-8b-instruct"
    vllm_api_key: str = ""
    hf_token: str = ""
    llm_fallback_gpt: bool = True
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.4
    llm_max_tokens: int = 400
    llm_max_tool_rounds: int = 4

    whisper_model_size: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_language: str = "fr"
    whisper_initial_prompt: str = (
        "Reservation restaurant francais: table, couverts, menu, tacos, "
        "Schmitt, Nguyen, vingt heures, vingt et une heures."
    )

    stt_provider: str = "faster-whisper"
    tts_provider: str = "kokoro"
    vad_provider: str = "silero"
    turn_detector: str = "heuristic"
    voice_orchestrator: str = "native"
    audio_preprocess: str = "agc"
    whisper_beam_size: int = 5
    parakeet_model: str = "nvidia/parakeet-tdt-0.6b-v3"

    llm_stream: bool = True
    tts_stream: bool = True
    tts_chunking: str = "sentence"
    turn_incomplete_wait_ms: int = 800
    barge_in_enabled: bool = True
    # Candidat 160 ms (bande 120-200). Pas une constante optimale : voir benchmark barge-in.
    barge_in_confirm_ms: int = 160

    kokoro_voice: str = "ff_siwis"
    kokoro_lang: str = "fr-fr"
    kokoro_speed: float = 1.0
    kokoro_model_path: str = ""
    kokoro_voices_path: str = ""

    vad_threshold: float = 0.35
    vad_silence_ms: int = 550
    vad_min_speech_ms: int = 180
    vad_model_path: str = ""

    greeting_text: str = "Bonjour, je vous ecoute."
    cache_dir: str = ""

    twilio_sample_rate: int = 8000
    stt_sample_rate: int = 16000
    tts_native_rate: int = 24000

    def node_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.node_api_key:
            headers["x-api-key"] = self.node_api_key
        return headers

    def resolve_cache_dir(self) -> Path:
        if self.cache_dir:
            return Path(self.cache_dir).expanduser()
        return Path.home() / ".cache" / "mysmartfood-voice"


def apply_huggingface_token(settings: Settings) -> None:
    """HF_TOKEN sert au telechargement gated (Llama, etc.), pas a vLLM en local."""
    token = (settings.hf_token or "").strip()
    if not token:
        return
    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)


def get_settings() -> Settings:
    settings = Settings()
    apply_huggingface_token(settings)
    return settings
