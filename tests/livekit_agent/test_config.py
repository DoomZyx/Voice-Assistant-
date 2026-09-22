from livekit_agent.config import LiveKitSettings, resolve_llm


def test_llm_base_url_prefers_explicit_value():
    settings = LiveKitSettings(
        _env_file=None,
        llm_base_url="http://vllm:8000/v1",
        vllm_base_url="http://localhost:8000/v1",
        llm_model="mistral-7b",
        vllm_model="legacy-model",
        llm_api_key="new-key",
        vllm_api_key="old-key",
    )
    resolved = resolve_llm(settings)
    assert resolved.base_url == "http://vllm:8000/v1"
    assert resolved.model == "mistral-7b"
    assert resolved.api_key == "new-key"


def test_llm_falls_back_to_vllm_env():
    settings = LiveKitSettings(
        _env_file=None,
        llm_base_url="",
        vllm_base_url="http://localhost:8000/v1/",
        llm_model="",
        vllm_model="restaurant-assistant",
        llm_api_key="",
        vllm_api_key="shared",
    )
    resolved = resolve_llm(settings)
    assert resolved.base_url == "http://localhost:8000/v1"
    assert resolved.model == "restaurant-assistant"
    assert resolved.api_key == "shared"


def test_phase4_defaults_disable_tools_and_set_kokoro_speed():
    settings = LiveKitSettings(_env_file=None)
    assert settings.voice_tools_enabled is False
    assert settings.voice_tools_profile == ""
    assert settings.livekit_kokoro_speed == 1.2
    assert settings.kokoro_lang == "fr-fr"
    assert settings.kokoro_voice == "ff_siwis"
    assert settings.whisper_language == "fr"
    assert settings.agent_name == "restaurant-voice"


def test_kokoro_speed_ignores_legacy_env(monkeypatch):
    monkeypatch.setenv("KOKORO_SPEED", "1.0")
    monkeypatch.delenv("LIVEKIT_KOKORO_SPEED", raising=False)
    settings = LiveKitSettings(_env_file=None)
    assert settings.livekit_kokoro_speed == 1.2


def test_legacy_kokoro_speed_ignores_livekit_env(monkeypatch):
    monkeypatch.setenv("LIVEKIT_KOKORO_SPEED", "1.2")
    monkeypatch.delenv("KOKORO_SPEED", raising=False)
    from app.config import Settings

    legacy = Settings(_env_file=None)
    assert legacy.kokoro_speed == 1.0
