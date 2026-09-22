import pytest

from app.config import Settings
from app.providers import available_providers, build_engines
from tests.fakes import FakeLLM, FakeSTT, FakeTTS, FakeVadModel


def test_available_providers_lists_known_families():
    catalog = available_providers()
    assert "faster-whisper" in catalog["stt"]
    assert "whisper-large-v3-turbo" in catalog["stt"]
    assert "parakeet" in catalog["stt"]
    assert "kokoro" in catalog["tts"]
    assert "vllm" in catalog["llm"]
    assert "silero" in catalog["vad"]
    assert "heuristic" in catalog["turn"]
    assert "smart_turn" in catalog["turn"]
    assert "native" in catalog["orchestrator"]


def test_build_engines_keeps_injected_fakes():
    settings = Settings(voice_eager_load=False, stt_provider="faster-whisper")
    engines = build_engines(
        settings,
        stt=FakeSTT(),
        tts=FakeTTS(),
        llm=FakeLLM(),
        vad_model=FakeVadModel(),
    )
    assert engines.stt is not None
    assert engines.tts is not None
    assert engines.llm is not None
    assert engines.vad_model is not None
    assert engines.stt_provider == "faster-whisper"
    assert engines.tts_provider == "kokoro"


def test_unknown_stt_provider_fails_fast():
    settings = Settings(voice_eager_load=False, stt_provider="unknown-asr")
    with pytest.raises(ValueError, match="STT_PROVIDER"):
        build_engines(settings, eager=True)
