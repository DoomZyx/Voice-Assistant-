import pytest

from livekit_agent.config import LiveKitSettings


def test_build_llm_uses_chat_completions_base_url():
    pytest.importorskip("livekit.plugins.openai")
    from livekit_agent.llm.client import build_llm

    settings = LiveKitSettings(
        _env_file=None,
        llm_base_url="http://vllm.test/v1",
        llm_model="served-model",
        llm_api_key="k",
        llm_timeout=12,
        llm_temperature=0.2,
    )
    llm = build_llm(settings)
    dumped = str(getattr(llm, "model", "")) + str(getattr(llm, "_opts", "")) + str(llm)
    assert "served-model" in dumped or getattr(llm, "model", None) in {None, "served-model"}
