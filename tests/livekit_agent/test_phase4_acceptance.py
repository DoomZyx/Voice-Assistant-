from livekit_agent.config import LiveKitSettings
from livekit_agent.phase4 import phase4_readiness, worker_exit_code


def test_phase4_readiness_reports_missing_env():
    settings = LiveKitSettings(
        _env_file=None,
        livekit_url="",
        llm_base_url="",
        vllm_base_url="",
        llm_model="",
        vllm_model="",
        voice_tools_enabled=True,
    )
    report = phase4_readiness(settings)
    assert report.ready is False
    assert "LIVEKIT_URL" in report.missing
    assert "LIVEKIT_API_KEY" in report.missing
    assert "LIVEKIT_API_SECRET" in report.missing
    assert "LLM_BASE_URL" in report.missing
    assert "LLM_MODEL" in report.missing
    assert any("VOICE_TOOLS_PROFILE" in item for item in report.blockers)


def test_phase4_readiness_passes_when_config_complete():
    settings = LiveKitSettings(
        _env_file=None,
        livekit_url="ws://127.0.0.1:7880",
        llm_base_url="http://127.0.0.1:8000/v1",
        llm_model="mistral-7b",
        livekit_api_key="devkey",
        livekit_api_secret="secret",
        voice_tools_enabled=False,
    )
    report = phase4_readiness(settings)
    assert report.ready is True
    assert report.missing == []
    assert report.blockers == []
    assert worker_exit_code(settings) == 0


def test_worker_exits_when_runtime_env_missing_even_if_tools_on():
    settings = LiveKitSettings(
        _env_file=None,
        livekit_url="",
        llm_base_url="http://127.0.0.1:8000/v1",
        llm_model="mistral-7b",
        livekit_api_key="devkey",
        livekit_api_secret="secret",
        voice_tools_enabled=True,
    )
    assert worker_exit_code(settings) == 1


def test_worker_allows_tools_after_phase4_when_runtime_env_is_complete():
    settings = LiveKitSettings(
        _env_file=None,
        livekit_url="ws://127.0.0.1:7880",
        llm_base_url="http://127.0.0.1:8000/v1",
        llm_model="mistral-7b",
        livekit_api_key="devkey",
        livekit_api_secret="secret",
        voice_tools_enabled=True,
    )
    assert worker_exit_code(settings) == 0
    assert phase4_readiness(settings).ready is False


def test_phase4_allows_reservation_profile_in_test():
    settings = LiveKitSettings(
        _env_file=None,
        livekit_url="ws://127.0.0.1:7880",
        llm_base_url="http://127.0.0.1:8000/v1",
        llm_model="mistral-7b",
        livekit_api_key="devkey",
        livekit_api_secret="secret",
        voice_tools_enabled=True,
        voice_tools_profile="reservations",
    )
    report = phase4_readiness(settings)
    assert report.ready is True
    assert report.blockers == []


def test_phase4_allows_full_profile():
    settings = LiveKitSettings(
        _env_file=None,
        livekit_url="ws://127.0.0.1:7880",
        llm_base_url="http://127.0.0.1:8000/v1",
        llm_model="mistral-7b",
        livekit_api_key="devkey",
        livekit_api_secret="secret",
        voice_tools_enabled=True,
        voice_tools_profile="full",
    )
    assert phase4_readiness(settings).ready is True
