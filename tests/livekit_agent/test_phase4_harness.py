from benchmarks.livekit.phase4_acceptance import CHECKLIST, main
from livekit_agent.config import LiveKitSettings


def test_phase4_checklist_covers_acceptance_items():
    text = " | ".join(CHECKLIST)
    assert len(CHECKLIST) == 10
    assert "DID de test" in text
    assert "barge-in" in text
    assert "call_id/session_id/restaurant_id" in text
    assert "vLLM" in text


def test_phase4_harness_exit_codes(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.livekit.phase4_acceptance.get_settings",
        lambda: LiveKitSettings(_env_file=None, livekit_url=""),
    )
    monkeypatch.setattr("sys.argv", ["phase4_acceptance"])
    assert main() == 2
    monkeypatch.setattr(
        "benchmarks.livekit.phase4_acceptance.get_settings",
        lambda: LiveKitSettings(
            _env_file=None,
            livekit_url="ws://127.0.0.1:7880",
            livekit_api_key="devkey",
            livekit_api_secret="secret",
            llm_base_url="http://127.0.0.1:8000/v1",
            llm_model="served",
            voice_tools_enabled=False,
        ),
    )
    assert main() == 0
