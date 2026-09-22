import sys
import types

import pytest

from livekit_agent.config import LiveKitSettings
from livekit_agent.preflight import LLMModelMismatch
from livekit_agent.worker import main


def test_main_exits_when_runtime_env_missing(monkeypatch):
    settings = LiveKitSettings(
        _env_file=None,
        livekit_url="",
        llm_base_url="http://127.0.0.1:8000/v1",
        llm_model="served",
        livekit_api_key="devkey",
        livekit_api_secret="secret",
    )
    monkeypatch.setattr("livekit_agent.worker.get_settings", lambda: settings)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


def test_main_exits_on_model_mismatch(monkeypatch):
    settings = LiveKitSettings(
        _env_file=None,
        livekit_url="ws://127.0.0.1:7880",
        llm_base_url="http://127.0.0.1:8000/v1",
        llm_model="wrong",
        livekit_api_key="devkey",
        livekit_api_secret="secret",
    )

    async def boom(*_args, **_kwargs):
        raise LLMModelMismatch("mismatch")

    monkeypatch.setattr("livekit_agent.worker.get_settings", lambda: settings)
    monkeypatch.setattr("livekit_agent.worker.assert_llm_model", boom)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


def test_main_defaults_to_start_subcommand(monkeypatch):
    settings = LiveKitSettings(
        _env_file=None,
        livekit_url="ws://127.0.0.1:7880",
        llm_base_url="http://127.0.0.1:8000/v1",
        llm_model="served",
        livekit_api_key="devkey",
        livekit_api_secret="secret",
    )
    captured = {}

    async def ok(*_args, **_kwargs):
        return ["served"]

    class FakeCli:
        @staticmethod
        def run_app(_server):
            captured["argv"] = list(sys.argv)

    fake_agents = types.ModuleType("livekit.agents")
    fake_agents.cli = FakeCli()
    monkeypatch.setitem(sys.modules, "livekit.agents", fake_agents)
    monkeypatch.setattr("livekit_agent.worker.get_settings", lambda: settings)
    monkeypatch.setattr("livekit_agent.worker.assert_llm_model", ok)
    monkeypatch.setattr("livekit_agent.worker.build_server", lambda _settings: object())
    monkeypatch.setattr("sys.argv", ["livekit_agent.worker"])
    main()
    assert captured["argv"][-1] == "start"


def test_restaurant_voice_is_pickleable_module_function():
    from livekit_agent.worker import restaurant_voice

    assert restaurant_voice.__module__ == "livekit_agent.worker"
    assert restaurant_voice.__qualname__ == "restaurant_voice"


def test_build_server_registers_agent_name():
    pytest.importorskip("livekit.agents")
    from livekit_agent.worker import build_server

    settings = LiveKitSettings(_env_file=None, agent_name="restaurant-voice")
    server = build_server(settings)
    assert server is not None
