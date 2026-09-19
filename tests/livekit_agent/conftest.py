import pytest


ISOLATED_ENV = (
    "VOICE_TOOLS_ENABLED",
    "LIVEKIT_KOKORO_SPEED",
    "SIP_DID_TENANT_MAP",
    "LIVEKIT_URL",
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "RESTAURANT_ID",
    "RESTAURANT_NAME",
)


@pytest.fixture(autouse=True)
def isolate_livekit_env(monkeypatch):
    for key in ISOLATED_ENV:
        monkeypatch.delenv(key, raising=False)
