import pytest

from app.config import Settings
from app.llm.tools import LLM_TOOLS
from app.session.bootstrap import (
    apply_forced_caller_number,
    build_call_plan,
    normalize_french_caller_number,
)
from app.stream_token import create_stream_token
from tests.fakes import FakeNode


SECRET = "test-twilio-token"


def _settings() -> Settings:
    return Settings(voice_eager_load=False, twilio_auth_token=SECRET, greeting_text="Bonjour, je vous ecoute.")


def _start(**custom) -> dict:
    return {
        "event": "start",
        "start": {
            "streamSid": "MZ1",
            "callSid": "CA1",
            "customParameters": custom,
        },
    }


class TrackingNode(FakeNode):
    def __init__(self) -> None:
        super().__init__()
        self.context_requests: list[str] = []

    async def get_voice_context(self, instance_id: str) -> dict:
        self.context_requests.append(instance_id)
        return await super().get_voice_context(instance_id)


@pytest.mark.asyncio
async def test_missing_stream_token_rejects_and_skips_context():
    node = TrackingNode()
    node.voice_contexts["restaurant-42"] = {"restaurantInfo": {"nom": "Secret"}}
    plan = await build_call_plan(_start(instanceId="restaurant-42"), _settings(), node)
    assert plan.rejected is True
    assert plan.reject_reason == "token absent"
    assert plan.instance_id is None
    assert plan.tools_enabled is False
    assert node.context_requests == []
    assert "Secret" not in plan.system_prompt


@pytest.mark.asyncio
async def test_valid_token_loads_restaurant_context_and_enables_tools():
    node = TrackingNode()
    node.voice_contexts["restaurant-42"] = {
        "restaurantInfo": {"nom": "Chez Test"},
        "menu": {"tacos": {"nom": "Tacos", "produits": [{"nom": "Tacos poulet", "prix": 9}]}},
    }
    plan = await build_call_plan(
        _start(
            instanceId="restaurant-42",
            streamToken=create_stream_token("restaurant-42", SECRET),
            callerNumber="+33612345678",
        ),
        _settings(),
        node,
    )
    assert plan.rejected is False
    assert plan.instance_id == "restaurant-42"
    assert plan.caller_number == "0612345678"
    assert plan.context_ready is True
    assert plan.tools_enabled is True
    assert plan.greeting == "Chez Test, bonjour."
    assert "Chez Test" in plan.system_prompt
    assert "standard generique" in plan.system_prompt
    assert "Oui. Pour combien de personnes ?" not in plan.system_prompt
    assert plan.seed_messages[0]["role"] == "system"
    assert plan.llm_tools == LLM_TOOLS
    assert node.context_requests == ["restaurant-42"]


@pytest.mark.asyncio
async def test_unusable_context_disables_tools_and_uses_degraded_greeting():
    node = TrackingNode()
    node.voice_contexts["restaurant-42"] = {}
    plan = await build_call_plan(
        _start(
            instanceId="restaurant-42",
            StreamToken=create_stream_token("restaurant-42", SECRET),
        ),
        _settings(),
        node,
    )
    assert plan.rejected is False
    assert plan.context_ready is False
    assert plan.tools_enabled is False
    assert plan.llm_tools == []
    assert plan.last_error == "voice_context_unavailable"
    assert "acces au menu" in plan.greeting


def test_forced_caller_number_overrides_llm_telephone():
    updated = apply_forced_caller_number(
        {"name": "Alice", "telephone": "0000000000"},
        "0612345678",
    )
    assert updated["telephone"] == "0612345678"
    assert updated["name"] == "Alice"


def test_forced_caller_number_is_noop_without_verified_number():
    original = {"telephone": "0000000000"}
    assert apply_forced_caller_number(original, None) == original


def test_normalize_stays_shared():
    assert normalize_french_caller_number("+33612345678") == "0612345678"
