import pytest


@pytest.mark.asyncio
async def test_restaurant_agent_greets_and_has_no_tools_without_dispatcher():
    pytest.importorskip("livekit.agents")
    from livekit_agent.restaurant_agent import build_restaurant_agent

    seen: list[str] = []

    class FakeSession:
        async def generate_reply(self, instructions: str) -> None:
            seen.append(instructions)

    agent = build_restaurant_agent(
        instructions="phase4",
        greeting="Bonjour, je vous ecoute.",
        dispatcher=None,
    )
    assert list(agent.tools) == []
    type(agent).session = property(lambda self: FakeSession())
    await agent.on_enter()
    assert seen == ["Bonjour, je vous ecoute."]
    assert agent._greeting == "Bonjour, je vous ecoute."
