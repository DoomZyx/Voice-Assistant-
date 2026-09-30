from __future__ import annotations

from livekit_agent.tools.dispatcher import ToolDispatcher
from livekit_agent.tools.livekit_tools import build_livekit_tools


def build_restaurant_agent(
    *,
    instructions: str,
    greeting: str,
    dispatcher: ToolDispatcher | None = None,
    tools_profile: str = "",
):
    from livekit.agents import Agent

    tools = (
        build_livekit_tools(dispatcher, profile=tools_profile)
        if dispatcher is not None
        else []
    )

    class RestaurantAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions=instructions, tools=tools)
            self._greeting = greeting

        async def on_enter(self) -> None:
            await self.session.say(self._greeting)

    return RestaurantAgent()
