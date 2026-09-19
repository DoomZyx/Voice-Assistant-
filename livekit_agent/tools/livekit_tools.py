from __future__ import annotations

from typing import Any

from livekit_agent.tools.dispatcher import ToolDispatcher


def build_livekit_tools(dispatcher: ToolDispatcher) -> list[Any]:
    from livekit.agents import RunContext, function_tool

    @function_tool
    async def get_restaurant_info(context: RunContext) -> dict[str, Any]:
        return await dispatcher.dispatch("get_restaurant_info", {})

    @function_tool
    async def get_menu(context: RunContext) -> dict[str, Any]:
        return await dispatcher.dispatch("get_menu", {})

    @function_tool
    async def search_product(context: RunContext, query: str) -> dict[str, Any]:
        return await dispatcher.dispatch("search_product", {"query": query})

    @function_tool
    async def get_product(context: RunContext, nom: str) -> dict[str, Any]:
        return await dispatcher.dispatch("get_product", {"nom": nom})

    @function_tool
    async def check_product_availability(context: RunContext, nom: str) -> dict[str, Any]:
        return await dispatcher.dispatch("check_product_availability", {"nom": nom})

    @function_tool
    async def create_order(
        context: RunContext,
        name: str,
        telephone: str,
        date: str,
        time: str,
        commandes: list[dict[str, Any]],
        description: str = "",
    ) -> dict[str, Any]:
        return await dispatcher.dispatch(
            "create_order",
            {
                "name": name,
                "telephone": telephone,
                "date": date,
                "time": time,
                "commandes": commandes,
                "description": description,
            },
        )

    @function_tool
    async def update_order(context: RunContext, id: str, statut: str | None = None) -> dict[str, Any]:
        arguments: dict[str, Any] = {"id": id}
        if statut:
            arguments["statut"] = statut
        return await dispatcher.dispatch("update_order", arguments)

    @function_tool
    async def cancel_order(context: RunContext, id: str) -> dict[str, Any]:
        return await dispatcher.dispatch("cancel_order", {"id": id})

    return [
        get_restaurant_info,
        get_menu,
        search_product,
        get_product,
        check_product_availability,
        create_order,
        update_order,
        cancel_order,
    ]
