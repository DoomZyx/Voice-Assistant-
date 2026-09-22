from __future__ import annotations

from typing import Any

from livekit_agent.tools.dispatcher import ToolDispatcher

# get_type_hints() lit les globals du module, pas le import local de build_livekit_tools.
try:
    from livekit.agents import RunContext, function_tool
except ImportError:  # tests sans le SDK LiveKit
    RunContext = Any  # type: ignore[misc,assignment]
    function_tool = None


def build_livekit_tools(dispatcher: ToolDispatcher, *, profile: str = "") -> list[Any]:
    if function_tool is None:
        raise RuntimeError("livekit.agents is required to register voice tools")

    @function_tool
    async def get_restaurant_info(context: RunContext) -> dict[str, Any]:
        return await dispatcher.dispatch("get_restaurant_info", {})

    @function_tool
    async def check_table_availability(
        context: RunContext,
        date: str,
        heure: str,
        duree: int = 60,
    ) -> dict[str, Any]:
        return await dispatcher.dispatch(
            "check_table_availability",
            {"date": date, "heure": heure, "duree": duree},
        )

    @function_tool
    async def list_available_slots(
        context: RunContext,
        date: str,
        heure: str = "",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"date": date}
        if heure:
            payload["heure"] = heure
        return await dispatcher.dispatch("list_available_slots", payload)

    @function_tool
    async def create_reservation(
        context: RunContext,
        date: str,
        time: str,
        name: str,
        telephone: str = "",
        nombrePersonnes: int = 1,
        description: str = "",
    ) -> dict[str, Any]:
        return await dispatcher.dispatch(
            "create_reservation",
            {
                "date": date,
                "time": time,
                "name": name,
                "telephone": telephone,
                "nombrePersonnes": nombrePersonnes,
                "description": description,
            },
        )

    @function_tool
    async def modify_reservation(
        context: RunContext,
        id: str = "",
        date: str = "",
        heure: str = "",
        nom: str = "",
        nombrePersonnes: int | None = None,
    ) -> dict[str, Any]:
        arguments: dict[str, Any] = {}
        if id:
            arguments["id"] = id
        if date:
            arguments["date"] = date
        if heure:
            arguments["heure"] = heure
        if nom:
            arguments["nom"] = nom
        if nombrePersonnes is not None:
            arguments["nombrePersonnes"] = nombrePersonnes
        return await dispatcher.dispatch("modify_reservation", arguments)

    @function_tool
    async def cancel_reservation(context: RunContext, id: str = "") -> dict[str, Any]:
        arguments: dict[str, Any] = {}
        if id:
            arguments["id"] = id
        return await dispatcher.dispatch("cancel_reservation", arguments)

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

    if (profile or "").strip().lower() == "reservations":
        return [
            get_restaurant_info,
            check_table_availability,
            list_available_slots,
            create_reservation,
            modify_reservation,
            cancel_reservation,
        ]
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
