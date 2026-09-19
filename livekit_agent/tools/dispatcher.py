from __future__ import annotations

import time
from typing import Any

from livekit_agent.observability.logging import emit_event
from livekit_agent.tools import menu as menu_tools
from livekit_agent.tools import orders as order_tools
from livekit_agent.tools import restaurant as restaurant_tools
from livekit_agent.tools.http import FastifyToolsClient
from livekit_agent.tools.reservations import not_implemented


class ToolDispatcher:
    def __init__(self, client: FastifyToolsClient, tenant_id: str) -> None:
        self._client = client
        self._tenant_id = tenant_id

    async def aclose(self) -> None:
        await self._client.aclose()

    async def dispatch(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = dict(arguments or {})
        started = time.perf_counter()
        emit_event("tool.start", tool=name, restaurant_id=self._tenant_id)
        result = await self._dispatch(name, payload)
        latency_ms = int(round((time.perf_counter() - started) * 1000))
        emit_event(
            "tool.fastify",
            tool=name,
            success=result.get("success"),
            latency_ms=latency_ms,
            restaurant_id=self._tenant_id,
        )
        return result

    async def _dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tenant_id = self._tenant_id
        if name == "get_restaurant_info":
            return await restaurant_tools.get_restaurant_info(self._client, tenant_id)
        if name == "get_menu":
            return await menu_tools.get_menu(self._client, tenant_id)
        if name == "search_product":
            return await menu_tools.search_product(
                self._client,
                tenant_id,
                str(arguments.get("query") or arguments.get("nom") or ""),
            )
        if name == "get_product":
            return await menu_tools.get_product(
                self._client,
                tenant_id,
                str(arguments.get("nom") or arguments.get("name") or ""),
            )
        if name == "check_product_availability":
            return await menu_tools.check_product_availability(
                self._client,
                tenant_id,
                str(arguments.get("nom") or arguments.get("name") or ""),
            )
        if name == "create_order":
            return await order_tools.create_order(self._client, tenant_id, arguments)
        if name == "update_order":
            return await order_tools.update_order(self._client, tenant_id, arguments)
        if name == "cancel_order":
            return await order_tools.cancel_order(self._client, tenant_id, arguments)
        if name in {
            "check_table_availability",
            "create_reservation",
            "modify_reservation",
            "cancel_reservation",
        }:
            return not_implemented(name)
        return {"success": False, "error": f"Fonction inconnue: {name}"}
