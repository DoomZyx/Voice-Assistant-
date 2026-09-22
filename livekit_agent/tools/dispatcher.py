from __future__ import annotations

import time
from typing import Any

from livekit_agent.observability.logging import emit_event
from livekit_agent.tools import menu as menu_tools
from livekit_agent.tools import orders as order_tools
from livekit_agent.tools import reservations as reservation_tools
from livekit_agent.tools import restaurant as restaurant_tools
from livekit_agent.tools.http import FastifyToolsClient

LAST_RESERVATION_ID_KEY = "last_reservation_id"
_ORDER_TOOLS = frozenset(
    {
        "get_menu",
        "search_product",
        "get_product",
        "check_product_availability",
        "create_order",
        "update_order",
        "cancel_order",
    }
)
_SESSION_ONLY_KEYS = frozenset(
    {
        "restaurant_id",
        "restaurantId",
        "restaurant",
        "tenant_id",
        "tenantId",
        "tenant",
        "instance_id",
        "instanceId",
        "instanceid",
        "schema",
        "database",
        "db",
        "search_path",
        "x-tenant-id",
        "x_tenant_id",
    }
)
_RESERVATION_ID_KEYS = ("id", "reservation_id", "reservationId")


def _strip_session_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_session_keys(item)
            for key, item in value.items()
            if str(key) not in _SESSION_ONLY_KEYS
        }
    if isinstance(value, list):
        return [_strip_session_keys(item) for item in value]
    return value


def session_tool_arguments(arguments: dict[str, Any] | None) -> dict[str, Any]:
    stripped = _strip_session_keys(dict(arguments or {}))
    return stripped if isinstance(stripped, dict) else {}


class ToolDispatcher:
    def __init__(
        self,
        client: FastifyToolsClient,
        tenant_id: str,
        session_store: dict[str, Any] | None = None,
        profile: str = "",
    ) -> None:
        self._client = client
        self._tenant_id = tenant_id
        self._session_store = session_store if session_store is not None else {}
        self.profile = (profile or "").strip().lower()

    async def aclose(self) -> None:
        await self._client.aclose()

    def _session_reservation_id(self, arguments: dict[str, Any]) -> str | None:
        last = str(self._session_store.get(LAST_RESERVATION_ID_KEY) or "").strip()
        offered = ""
        for key in _RESERVATION_ID_KEYS:
            value = arguments.get(key)
            if value is not None and str(value).strip():
                offered = str(value).strip()
                break
        if offered:
            return offered if offered == last else None
        return last or None

    async def dispatch(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        if not str(self._tenant_id or "").strip():
            emit_event("tool.denied", tool=name, error="TENANT_UNRESOLVED")
            return {"success": False, "error": "TENANT_UNRESOLVED"}
        payload = session_tool_arguments(arguments)
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
        if self.profile == "reservations" and name in _ORDER_TOOLS:
            return {"success": False, "error": "TOOL_DISABLED"}
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
        if name == "check_table_availability":
            return await reservation_tools.check_table_availability(self._client, tenant_id, arguments)
        if name == "list_available_slots":
            return await reservation_tools.list_available_slots(self._client, tenant_id, arguments)
        if name == "create_reservation":
            result = await reservation_tools.create_reservation(
                self._client,
                tenant_id,
                arguments,
                caller_phone=str(self._session_store.get("caller_phone") or ""),
            )
            created_id = str(result.get("id") or "").strip()
            if result.get("success") and created_id:
                self._session_store[LAST_RESERVATION_ID_KEY] = created_id
            return result
        if name in {"modify_reservation", "cancel_reservation"}:
            reservation_id = self._session_reservation_id(arguments)
            if not reservation_id:
                return {
                    "success": False,
                    "error": "RESERVATION_ID_DENIED",
                    "message": "Je ne peux modifier que la reservation de cet appel.",
                }
            if name == "modify_reservation":
                return await reservation_tools.modify_reservation(
                    self._client,
                    tenant_id,
                    reservation_id,
                    arguments,
                )
            return await reservation_tools.cancel_reservation(self._client, tenant_id, reservation_id)
        return {"success": False, "error": f"Fonction inconnue: {name}"}
