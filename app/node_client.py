"""Client HTTP vers les endpoints AI Fastify. Aucune logique metier ici."""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import quote

import httpx

from app.config import Settings

RESERVATION_TYPE = "Réservation de table"
ORDER_TYPE = "Commande à emporter"


class NodeApiError(Exception):
    def __init__(self, message: str, status_code: int | None = None, payload: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class NodeApiClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self._settings = settings
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=settings.node_base_url.rstrip("/"),
            headers=settings.node_headers(),
            timeout=15.0,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def ping(self) -> dict[str, Any]:
        response = await self._client.get("/api/ping")
        self._raise_for_status(response, "ping Node")
        return response.json()

    async def get_voice_context(self, instance_id: str) -> dict[str, Any]:
        safe_instance_id = quote(instance_id, safe="")
        response = await self._client.get(
            f"/api/voice/context/{safe_instance_id}",
        )
        self._raise_for_status(response, "contexte vocal")
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
            raise NodeApiError("Contexte vocal invalide", response.status_code, payload)
        return payload["data"]

    async def check_availability(self, date: str) -> dict[str, Any]:
        orders, reservations = await asyncio.gather(
            self._client.get("/api/orders/ai/available-slots", params={"date": date}),
            self._client.get("/api/reservations/ai/available-slots", params={"date": date}),
        )
        if orders.status_code >= 400:
            raise NodeApiError(
                f"Impossible de verifier les creneaux commandes: HTTP {orders.status_code}",
                orders.status_code,
                _safe_json(orders),
            )
        orders_data = orders.json()
        result: dict[str, Any] = {
            "success": True,
            "date": date,
            "slots": orders_data.get("availableSlots") or [],
            "message": orders_data.get("message") or "Disponibilites recuperees",
        }
        if reservations.status_code < 400:
            resa_data = reservations.json()
            if resa_data.get("remainingCoversMidi") is not None:
                result["remainingCoversMidi"] = resa_data["remainingCoversMidi"]
            if resa_data.get("remainingCoversSoir") is not None:
                result["remainingCoversSoir"] = resa_data["remainingCoversSoir"]
        return result

    async def create_appointment(self, payload: dict[str, Any]) -> dict[str, Any]:
        if _is_reservation(payload.get("type")):
            return await self.create_reservation(payload)
        return await self.create_order(payload)

    async def create_reservation(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = {
            "name": payload.get("name") or payload.get("nom"),
            "telephone": payload.get("telephone"),
            "date": payload.get("date"),
            "time": payload.get("time") or payload.get("heure"),
            "description": payload.get("description") or "",
            "nombrePersonnes": payload.get("nombrePersonnes") or 1,
            "notes_internes": payload.get("notes_internes") or "",
        }
        response = await self._client.post("/api/reservations/ai/create", json=body)
        return self._parse_create(response, "reservation")

    async def create_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = {
            "name": payload.get("name") or payload.get("nom"),
            "telephone": payload.get("telephone"),
            "date": payload.get("date"),
            "time": payload.get("time") or payload.get("heure"),
            "type": ORDER_TYPE,
            "description": payload.get("description") or "",
            "commandes": payload.get("commandes") or [],
        }
        response = await self._client.post("/api/orders/ai/create", json=body)
        return self._parse_create(response, "commande")

    def _parse_create(self, response: httpx.Response, kind: str) -> dict[str, Any]:
        data = _safe_json(response)
        if response.status_code >= 400:
            message = ""
            if isinstance(data, dict):
                message = str(data.get("error") or data.get("message") or "")
            raise NodeApiError(
                message or f"Echec creation {kind}: HTTP {response.status_code}",
                response.status_code,
                data,
            )
        record = None
        if isinstance(data, dict):
            record = data.get("data") or data.get("appointment")
        return {
            "success": True,
            "appointment": record,
            "message": (data.get("message") if isinstance(data, dict) else None)
            or f"{kind} creee",
        }

    @staticmethod
    def _raise_for_status(response: httpx.Response, context: str) -> None:
        if response.status_code >= 400:
            raise NodeApiError(
                f"{context}: HTTP {response.status_code}",
                response.status_code,
                _safe_json(response),
            )


def _is_reservation(appointment_type: Any) -> bool:
    normalized = str(appointment_type or "").lower()
    return "reserv" in normalized or "réserv" in normalized


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"raw": response.text}
