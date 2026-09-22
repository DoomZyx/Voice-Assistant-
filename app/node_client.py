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

    def _tenant_headers(self, tenant_id: str | None) -> dict[str, str]:
        value = str(tenant_id or "").strip()
        if not value:
            return {}
        return {"x-tenant-id": value}

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Centralise les appels HTTP vers Fastify.

        Une panne reseau/timeout httpx devient un NodeApiError (meme famille
        que les erreurs HTTP >= 400) au lieu d'une exception httpx brute : les
        appelants (dispatcher d'outils, /v1/tools/*) savent deja convertir un
        NodeApiError en reponse conversationnelle pour le LLM plutot que de
        laisser la session vocale planter sur une exception inattendue.
        """
        try:
            return await self._client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            raise NodeApiError(
                f"Fastify injoignable ({type(exc).__name__})",
                status_code=None,
                payload={"network_error": True},
            ) from exc

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def ping(self) -> dict[str, Any]:
        response = await self._request("GET", "/api/ping")
        self._raise_for_status(response, "ping Node")
        return response.json()

    async def get_voice_context(self, instance_id: str) -> dict[str, Any]:
        safe_instance_id = quote(instance_id, safe="")
        response = await self._request(
            "GET",
            f"/api/voice/context/{safe_instance_id}",
            headers=self._tenant_headers(instance_id),
        )
        self._raise_for_status(response, "contexte vocal")
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
            raise NodeApiError("Contexte vocal invalide", response.status_code, payload)
        return payload["data"]

    async def get_voice_snapshot(self, instance_id: str) -> dict[str, Any]:
        safe_instance_id = quote(instance_id, safe="")
        response = await self._request(
            "GET",
            f"/api/voice/context/{safe_instance_id}/snapshot",
            headers=self._tenant_headers(instance_id),
        )
        self._raise_for_status(response, "snapshot vocal")
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
            raise NodeApiError("Snapshot vocal invalide", response.status_code, payload)
        return payload["data"]

    async def check_availability(self, date: str, tenant_id: str | None = None) -> dict[str, Any]:
        headers = self._tenant_headers(tenant_id)
        orders, reservations = await asyncio.gather(
            self._request(
                "GET",
                "/api/orders/ai/available-slots",
                params={"date": date},
                headers=headers,
            ),
            self._request(
                "GET",
                "/api/reservations/ai/available-slots",
                params={"date": date},
                headers=headers,
            ),
        )
        if orders.status_code >= 400:
            raise NodeApiError(
                f"Impossible de verifier les creneaux commandes: HTTP {orders.status_code}",
                orders.status_code,
                _safe_json(orders),
            )
        orders_data = orders.json()
        resa_data = reservations.json() if reservations.status_code < 400 else {}
        return summarize_availability_for_llm(
            date=date,
            slots=orders_data.get("availableSlots") or [],
            message=orders_data.get("message") or "Disponibilites recuperees",
            remaining_covers_midi=resa_data.get("remainingCoversMidi"),
            remaining_covers_soir=resa_data.get("remainingCoversSoir"),
        )

    async def create_appointment(
        self,
        payload: dict[str, Any],
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        if _is_reservation(payload.get("type")):
            return await self.create_reservation(payload, tenant_id=tenant_id)
        return await self.create_order(payload, tenant_id=tenant_id)

    async def create_reservation(
        self,
        payload: dict[str, Any],
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        body = {
            "name": payload.get("name") or payload.get("nom"),
            "telephone": payload.get("telephone"),
            "date": payload.get("date"),
            "time": payload.get("time") or payload.get("heure"),
            "description": payload.get("description") or "",
            "nombrePersonnes": payload.get("nombrePersonnes") or 1,
            "notes_internes": payload.get("notes_internes") or "",
        }
        response = await self._request(
            "POST",
            "/api/reservations/ai/create",
            json=body,
            headers=self._tenant_headers(tenant_id),
        )
        return self._parse_create(response, "reservation")

    async def report_usage(self, payload: dict[str, Any]) -> None:
        response = await self._request(
            "POST",
            "/api/voice/usage",
            json=payload,
            headers=self._tenant_headers(payload.get("instanceId")),
        )
        if response.status_code >= 400:
            raise NodeApiError(
                f"Usage LLM: HTTP {response.status_code}",
                response.status_code,
                _safe_json(response),
            )

    async def create_order(
        self,
        payload: dict[str, Any],
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        body = {
            "name": payload.get("name") or payload.get("nom"),
            "telephone": payload.get("telephone"),
            "date": payload.get("date"),
            "time": payload.get("time") or payload.get("heure"),
            "type": ORDER_TYPE,
            "description": payload.get("description") or "",
            "commandes": payload.get("commandes") or [],
        }
        response = await self._request(
            "POST",
            "/api/orders/ai/create",
            json=body,
            headers=self._tenant_headers(tenant_id),
        )
        return self._parse_create(response, "order")

    async def report_call_ended(
        self,
        payload: dict[str, Any],
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        response = await self._request(
            "POST",
            "/api/voice/call-ended",
            json=payload,
            headers=self._tenant_headers(tenant_id or payload.get("instanceId")),
        )
        self._raise_for_status(response, "fin d'appel")
        data = _safe_json(response)
        return data if isinstance(data, dict) else {"success": True}

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
        message = None
        created_id = None
        if isinstance(data, dict):
            message = data.get("message")
            record = data.get("data")
            if isinstance(record, dict):
                raw_id = record.get("_id") or record.get("id")
                if raw_id is not None and str(raw_id).strip():
                    created_id = str(raw_id).strip()
        created_type = "reservation" if kind == "reservation" else "order"
        return {
            "success": True,
            "created": True,
            "id": created_id,
            "type": created_type,
            "message": message
            or (
                f"{kind} creee. Confirme oralement en une phrase courte (nom et heure) "
                "puis cloture. Ne lis aucun identifiant."
            ),
        }

    @staticmethod
    def _raise_for_status(response: httpx.Response, context: str) -> None:
        if response.status_code >= 400:
            raise NodeApiError(
                f"{context}: HTTP {response.status_code}",
                response.status_code,
                _safe_json(response),
            )


def _minutes_of_slot(slot: Any) -> int | None:
    parts = str(slot or "").split(":")
    if len(parts) < 2:
        return None
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
    except ValueError:
        return None
    return hours * 60 + minutes


def _format_minutes(total_minutes: int) -> str:
    hours = (total_minutes // 60) % 24
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}"


def compact_slot_windows(slots: list[Any] | None, step_minutes: int = 30) -> list[dict[str, str]]:
    minutes = sorted(
        {
            value
            for value in (_minutes_of_slot(slot) for slot in (slots or []))
            if value is not None
        }
    )
    if not minutes:
        return []
    windows: list[dict[str, str]] = []
    start = previous = minutes[0]
    for current in minutes[1:]:
        if current <= previous + step_minutes:
            previous = current
            continue
        windows.append({"debut": _format_minutes(start), "fin": _format_minutes(previous)})
        start = previous = current
    windows.append({"debut": _format_minutes(start), "fin": _format_minutes(previous)})
    return windows


def summarize_availability_for_llm(
    date: str,
    slots: list[Any] | None,
    message: str | None = None,
    remaining_covers_midi: Any = None,
    remaining_covers_soir: Any = None,
) -> dict[str, Any]:
    fenetres = compact_slot_windows(slots)
    result: dict[str, Any] = {
        "success": True,
        "date": date,
        "closed": len(fenetres) == 0,
        "fenetres": fenetres,
        "instruction": (
            "Ne lis pas les creneaux. Confirme seulement si l'heure demandee est dans une fenetre. "
            "Sinon propose une seule alternative."
        ),
    }
    if message:
        result["message"] = message
    if remaining_covers_midi is not None:
        result["remainingCoversMidi"] = remaining_covers_midi
    if remaining_covers_soir is not None:
        result["remainingCoversSoir"] = remaining_covers_soir
    return result


def _is_reservation(appointment_type: Any) -> bool:
    normalized = str(appointment_type or "").lower()
    return "reserv" in normalized or "réserv" in normalized


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"raw": response.text}
