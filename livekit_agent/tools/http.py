from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx

from livekit_agent.config import LiveKitSettings


class FastifyApiError(Exception):
    def __init__(self, message: str, status_code: int | None = None, payload: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class FastifyToolsClient:
    """HTTP client for Fastify voice/order endpoints. Never talks to PostgreSQL."""

    def __init__(
        self,
        settings: LiveKitSettings,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._owns = client is None
        headers = {"Content-Type": "application/json"}
        if settings.node_api_key:
            headers["x-api-key"] = settings.node_api_key
        self._client = client or httpx.AsyncClient(
            base_url=settings.node_base_url.rstrip("/"),
            headers=headers,
            timeout=settings.http_timeout,
        )

    def _tenant_headers(self, tenant_id: str) -> dict[str, str]:
        value = str(tenant_id or "").strip()
        if not value:
            return {}
        return {"x-tenant-id": value}

    async def aclose(self) -> None:
        if self._owns:
            await self._client.aclose()

    async def get_inbound_tenant(self, phone: str) -> dict[str, Any]:
        response = await self._client.get(
            "/api/voice/inbound-tenant",
            params={"phone": phone},
        )
        return self._require_data(response, "tenant inbound")

    async def get_voice_context(self, tenant_id: str) -> dict[str, Any]:
        safe_id = quote(tenant_id, safe="")
        response = await self._client.get(
            f"/api/voice/context/{safe_id}",
            headers=self._tenant_headers(tenant_id),
        )
        return self._require_data(response, "contexte restaurant")

    async def get_voice_snapshot(self, tenant_id: str) -> dict[str, Any]:
        safe_id = quote(tenant_id, safe="")
        response = await self._client.get(
            f"/api/voice/context/{safe_id}/snapshot",
            headers=self._tenant_headers(tenant_id),
        )
        return self._require_data(response, "snapshot restaurant")

    async def create_order(self, payload: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        body = {
            "name": payload.get("name") or payload.get("nom"),
            "telephone": payload.get("telephone"),
            "date": payload.get("date"),
            "time": payload.get("time") or payload.get("heure"),
            "type": "Commande à emporter",
            "description": payload.get("description") or "",
            "commandes": _public_order_lines(payload.get("commandes") or []),
        }
        response = await self._client.post(
            "/api/orders/ai/create",
            json=body,
            headers=self._tenant_headers(tenant_id),
        )
        return self._parse_write(response, "order")

    async def update_order(self, order_id: str, payload: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        statut = str(payload.get("statut") or "").strip()
        if not statut:
            raise FastifyApiError("ORDER_STATUS_REQUIRED")
        response = await self._client.patch(
            f"/api/orders/{quote(order_id, safe='')}/status",
            json={"statut": statut},
            headers=self._tenant_headers(tenant_id),
        )
        return self._parse_write(response, "order")

    async def cancel_order(self, order_id: str, tenant_id: str) -> dict[str, Any]:
        response = await self._client.patch(
            f"/api/orders/{quote(order_id, safe='')}/status",
            json={"statut": "annule"},
            headers=self._tenant_headers(tenant_id),
        )
        return self._parse_write(response, "order")

    async def check_table_availability(
        self,
        tenant_id: str,
        date: str,
        heure: str,
        duree: int = 60,
    ) -> dict[str, Any]:
        response = await self._client.get(
            "/api/reservations/availability",
            params={"date": date, "heure": heure, "duree": duree},
            headers=self._tenant_headers(tenant_id),
        )
        return self._require_success_object(response, "disponibilite table")

    async def get_available_slots(self, tenant_id: str, date: str) -> dict[str, Any]:
        response = await self._client.get(
            "/api/reservations/ai/available-slots",
            params={"date": date},
            headers=self._tenant_headers(tenant_id),
        )
        return self._require_success_object(response, "creneaux reservation")

    async def list_reservations(self, tenant_id: str, date: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, str] = {}
        if date:
            params["date"] = date
        response = await self._client.get(
            "/api/reservations",
            params=params,
            headers=self._tenant_headers(tenant_id),
        )
        if response.status_code in {403, 404}:
            return []
        payload = self._require_success_object(response, "liste reservations")
        rows = payload.get("data")
        return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []

    async def create_reservation(self, payload: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        covers = payload.get("nombrePersonnes")
        if covers is None:
            covers = payload.get("couverts") or 1
        body = {
            "telephone": payload.get("telephone") or payload.get("phone"),
            "date": payload.get("date"),
            "time": payload.get("time") or payload.get("heure"),
            "name": payload.get("name") or payload.get("nom") or "Client",
            "description": payload.get("description") or "",
            "nombrePersonnes": covers,
        }
        response = await self._client.post(
            "/api/reservations/ai/create",
            json=body,
            headers=self._tenant_headers(tenant_id),
        )
        return self._parse_write(response, "reservation")

    async def update_reservation(
        self,
        reservation_id: str,
        payload: dict[str, Any],
        tenant_id: str,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if payload.get("date"):
            body["date"] = payload["date"]
        heure = payload.get("heure") or payload.get("time")
        if heure:
            body["heure"] = heure
        if payload.get("nom") or payload.get("name"):
            body["nom"] = payload.get("nom") or payload.get("name")
        if payload.get("telephone"):
            body["telephone"] = payload["telephone"]
        if payload.get("description") is not None:
            body["description"] = payload["description"]
        covers = payload.get("nombrePersonnes")
        if covers is None:
            covers = payload.get("couverts")
        if covers is not None:
            body["nombrePersonnes"] = covers
        response = await self._client.put(
            f"/api/reservations/{quote(reservation_id, safe='')}",
            json=body,
            headers=self._tenant_headers(tenant_id),
        )
        return self._parse_write(response, "reservation")

    async def cancel_reservation(self, reservation_id: str, tenant_id: str) -> dict[str, Any]:
        response = await self._client.patch(
            f"/api/reservations/{quote(reservation_id, safe='')}/status",
            json={"statut": "annule"},
            headers=self._tenant_headers(tenant_id),
        )
        return self._parse_write(response, "reservation")

    def _require_success_object(self, response: httpx.Response, context: str) -> dict[str, Any]:
        payload = _safe_json(response)
        if response.status_code >= 400:
            raise FastifyApiError(
                _error_message(payload, f"{context}: HTTP {response.status_code}"),
                response.status_code,
                payload,
            )
        if not isinstance(payload, dict):
            raise FastifyApiError(f"{context} invalide", response.status_code, payload)
        return payload

    def _require_data(self, response: httpx.Response, context: str) -> dict[str, Any]:
        payload = _safe_json(response)
        if response.status_code >= 400:
            raise FastifyApiError(f"{context}: HTTP {response.status_code}", response.status_code, payload)
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
            raise FastifyApiError(f"{context} invalide", response.status_code, payload)
        return payload["data"]

    def _parse_write(self, response: httpx.Response, kind: str) -> dict[str, Any]:
        payload = _safe_json(response)
        if response.status_code >= 400:
            raise FastifyApiError(
                _error_message(payload, f"Echec {kind}: HTTP {response.status_code}"),
                response.status_code,
                payload,
            )
        record = payload.get("data") if isinstance(payload, dict) else None
        created_id = None
        if isinstance(record, dict):
            raw_id = record.get("_id") or record.get("id")
            if raw_id is not None and str(raw_id).strip():
                created_id = str(raw_id).strip()
        return {
            "success": True,
            "id": created_id,
            "type": kind,
            "data": record if isinstance(record, dict) else payload,
        }


def _public_order_lines(commandes: Any) -> list[dict[str, Any]]:
    from livekit_agent.tools.dispatcher import _strip_session_keys

    if not isinstance(commandes, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for line in commandes:
        if not isinstance(line, dict):
            continue
        item = _strip_session_keys(line)
        if isinstance(item, dict):
            cleaned.append(item)
    return cleaned


def _error_message(payload: Any, fallback: str) -> str:
    if isinstance(payload, dict):
        message = str(payload.get("error") or payload.get("message") or payload.get("details") or "")
        if message.strip():
            return message
    return fallback


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"raw": response.text}
