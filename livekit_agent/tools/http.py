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
            "commandes": payload.get("commandes") or [],
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
            message = ""
            if isinstance(payload, dict):
                message = str(payload.get("error") or payload.get("message") or "")
            raise FastifyApiError(
                message or f"Echec {kind}: HTTP {response.status_code}",
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


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"raw": response.text}
