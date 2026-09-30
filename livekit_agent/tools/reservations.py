from __future__ import annotations

from typing import Any

import httpx

from livekit_agent.tools.http import FastifyApiError, FastifyToolsClient

REGISTER_UNAVAILABLE = "je ne peux pas enregistrer pour le moment"
DEFAULT_SLOT_MINUTES = 60


def _party_size(arguments: dict[str, Any]) -> int | None:
    value = arguments.get("nombrePersonnes")
    if isinstance(value, bool):
        return None
    try:
        size = int(value)
    except (TypeError, ValueError):
        return None
    return size if 1 <= size <= 100 else None


def not_implemented(name: str) -> dict[str, Any]:
    return {
        "success": False,
        "error": "NOT_IMPLEMENTED",
        "message": f"{name} n'est pas active sur le chemin LiveKit pour le moment.",
    }


def reservation_failure(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return {
            "success": False,
            "error": "BACKEND_UNAVAILABLE",
            "message": REGISTER_UNAVAILABLE,
        }
    if isinstance(exc, FastifyApiError):
        status = exc.status_code or 0
        if status >= 500 or status == 0:
            return {
                "success": False,
                "error": "BACKEND_UNAVAILABLE",
                "message": REGISTER_UNAVAILABLE,
            }
        payload = exc.payload if isinstance(exc.payload, dict) else {}
        result: dict[str, Any] = {
            "success": False,
            "error": "RESERVATION_REJECTED",
            "message": str(exc) or REGISTER_UNAVAILABLE,
        }
        if payload.get("remainingCovers") is not None:
            result["remainingCovers"] = payload["remainingCovers"]
        if payload.get("requestedCovers") is not None:
            result["requestedCovers"] = payload["requestedCovers"]
        return result
    return {
        "success": False,
        "error": "BACKEND_UNAVAILABLE",
        "message": REGISTER_UNAVAILABLE,
    }


async def check_table_availability(
    client: FastifyToolsClient,
    tenant_id: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    date = str(arguments.get("date") or "").strip()
    heure = str(arguments.get("heure") or arguments.get("time") or "").strip()
    if not date or not heure:
        return {"success": False, "error": "SLOT_REQUIRED", "message": "Date et heure requises."}
    party_size = _party_size(arguments)
    if party_size is None:
        return {"success": False, "error": "PARTY_SIZE_REQUIRED", "message": "Nombre de personnes requis."}
    duree = arguments.get("duree") or DEFAULT_SLOT_MINUTES
    try:
        payload = await client.check_table_availability(tenant_id, date, heure, party_size, int(duree))
    except (FastifyApiError, httpx.HTTPError) as exc:
        return reservation_failure(exc)
    return {
        "success": True,
        "data": {
            "available": bool(payload.get("available")),
            "date": date,
            "heure": heure,
            "requestedCovers": party_size,
            "remainingCoversMidi": payload.get("remainingCoversMidi"),
            "remainingCoversSoir": payload.get("remainingCoversSoir"),
        },
    }


async def list_available_slots(
    client: FastifyToolsClient,
    tenant_id: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    date = str(arguments.get("date") or "").strip()
    if not date:
        return {"success": False, "error": "DATE_REQUIRED", "message": "Date requise."}
    party_size = _party_size(arguments)
    if party_size is None:
        return {"success": False, "error": "PARTY_SIZE_REQUIRED", "message": "Nombre de personnes requis."}
    requested = str(arguments.get("heure") or arguments.get("time") or "").strip()
    try:
        payload = await client.get_available_slots(tenant_id, date, party_size)
    except (FastifyApiError, httpx.HTTPError) as exc:
        return reservation_failure(exc)
    candidates = [str(slot) for slot in payload.get("availableSlots") or [] if str(slot) != requested]
    alternative = candidates[0] if candidates else None
    try:
        hour, minute = map(int, requested.split(":"))
        requested_minutes = hour * 60 + minute
        alternative = min(
            candidates,
            key=lambda slot: (abs(int(slot[:2]) * 60 + int(slot[3:5]) - requested_minutes), slot),
            default=None,
        )
    except (TypeError, ValueError):
        pass
    return {
        "success": True,
        "data": {
            "date": date,
            "requestedCovers": party_size,
            "alternative": alternative,
            "remainingCoversMidi": payload.get("remainingCoversMidi"),
            "remainingCoversSoir": payload.get("remainingCoversSoir"),
            "message": payload.get("message"),
        },
    }


async def create_reservation(
    client: FastifyToolsClient,
    tenant_id: str,
    arguments: dict[str, Any],
    caller_phone: str = "",
) -> dict[str, Any]:
    payload = dict(arguments)
    party_size = _party_size(payload)
    if party_size is None:
        return {"success": False, "error": "PARTY_SIZE_REQUIRED", "message": "Nombre de personnes requis."}
    payload["nombrePersonnes"] = party_size
    if not str(payload.get("telephone") or payload.get("phone") or "").strip():
        if caller_phone:
            payload["telephone"] = caller_phone
    if not str(payload.get("telephone") or "").strip():
        return {
            "success": False,
            "error": "PHONE_REQUIRED",
            "message": "Numero de telephone requis pour enregistrer la reservation.",
        }
    try:
        return await client.create_reservation(payload, tenant_id)
    except (FastifyApiError, httpx.HTTPError) as exc:
        return reservation_failure(exc)


async def modify_reservation(
    client: FastifyToolsClient,
    tenant_id: str,
    reservation_id: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    try:
        return await client.update_reservation(reservation_id, arguments, tenant_id)
    except (FastifyApiError, httpx.HTTPError) as exc:
        return reservation_failure(exc)


async def cancel_reservation(
    client: FastifyToolsClient,
    tenant_id: str,
    reservation_id: str,
) -> dict[str, Any]:
    try:
        return await client.cancel_reservation(reservation_id, tenant_id)
    except (FastifyApiError, httpx.HTTPError) as exc:
        return reservation_failure(exc)
