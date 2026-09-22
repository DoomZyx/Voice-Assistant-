from __future__ import annotations

from typing import Any

import httpx

from livekit_agent.tools.http import FastifyApiError, FastifyToolsClient

REGISTER_UNAVAILABLE = "je ne peux pas enregistrer pour le moment"
DEFAULT_SLOT_MINUTES = 60


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
    duree = arguments.get("duree") or DEFAULT_SLOT_MINUTES
    try:
        payload = await client.check_table_availability(tenant_id, date, heure, int(duree))
    except (FastifyApiError, httpx.HTTPError) as exc:
        return reservation_failure(exc)
    return {
        "success": True,
        "data": {
            "available": bool(payload.get("available")),
            "date": date,
            "heure": heure,
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
    requested = str(arguments.get("heure") or arguments.get("time") or "").strip()
    try:
        payload = await client.get_available_slots(tenant_id, date)
    except (FastifyApiError, httpx.HTTPError) as exc:
        return reservation_failure(exc)
    alternative = None
    for slot in payload.get("availableSlots") or []:
        if requested and str(slot) == requested:
            continue
        alternative = slot
        break
    return {
        "success": True,
        "data": {
            "date": date,
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
