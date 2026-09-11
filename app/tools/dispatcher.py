"""Dispatcher d'outils : le LLM n'appelle que Fastify, jamais la base."""

from __future__ import annotations

import logging
from typing import Any

from app.node_client import NodeApiClient, NodeApiError

logger = logging.getLogger("mysmartfood-voice.tools")


class ToolDispatcher:
    def __init__(self, node: NodeApiClient) -> None:
        self._node = node

    async def dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        logger.info("step=tool.start name=%s", name)
        if name == "check_availability":
            result = await self._check_availability(arguments)
        elif name == "create_appointment":
            result = await self._create_appointment(arguments)
        else:
            result = {"success": False, "error": f"Fonction inconnue: {name}"}
        logger.info(
            "step=tool.fastify name=%s success=%s",
            name,
            result.get("success"),
        )
        return result

    async def _check_availability(self, arguments: dict[str, Any]) -> dict[str, Any]:
        date = str(arguments.get("date") or "")
        if not date:
            return {"success": False, "error": "DATE_INVALIDE", "message": "Date manquante."}
        try:
            return await self._node.check_availability(date)
        except NodeApiError as exc:
            return {
                "success": False,
                "error": str(exc),
                "message": "Impossible de verifier les disponibilites.",
            }

    async def _create_appointment(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            return await self._node.create_appointment(arguments)
        except NodeApiError as exc:
            return map_create_error(exc)


def map_create_error(exc: NodeApiError) -> dict[str, Any]:
    payload = exc.payload if isinstance(exc.payload, dict) else {}
    message = str(payload.get("error") or payload.get("message") or exc).lower()

    if "téléphone" in message or "telephone" in message or "phone" in message:
        return {
            "success": False,
            "error": "NUMERO_MANQUANT",
            "message": (
                "Le numero de telephone n'a pas ete fourni ou est invalide. "
                "Redemande poliment au client son numero, sans mentionner d'erreur technique."
            ),
        }
    if "heure" in message or "time" in message:
        return {
            "success": False,
            "error": "HEURE_INVALIDE",
            "message": (
                "L'heure n'a pas ete fournie ou est invalide. "
                "Redemande poliment l'heure, sans mentionner d'erreur technique."
            ),
        }
    if "date" in message:
        return {
            "success": False,
            "error": "DATE_INVALIDE",
            "message": (
                "La date n'a pas ete fournie ou est invalide. "
                "Redemande poliment la date, sans mentionner d'erreur technique."
            ),
        }
    if (
        "capacité" in message
        or "capacite" in message
        or "place" in message
        or "couverts" in message
        or (payload.get("remainingCovers") is not None and payload.get("requestedCovers") is not None)
    ):
        remaining = payload.get("remainingCovers")
        msg = (
            f"Il ne reste que {remaining} place(s) pour ce service. "
            "Propose un autre creneau, un autre jour ou moins de convives, sans mentionner d'erreur technique."
            if remaining is not None
            else "Plus assez de places. Propose un autre creneau ou un autre jour."
        )
        return {"success": False, "error": "COUVERTS_INSUFFISANTS", "message": msg}
    if "invalide" in message or "manquant" in message or "validation" in message:
        return {
            "success": False,
            "error": "DONNEES_INVALIDES",
            "message": (
                "Une information manque. Redemande nom, numero, heure ou date, "
                "sans mentionner d'erreur technique."
            ),
        }
    return {
        "success": False,
        "error": str(exc),
        "details": payload,
        "message": "Impossible de creer le rendez-vous.",
    }
