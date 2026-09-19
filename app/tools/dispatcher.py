"""Dispatcher d'outils : le LLM n'appelle que Fastify, jamais la base."""

from __future__ import annotations

import logging
import unicodedata
from typing import Any

from app.node_client import NodeApiClient, NodeApiError

logger = logging.getLogger("mysmartfood-voice.tools")


def _norm_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.casefold().split())


class ToolDispatcher:
    def __init__(self, node: NodeApiClient) -> None:
        self._node = node

    async def dispatch(
        self,
        name: str,
        arguments: dict[str, Any],
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        logger.info("step=tool.start name=%s", name)
        if name == "check_availability":
            result = await self._check_availability(arguments, tenant_id)
        elif name == "create_appointment":
            result = await self._create_appointment(arguments, tenant_id)
        else:
            result = {"success": False, "error": f"Fonction inconnue: {name}"}
        logger.info(
            "step=tool.fastify name=%s success=%s",
            name,
            result.get("success"),
        )
        return result

    async def _check_availability(
        self,
        arguments: dict[str, Any],
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        date = str(arguments.get("date") or "")
        if not date:
            return {"success": False, "error": "DATE_INVALIDE", "message": "Date manquante."}
        try:
            return await self._node.check_availability(date, tenant_id=tenant_id)
        except NodeApiError as exc:
            return {
                "success": False,
                "error": str(exc),
                "message": "Impossible de verifier les disponibilites.",
            }

    async def _create_appointment(
        self,
        arguments: dict[str, Any],
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        gate = await self._gate_live_availability(arguments, tenant_id)
        if gate is not None:
            return gate
        try:
            return await self._node.create_appointment(arguments, tenant_id=tenant_id)
        except NodeApiError as exc:
            return map_create_error(exc)

    async def _gate_live_availability(
        self,
        arguments: dict[str, Any],
        tenant_id: str | None,
    ) -> dict[str, Any] | None:
        """Revalide les produits a emporter avant create (anti stale mid-call).

        Ne bloque PAS sur isOpenNow : une resa/commande pour plus tard reste valide
        meme si le restaurant est ferme au moment de l'appel.
        """
        if not tenant_id:
            return None
        get_snapshot = getattr(self._node, "get_voice_snapshot", None)
        if not callable(get_snapshot):
            return None
        appointment_type = str(arguments.get("type") or "").lower()
        is_order = "commande" in appointment_type or "emporter" in appointment_type
        if not is_order:
            return None
        commandes = arguments.get("commandes") or []
        if not isinstance(commandes, list) or not commandes:
            return None
        try:
            snapshot = await get_snapshot(tenant_id)
        except NodeApiError:
            return {
                "success": False,
                "error": "SNAPSHOT_INDISPONIBLE",
                "message": (
                    "Impossible de verifier le menu en direct. "
                    "Propose de rappeler plus tard, sans detail technique."
                ),
            }
        if not isinstance(snapshot, dict):
            return None
        products = snapshot.get("products") or []
        available = {
            _norm_name(item.get("nom"))
            for item in products
            if isinstance(item, dict) and item.get("nom")
        }
        missing: list[str] = []
        for line in commandes:
            if not isinstance(line, dict):
                continue
            nom = str(line.get("nom") or "").strip()
            if not nom:
                continue
            if _norm_name(nom) not in available:
                missing.append(nom)
        if missing:
            return {
                "success": False,
                "error": "PRODUIT_INDISPONIBLE",
                "missing": missing,
                "message": (
                    f"Plat(s) indisponible(s) : {', '.join(missing)}. "
                    "Propose un autre plat du menu, sans inventer."
                ),
            }
        return None


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
