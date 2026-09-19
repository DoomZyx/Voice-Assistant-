from __future__ import annotations


def model_id_error(configured: str, served: list[str]) -> str | None:
    """None si LLM_MODEL est exactement un id servi par GET /v1/models."""
    requested = (configured or "").strip()
    if not requested:
        return (
            "LLM_MODEL est vide. Copier l'id exact renvoye par GET /v1/models "
            "(ex. mistralai/Mistral-7B-Instruct-v0.3), pas un alias local."
        )
    if not served:
        return "vLLM n'expose aucun modele (GET /v1/models vide). Le serveur n'a peut-etre pas fini de charger."
    if requested in served:
        return None
    return (
        f"LLM_MODEL={requested!r} ne correspond a aucun modele charge. "
        f"Modeles servis: {served}. "
        "L'id doit matcher exactement l'identifiant renvoye par vLLM."
    )
