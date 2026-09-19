"""Schemas outils (format Chat Completions). Le LLM n'accede jamais a la base."""

CHECK_AVAILABILITY = {
    "type": "function",
    "function": {
        "name": "check_availability",
        "description": (
            "Verifier si une date est ouverte. La reponse contient des fenetres debut/fin, "
            "pas une liste a lire, plus remainingCoversMidi et remainingCoversSoir. "
            "Les plages MIDI/SOIR viennent des horaires du restaurant, pas de valeurs inventees. "
            "Ne recites jamais tous les creneaux. "
            "A n'utiliser que si l'heure manque, est hors horaires, ou si create_appointment "
            "renvoie COUVERTS_INSUFFISANTS."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Date au format YYYY-MM-DD",
                }
            },
            "required": ["date"],
        },
    },
}

CREATE_APPOINTMENT = {
    "type": "function",
    "function": {
        "name": "create_appointment",
        "description": (
            "Creer une reservation de table ou une commande a emporter apres confirmation client. "
            "Ne pas inventer de plat. Reservation : nombrePersonnes obligatoire, commandes []. "
            "Commande a emporter : commandes = liste de plats (nom + quantite) presents au menu. "
            "Le serveur revalide ouverture et disponibilite produits juste avant creation. "
            "Apres succes, confirme en une phrase courte sans lire d'identifiant."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Nom complet du client"},
                "telephone": {
                    "type": "string",
                    "description": "Telephone au format 06 12 34 56 78",
                },
                "date": {"type": "string", "description": "Date YYYY-MM-DD"},
                "time": {
                    "type": "string",
                    "description": "Heure HH:MM dans une plage d'ouverture du restaurant",
                },
                "type": {
                    "type": "string",
                    "enum": ["Commande à emporter", "Réservation de table"],
                },
                "description": {"type": "string"},
                "nombrePersonnes": {
                    "type": "integer",
                    "description": "Obligatoire pour une reservation de table",
                },
                "commandes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "nom": {"type": "string"},
                            "quantite": {"type": "integer"},
                            "categorie": {"type": "string"},
                            "composition": {"type": "string"},
                            "options": {"type": "object"},
                        },
                        "required": ["nom", "quantite"],
                    },
                },
            },
            "required": ["name", "telephone", "date", "time", "type"],
        },
    },
}

LLM_TOOLS = [CHECK_AVAILABILITY, CREATE_APPOINTMENT]
