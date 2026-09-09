"""Schemas outils (format Chat Completions). Le LLM n'accede jamais a la base."""

CHECK_AVAILABILITY = {
    "type": "function",
    "function": {
        "name": "check_availability",
        "description": (
            "Verifier les creneaux et places restantes pour une date YYYY-MM-DD. "
            "La reponse peut contenir remainingCoversMidi et remainingCoversSoir. "
            "Deux services : MIDI 11h-15h, SOIR 18h-00h."
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
            "Reservation : nombrePersonnes obligatoire, commandes []. "
            "Commande a emporter : commandes = liste de plats (nom + quantite)."
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
                    "description": "Heure HH:MM. MIDI 11:00-14:59, SOIR 18:00-23:59",
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
