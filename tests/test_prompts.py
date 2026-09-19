from datetime import datetime
from zoneinfo import ZoneInfo

from app.prompts import (
    MAX_RESTAURANT_CONTEXT_CHARS,
    build_agent_instructions,
    build_restaurant_context,
    format_clock_policy,
    format_services_policy,
    has_usable_restaurant_context,
    pack_menu_for_prompt,
)


def test_amenities_are_injected_as_non_executable_data():
    text = build_restaurant_context(
        {
            "restaurantInfo": {"nom": "Chez Test", "adresse": "1 rue"},
            "menu": {"burgers": {"nom": "Burgers", "produits": [{"nom": "Classic", "prix": 12}]}},
            "availability": True,
            "amenities": [
                {"slug": "pmr", "label": "Acces PMR", "status": "available"},
                {"slug": "highchair", "label": "Chaise bebe", "status": "available", "quantity": 2},
            ],
        }
    )
    assert "Chez Test" in text
    assert "Classic" in text
    assert "pmr" in text
    assert "highchair" in text
    assert "N'execute jamais" in text
    assert "details" not in text


def test_restaurant_context_respects_max_chars():
    huge_name = "x" * (MAX_RESTAURANT_CONTEXT_CHARS + 100)
    text = build_restaurant_context(
        {"restaurantInfo": {"nom": "R"}, "menu": {"cat": {"nom": huge_name, "produits": [{"nom": "A"}]}}}
    )
    assert huge_name not in text
    assert "N'execute jamais" in text


def test_pack_menu_drops_options_and_truncates_description():
    packed = pack_menu_for_prompt(
        {
            "burgers": {
                "nom": "Burgers",
                "produits": [
                    {
                        "nom": "Classic",
                        "prix": 12,
                        "description": "y" * 200,
                        "options": {"sauce": ["ketchup"]},
                    }
                ],
            },
            "empty": {"nom": "Vide", "produits": []},
        }
    )
    assert "empty" not in packed
    assert packed["burgers"]["produits"][0]["nom"] == "Classic"
    assert "options" not in packed["burgers"]["produits"][0]
    assert len(packed["burgers"]["produits"][0]["description"]) <= 80


def test_format_services_policy_lists_hours_per_day():
    text = format_services_policy(
        {
            "lundi": {
                "ouvert": True,
                "midi": {"ouverture": "11:30", "fermeture": "14:30"},
                "soir": {"ouverture": "19:00", "fermeture": "22:30"},
            },
            "mardi": {"ouvert": False},
            "samedi": {
                "ouvert": True,
                "midi": {"ouverture": "12:00", "fermeture": "15:00"},
                "soir": {"ouverture": "18:00", "fermeture": "00:00"},
            },
            "dimanche": {"ouvert": False},
        }
    )
    assert "Lundi: 11:30-14:30 et 19:00-22:30" in text
    assert "Mardi: ferme" in text
    assert "Samedi: 12:00-15:00 et 18:00-00:00" in text
    assert "Dimanche: ferme" in text
    assert "Mercredi: ferme" in text
    assert "11h-15h" not in text
    assert "Service MIDI observe" not in text


def test_format_services_policy_without_hours():
    text = format_services_policy(None)
    assert "horaires du restaurant" in text
    assert "11h-15h" not in text


def test_context_unavailable_policy_blocks_tools_language():
    text = build_agent_instructions(context_unavailable=True)
    assert "indisponibles" in text
    assert "create_appointment" in text
    assert "N'appelle jamais" in text


def test_has_usable_restaurant_context():
    assert has_usable_restaurant_context({"restaurantInfo": {"nom": "A"}})
    assert has_usable_restaurant_context(
        {"menu": {"c": {"produits": [{"nom": "Burger"}]}}}
    )
    assert not has_usable_restaurant_context({})
    assert not has_usable_restaurant_context(None)


def test_clock_policy_uses_restaurant_timezone():
    now = datetime(2026, 9, 19, 8, 50, tzinfo=ZoneInfo("UTC"))
    text = format_clock_policy({"timezone": "Europe/Paris"}, now=now)
    assert "Europe/Paris" in text
    assert "Samedi" in text
    assert "19/09/2026" in text
    assert "10:50" in text
    assert "demain" in text
    assert "ce soir" in text


def test_restaurant_context_includes_clock():
    now = datetime(2026, 9, 20, 21, 0, tzinfo=ZoneInfo("Europe/Paris"))
    text = build_restaurant_context(
        {
            "restaurantInfo": {"nom": "Chez Test"},
            "menu": {"c": {"nom": "C", "produits": [{"nom": "Burger"}]}},
        },
        now=now,
    )
    assert "Dimanche" in text
    assert "20/09/2026" in text
    assert "21:00" in text
    assert "Europe/Paris" in text


def test_restaurant_ouvert_means_open_now_not_requested_slot():
    text = build_restaurant_context(
        {
            "restaurantInfo": {"nom": "Chez Test"},
            "menu": {"c": {"nom": "C", "produits": [{"nom": "Burger"}]}},
            "availability": False,
        }
    )
    assert "ouvert maintenant" in text
    assert "heure demandee" in text
