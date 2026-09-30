from livekit_agent.prompts import (
    build_full_instructions,
    build_phase4_instructions,
    build_reservation_instructions,
    build_tool_instructions,
)


def test_phase4_prompt_has_no_menu_and_no_tools():
    text = build_phase4_instructions(restaurant_name="Le Comptoir")
    assert "Le Comptoir" in text
    assert "francais" in text.casefold() or "français" in text.casefold()
    assert "menu" not in text.casefold()
    assert "get_restaurant_info" not in text
    assert "create_order" not in text
    assert "json" not in text.casefold()


def test_tool_prompt_does_not_embed_catalog():
    text = build_tool_instructions(restaurant_name="Le Comptoir")
    assert "get_restaurant_info" in text
    assert "get_menu" in text
    assert '"prix"' not in text
    assert "produits" not in text


def test_reservation_prompt_has_no_order_tools():
    text = build_reservation_instructions(restaurant_name="Le Comptoir")
    assert "Le Comptoir" in text
    assert "create_reservation" in text
    assert "check_table_availability" in text
    assert "list_available_slots" in text
    assert "create_order" not in text
    assert "get_menu" not in text
    assert "json" not in text.casefold()


def test_full_prompt_covers_requests_and_requires_write_success():
    text = build_full_instructions(restaurant_name="Chez Alice")
    assert "Chez Alice" in text
    assert "create_reservation" in text
    assert "create_order" in text
    assert "get_restaurant_info" in text
    assert "chaises hautes" in text
    assert "success=true" in text
