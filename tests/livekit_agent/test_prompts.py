from livekit_agent.prompts import build_phase4_instructions, build_tool_instructions


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
