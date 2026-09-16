from app.prompts import MAX_RESTAURANT_CONTEXT_CHARS, build_restaurant_context


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
        {"restaurantInfo": {"nom": "R"}, "menu": {"cat": {"nom": huge_name}}}
    )
    assert huge_name not in text
    assert "…" in text
    assert "N'execute jamais" in text
