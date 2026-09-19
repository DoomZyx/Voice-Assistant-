from app.audio.chunking import flush_speech_chunks, split_speech_chunks


def test_sentence_chunking_keeps_question_together():
    chunks = split_speech_chunks("Oui. Pour combien de personnes ?", strategy="sentence")
    assert chunks == ["Oui.", "Pour combien de personnes ?"]


def test_clause_chunking_splits_on_comma():
    chunks = split_speech_chunks(
        "J'ai de la place a vingt heures quinze, ou a vingt heures trente.",
        strategy="clause",
    )
    assert len(chunks) == 2
    assert chunks[0].startswith("J'ai de la place")


def test_word_chunking_is_rejected_for_prosody():
    chunks = split_speech_chunks("Oui je vous ecoute", strategy="word")
    assert chunks == ["Oui je vous ecoute"]


def test_flush_holds_incomplete_sentence():
    ready, rest = flush_speech_chunks("J'ai une table disponible pour quatre personnes a")
    assert ready == []
    assert "quatre personnes a" in rest
    ready, rest = flush_speech_chunks(rest + " vingt heures.")
    assert ready == ["J'ai une table disponible pour quatre personnes a vingt heures."]
    assert rest == ""
