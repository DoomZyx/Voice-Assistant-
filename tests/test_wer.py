from benchmarks.wer import normalize_transcript, summarize_conditions, word_error_rate


def test_wer_identical_is_zero():
    assert word_error_rate("table pour quatre", "table pour quatre") == 0.0


def test_wer_ignores_punctuation_and_case():
    assert word_error_rate("C'est au nom de Schmitt.", "cest au nom de schmitt") == 0.0


def test_wer_substitution():
    wer = word_error_rate("table pour quatre", "table pour six")
    assert 0.3 < wer < 0.4


def test_normalize_keeps_french_names():
    assert "nguyen" in normalize_transcript("Monsieur Nguyen")
    assert "schmitt" in normalize_transcript("Schmitt")


def test_summarize_conditions_splits_clean_and_street():
    details = [
        {"condition": "clean", "wer": 0.2, "latency_ms": 1000, "hypothesis": "ok"},
        {"condition": "clean", "wer": 0.4, "latency_ms": 2000, "hypothesis": "ok"},
        {"condition": "street", "wer": 1.0, "latency_ms": 5000, "hypothesis": ""},
    ]
    summary = summarize_conditions(details)
    assert summary["clean"]["samples"] == 2
    assert summary["clean"]["wer_mean"] == 0.3
    assert summary["clean"]["empty_hypotheses"] == 0
    assert summary["street"]["empty_hypotheses"] == 1
