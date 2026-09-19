from app.turn.detector import HeuristicTurnDetector, SilenceTurnDetector


def test_silence_detector_always_completes():
    detector = SilenceTurnDetector()
    assert detector.is_complete("Je voudrais reserver pour") is True


def test_heuristic_waits_after_trailing_preposition():
    detector = HeuristicTurnDetector()
    assert detector.is_complete("Je voudrais reserver pour") is False
    assert detector.is_complete("Je voudrais reserver pour six personnes.") is True


def test_heuristic_waits_on_correction_marker_alone():
    detector = HeuristicTurnDetector()
    assert detector.is_complete("Attendez") is False
    assert detector.is_complete("Attendez, non, plutot vingt et une heures.") is True


def test_heuristic_completes_short_confirmation():
    detector = HeuristicTurnDetector()
    assert detector.is_complete("Oui") is True
    assert detector.is_complete("Non") is True
    assert detector.is_complete("Six") is True
