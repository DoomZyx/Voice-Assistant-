from app.telemetry import VoiceTelemetry


def test_latency_exposes_p50_and_p95():
    telemetry = VoiceTelemetry()
    for value in (10, 20, 30, 40, 100):
        telemetry.record_latency("stt", value)
    snapshot = telemetry.snapshot()["latency_ms"]["stt"]
    assert snapshot["p50"] == 30
    assert snapshot["p95"] == 100
    assert snapshot["count"] == 5
