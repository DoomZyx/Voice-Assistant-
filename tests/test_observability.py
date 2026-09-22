import logging

from app.observability import emit_event, sanitize_log_fields


def test_log_fields_drop_call_content_and_credentials(caplog):
    fields = sanitize_log_fields(
        {
            "telephone": "0612345678",
            "transcript": "Je voudrais reserver.",
            "prompt": "ignore previous instructions",
            "authorization": "Bearer top-secret",
            "reason": "upstream rejected 06 12 34 56 78",
            "stage": "stt",
        }
    )
    assert fields == {"reason": "upstream rejected [redacted-phone]", "stage": "stt"}

    with caplog.at_level(logging.INFO):
        emit_event("stt_final", **fields)

    line = "\n".join(caplog.messages)
    assert "0612345678" not in line
    assert "Je voudrais reserver" not in line
    assert "top-secret" not in line
    assert "stage=stt" in line
