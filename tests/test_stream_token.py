import time

from app.stream_token import create_stream_token, stream_token_failure_reason

TENANT = "ff3e05bf-2da5-465c-a7a2-f635021f49a9"
SECRET = "test-twilio-token"


def test_round_trip_validates():
    token = create_stream_token(TENANT, SECRET)
    assert stream_token_failure_reason(TENANT, token, SECRET) is None


def test_rejects_missing_token():
    assert stream_token_failure_reason(TENANT, "", SECRET) == "token absent"


def test_rejects_other_tenant():
    token = create_stream_token(TENANT, SECRET)
    assert stream_token_failure_reason("other-tenant", token, SECRET) == "tenant mismatch"


def test_rejects_bad_signature():
    token = create_stream_token(TENANT, SECRET)
    bad = token[:-4] + "dead"
    assert stream_token_failure_reason(TENANT, bad, SECRET) == "signature invalide"


def test_rejects_expired():
    token = create_stream_token(TENANT, SECRET, now=int(time.time()) - 400, ttl_seconds=60)
    assert stream_token_failure_reason(TENANT, token, SECRET) == "token expire"
