from app.twilio.protocol import clear_stream, media_out, parse_event


def test_parse_start():
    data = parse_event('{"event":"start","start":{"streamSid":"MZ1"}}')
    assert data["event"] == "start"
    assert data["start"]["streamSid"] == "MZ1"


def test_parse_start_bytes():
    data = parse_event(b'{"event":"start","start":{"streamSid":"MZ1"}}')
    assert data["event"] == "start"


def test_media_out_and_clear():
    assert '"event": "media"' in media_out("MZ1", "abc")
    assert '"event": "clear"' in clear_stream("MZ1")
