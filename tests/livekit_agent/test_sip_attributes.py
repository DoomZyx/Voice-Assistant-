from types import SimpleNamespace

from livekit_agent.config import LiveKitSettings
from livekit_agent.session import extract_call_identity, first_sip_attributes, parse_did_map


def test_parse_did_map_rejects_invalid_payloads():
    assert parse_did_map("") == {}
    assert parse_did_map("not-json") == {}
    assert parse_did_map("[1, 2]") == {}
    assert parse_did_map('{"+33111": "tenant-1"}') == {"+33111": "tenant-1"}


def test_first_sip_attributes_reads_remote_participants():
    empty = SimpleNamespace(remote_participants={})
    assert first_sip_attributes(empty) == {}
    room = SimpleNamespace(
        remote_participants={
            "sip": SimpleNamespace(attributes={"sip.to": "+33111111111", "sip.callId": "X"}),
        }
    )
    assert first_sip_attributes(room)["sip.callId"] == "X"


def test_identity_accepts_alt_sip_keys():
    settings = LiveKitSettings(_env_file=None, restaurant_id="dev")
    identity = extract_call_identity(
        settings,
        room_name="call-1",
        sip_attributes={"sip.callId": "ALT", "sip.from": "+33600", "sip.to": "+33111"},
    )
    assert identity.call_id == "ALT"
    assert identity.caller_number == "+33600"
    assert identity.trunk_number == "+33111"
