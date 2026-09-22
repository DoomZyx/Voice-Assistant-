from livekit_agent.config import LiveKitSettings
from livekit_agent.session import CallIdentity, extract_call_identity


def test_identity_from_sip_attributes_and_room():
    settings = LiveKitSettings(_env_file=None, restaurant_id="fallback-tenant")
    identity = extract_call_identity(
        settings,
        room_name="call-abc",
        sip_attributes={
            "sip.callID": "CA123",
            "sip.phoneNumber": "+33600000000",
            "sip.trunkPhoneNumber": "+33111111111",
        },
    )
    assert identity == CallIdentity(
        call_id="CA123",
        session_id="call-abc",
        restaurant_id="",
        caller_number="+33600000000",
        trunk_number="+33111111111",
    )


def test_sip_did_waits_for_backend_and_ignores_static_map():
    settings = LiveKitSettings(
        _env_file=None,
        restaurant_id="fallback-tenant",
        sip_did_tenant_map='{"+33111111111": "tenant-42"}',
    )
    identity = extract_call_identity(
        settings,
        room_name="call-xyz",
        sip_attributes={"sip.trunkPhoneNumber": "+33 111 111 111", "sip.callID": "sip-1"},
    )
    assert identity.restaurant_id == ""
    assert identity.call_id == "sip-1"
    assert identity.trunk_number == "+33111111111"


def test_caller_number_never_selects_tenant():
    settings = LiveKitSettings(
        _env_file=None,
        restaurant_id="fallback-tenant",
        sip_did_tenant_map='{"+33600000000": "tenant-leaked"}',
    )
    identity = extract_call_identity(
        settings,
        room_name="call-xyz",
        sip_attributes={
            "sip.trunkPhoneNumber": "+33111111111",
            "sip.phoneNumber": "+33600000000",
        },
    )
    assert identity.restaurant_id == ""
    assert identity.caller_number == "+33600000000"


def test_unknown_did_does_not_use_default_tenant_when_map_exists():
    settings = LiveKitSettings(
        _env_file=None,
        restaurant_id="fallback-tenant",
        sip_did_tenant_map='{"+33111111111": "tenant-a"}',
    )
    identity = extract_call_identity(
        settings,
        room_name="call-unknown",
        sip_attributes={"sip.trunkPhoneNumber": "+33999999999", "sip.callID": "sip-x"},
    )
    assert identity.restaurant_id == ""
    assert identity.trunk_number == "+33999999999"


def test_caller_or_llm_fields_never_become_restaurant_id():
    settings = LiveKitSettings(_env_file=None, restaurant_id="fallback-tenant")
    identity = extract_call_identity(
        settings,
        room_name="call-xyz",
        sip_attributes={
            "sip.trunkPhoneNumber": "+33111111111",
            "restaurant_id": "tenant-llm",
            "tenant_id": "tenant-llm",
            "instance_id": "tenant-llm",
            "instanceId": "tenant-llm",
        },
    )
    assert identity.restaurant_id == ""
    assert identity.tenant_id == ""
    assert identity.instance_id == ""
    assert identity.trunk_number == "+33111111111"


def test_job_identity_is_bound_once():
    from types import SimpleNamespace

    from livekit_agent.session import CallIdentity, bind_job_identity, read_job_identity

    ctx = SimpleNamespace()
    first = CallIdentity(call_id="CA1", session_id="call-1", restaurant_id="tenant-a")
    second = CallIdentity(call_id="CA2", session_id="call-1", restaurant_id="tenant-b")
    assert bind_job_identity(ctx, first) is first
    assert bind_job_identity(ctx, second) is first
    assert read_job_identity(ctx) == first
    assert first.tenant_id == "tenant-a"
    assert first.instance_id == "tenant-a"


def test_identity_fallbacks_when_sip_missing():
    settings = LiveKitSettings(_env_file=None, restaurant_id="dev-tenant")
    identity = extract_call_identity(settings, room_name="room-1", sip_attributes={})
    assert identity.call_id == "room-1"
    assert identity.session_id == "room-1"
    assert identity.restaurant_id == "dev-tenant"
