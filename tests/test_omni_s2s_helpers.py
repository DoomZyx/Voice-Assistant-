from prototypes.omni_s2s.helpers import (
    FRENCH_CASES,
    OFFICIAL_AUDIO_SYSTEM_PROMPT,
    build_conversation,
    compatibility_audit,
    decide_integration,
    fake_fastify,
    looks_like_french,
    parse_tool_calls,
    percentile,
    reply_quality_flags,
    score_french_case,
    talker_user_text,
)
from prototypes.omni_s2s.proto_s2s import build_report


def test_parse_tool_call_block():
    text = (
        'Avant <tool_call>{"name":"check_availability","arguments":{"date":"2026-09-20"}}'
        "</tool_call> apres"
    )
    calls = parse_tool_calls(text)
    assert calls == [
        {"name": "check_availability", "arguments": {"date": "2026-09-20"}}
    ]


def test_parse_bare_json_tool_call():
    text = 'OK {"name":"create_appointment","arguments":{"name":"Marie","telephone":"06"}}'
    calls = parse_tool_calls(text)
    assert calls[0]["name"] == "create_appointment"
    assert calls[0]["arguments"]["name"] == "Marie"


def test_parse_tool_calls_rejects_invalid_json():
    assert parse_tool_calls("<tool_call>{nope}</tool_call>") == []
    assert parse_tool_calls("") == []


def test_build_conversation_keeps_official_system_prompt():
    history = ({"role": "user", "content": [{"type": "text", "text": "hi"}]},)
    first = build_conversation(user_text="Bonsoir", history=history)
    second = build_conversation(audio_path="/tmp/a.wav", include_tools=True)
    assert first[0]["content"][0]["text"] == OFFICIAL_AUDIO_SYSTEM_PROMPT
    assert first[-1]["content"][0]["text"] == "Bonsoir"
    assert history[0]["content"][0]["text"] == "hi"
    assert second[-1]["content"][0]["type"] == "text"
    assert second[-1]["content"][1]["audio"] == "/tmp/a.wav"


def test_french_and_quality_flags():
    flags = reply_quality_flags("Oui, pour combien de personnes ?")
    assert flags["looks_french"] is True
    assert flags["empty"] is False
    assert looks_like_french("Hello there") is False


def test_french_case_and_tool_loop_do_not_speak_json():
    case = FRENCH_CASES[0]
    score = score_french_case(
        case,
        "Table pour quatre ce soir a vingt heures, c'est note.",
        "Oui, pour quatre personnes a vingt heures.",
    )
    assert score["acceptable"] is True
    result = fake_fastify({"name": "check_availability", "arguments": {"date": "2026-09-19"}})
    spoken = talker_user_text(result["spoken"])
    assert result["ok"] is True
    assert "{" not in result["spoken"]
    assert "check_availability" not in result["spoken"]
    assert "JSON" in spoken
    go = decide_integration(
        oom=False,
        french_quality="acceptable",
        tool_call_success=True,
        output_audio_ok=True,
        thinker_p50_ms=1800,
        total_p50_ms=5200,
    )
    assert go["integration_allowed"] is True


def test_percentile_and_dry_report_block_integration():
    assert percentile([], 0.5) is None
    assert percentile([10.0], 0.95) == 10.0
    assert percentile([10.0, 20.0, 30.0, 40.0], 0.5) == 25.0
    report = build_report(
        {"status": "not_measured", "reason": "dry-run", "hardware": {"available": False}},
        None,
    )
    audit = compatibility_audit()
    assert report["go_nogo"]["integration_allowed"] is False
    assert audit["vram"]["l4_fits_bf16_7b"] is False
    assert audit["vram"]["l4_fits_int4_15s_video"] is True
    assert audit["function_calling"]["local_generate"]
